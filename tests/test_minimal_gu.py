import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

from trainer.base import FinetuneTrainer
from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_batch_nll


class TinyCausalLM(nn.Module):
    main_input_name = "input_ids"

    def __init__(self, vocab_size=11, hidden_size=5):
        super().__init__()
        self.config = PretrainedConfig(vocab_size=vocab_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.protected = nn.Linear(hidden_size, vocab_size, bias=False)
        self.unmatched = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids, labels=None):
        logits = self.protected(self.embed(input_ids)) + self.unmatched
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits)


class TinyObjectiveTrainer(UnlearnTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        outputs = model(**inputs["forget"])
        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss


def gu_config(**overrides):
    config = {
        "enabled": True,
        "parameter_regex": ["protected[.]weight"],
        "retain_history_rank": 8,
        "projection_eps": 1.0e-6,
        "retain_filter": "first_order",
        "retain_budget": 1.0e-4,
        "backtracking_scales": [1.0, 0.5, 0.25, 0.125],
        "diagnostics_path": "gu_diagnostics.jsonl",
    }
    config.update(overrides)
    return config


def make_trainer(
    model,
    output_dir,
    gradient_accumulation_steps=1,
    adam_beta1=0.9,
    weight_decay=0.0,
    **kwargs,
):
    args = TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=True,
        report_to=[],
        learning_rate=1.0e-3,
        optim="adamw_torch",
        adam_beta1=adam_beta1,
        weight_decay=weight_decay,
        remove_unused_columns=False,
        disable_tqdm=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return TinyObjectiveTrainer(model=model, args=args, **kwargs)


def run_objective_update(trainer, batch):
    trainer.create_optimizer()
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    trainer.model.train()
    loss = trainer.training_step(trainer.model, batch)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    return loss


def assert_ordered_optimizer_state_equal(left, right):
    left_names = {parameter: name for name, parameter in left.model.named_parameters()}
    right_names = {
        parameter: name for name, parameter in right.model.named_parameters()
    }
    left_state = list(left.optimizer.state.items())
    right_state = list(right.optimizer.state.items())

    assert [left_names[parameter] for parameter, _ in left_state] == [
        right_names[parameter] for parameter, _ in right_state
    ]
    for (_, left_values), (_, right_values) in zip(left_state, right_state):
        assert list(left_values) == list(right_values)
        for key in left_values:
            left_value = left_values[key]
            right_value = right_values[key]
            if isinstance(left_value, torch.Tensor):
                assert torch.equal(left_value, right_value)
            else:
                assert left_value == right_value


def gu_batch(forget_tokens, retain_tokens):
    return {
        "forget": {
            "input_ids": forget_tokens,
            "labels": forget_tokens.clone(),
        },
        "retain": {
            "input_ids": retain_tokens,
            "labels": retain_tokens.clone(),
        },
    }


def install_constraints(trainer, constraints):
    trainer._gu_constraints_used = tuple(constraints)
    trainer._gu_pending_history_covector = constraints[0]
    trainer._gu_constraint_history = tuple(constraints[1:])


def coordinate_constraint(trainer, entries):
    blocks = [
        torch.zeros_like(parameter, dtype=torch.float32)
        for _, parameter in trainer._gu_selected
    ]
    for block_index, coordinate, value in entries:
        blocks[block_index].view(-1)[coordinate] = value
    return tuple(blocks)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"parameter_regex": []}, "parameter_regex"),
        ({"parameter_regex": "protected"}, "parameter_regex"),
        ({"parameter_regex": [""]}, "parameter_regex"),
        ({"parameter_regex": [1]}, "parameter_regex"),
        ({"parameter_regex": ["("]}, "parameter_regex"),
        ({"retain_history_rank": -1}, "retain_history_rank"),
        ({"retain_history_rank": 9}, "retain_history_rank"),
        ({"retain_history_rank": 1.5}, "retain_history_rank"),
        ({"retain_history_rank": True}, "retain_history_rank"),
        ({"projection_eps": 0.0}, "projection_eps"),
        ({"projection_eps": float("inf")}, "projection_eps"),
        ({"projection_eps": float("nan")}, "projection_eps"),
        ({"projection_eps": True}, "projection_eps"),
        ({"retain_filter": "quadratic"}, "retain_filter"),
        ({"retain_budget": -1.0}, "retain_budget"),
        ({"retain_budget": float("inf")}, "retain_budget"),
        ({"retain_budget": float("nan")}, "retain_budget"),
        ({"retain_budget": True}, "retain_budget"),
        ({"backtracking_scales": []}, "backtracking_scales"),
        ({"backtracking_scales": "1.0"}, "backtracking_scales"),
        ({"backtracking_scales": [0.0]}, "backtracking_scales"),
        ({"backtracking_scales": [1.1]}, "backtracking_scales"),
        ({"backtracking_scales": [0.5, 1.0]}, "backtracking_scales"),
        ({"backtracking_scales": [1.0, 0.5, 0.5]}, "backtracking_scales"),
        ({"backtracking_scales": [1.0, float("inf")]}, "backtracking_scales"),
        ({"backtracking_scales": [1.0, float("nan")]}, "backtracking_scales"),
        ({"backtracking_scales": [True]}, "backtracking_scales"),
        ({"diagnostics_path": ""}, "diagnostics_path"),
    ],
)
def test_gu_rejects_invalid_configuration(tmp_path, override, message):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(**override),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        trainer.create_optimizer()


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_gu_rejects_nonexact_configuration_keys(tmp_path, change):
    config = gu_config()
    if change == "missing":
        del config["projection_eps"]
    else:
        config["unexpected"] = True
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=config)

    with pytest.raises(ValueError, match="keys"):
        trainer.create_optimizer()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_enabled", "keys"),
        ("extra_disabled", "keys"),
        ("zero", "enabled.*bool"),
        ("none", "enabled.*bool"),
        ("empty", "enabled.*bool"),
    ],
)
def test_gu_validates_contract_before_disabled_branch(tmp_path, change, message):
    config = gu_config(enabled=False)
    if change == "missing_enabled":
        del config["enabled"]
    elif change == "extra_disabled":
        config["unexpected"] = True
    elif change == "zero":
        config["enabled"] = 0
    elif change == "none":
        config["enabled"] = None
    else:
        config["enabled"] = ""
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=config)

    with pytest.raises(ValueError, match=message):
        trainer.create_optimizer()


def test_gu_rejects_parameter_regex_without_a_match(tmp_path):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(parameter_regex=["does_not_exist"]),
    )

    with pytest.raises(ValueError, match="parameter_regex"):
        trainer.create_optimizer()


def test_gu_freezes_unmatched_parameters_and_optimizer_scope_is_exact(tmp_path):
    model = TinyCausalLM()
    trainer = make_trainer(model, tmp_path, gu=gu_config())

    trainer.create_optimizer()

    assert [name for name, _ in trainer._gu_selected] == ["protected.weight"]
    assert model.protected.weight.requires_grad
    assert not model.embed.weight.requires_grad
    assert not model.unmatched.requires_grad
    optimizer_parameters = [
        parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    ]
    selected_parameters = [parameter for _, parameter in trainer._gu_selected]
    assert len(optimizer_parameters) == len({id(p) for p in optimizer_parameters})
    assert {id(p) for p in optimizer_parameters} == {id(p) for p in selected_parameters}
    assert not trainer.optimizer.state


def test_gu_compiles_parameter_regex_only_during_first_setup(tmp_path):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())

    trainer.create_optimizer()
    compiled_patterns = trainer._gu_parameter_patterns
    trainer.create_optimizer()

    assert trainer._gu_parameter_patterns is compiled_patterns


def test_gu_rejects_sgd_optimizer(tmp_path):
    model = TinyCausalLM()
    optimizer = torch.optim.SGD([model.protected.weight], lr=1.0e-3)
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(),
        optimizers=(optimizer, None),
    )

    with pytest.raises(ValueError, match="AdamW"):
        trainer.create_optimizer()


def test_gu_rejected_optimizer_rolls_back_entire_setup(tmp_path):
    model = TinyCausalLM()
    model.embed.weight.requires_grad_(False)
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.SGD([model.protected.weight], lr=1.0e-3)
    output_dir = tmp_path / "output"
    diagnostics_path = output_dir / "diagnostics" / "gu.jsonl"
    trainer = make_trainer(
        model,
        output_dir,
        gu=gu_config(diagnostics_path="diagnostics/gu.jsonl"),
        optimizers=(optimizer, None),
    )

    with pytest.raises(ValueError, match="AdamW"):
        trainer.create_optimizer()

    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == original_requires_grad
    assert not diagnostics_path.exists()
    for attribute in (
        "_gu_selected",
        "_gu_parameter_patterns",
        "_gu_diagnostics_path",
        "_gu_setup_complete",
    ):
        assert not hasattr(trainer, attribute)


def test_gu_rejects_optimizer_parameters_outside_selected_scope(tmp_path):
    model = TinyCausalLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(),
        optimizers=(optimizer, None),
    )

    with pytest.raises(ValueError, match="exactly"):
        trainer.create_optimizer()


def test_gu_rejects_bitsandbytes_8bit_optimizer(tmp_path):
    bitsandbytes = pytest.importorskip("bitsandbytes")
    model = TinyCausalLM()
    optimizer = bitsandbytes.optim.PagedAdamW8bit([model.protected.weight])
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(),
        optimizers=(optimizer, None),
    )

    with pytest.raises(ValueError, match="32-bit paged AdamW"):
        trainer.create_optimizer()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("deepspeed", "DeepSpeed"),
        ("fsdp", "FSDP"),
        ("apex", "Apex"),
        ("fp16", "FP16"),
        ("world_size", "single process"),
        ("n_gpu", "single process"),
        ("reentrant", "reentrant"),
    ],
)
def test_gu_rejects_unsupported_runtime_modes(tmp_path, monkeypatch, mode, message):
    trainer = make_trainer(TinyCausalLM(), tmp_path / mode, gu=gu_config())
    if mode == "deepspeed":
        monkeypatch.setattr(trainer, "is_deepspeed_enabled", True)
    elif mode == "fsdp":
        monkeypatch.setattr(trainer, "is_fsdp_enabled", True)
    elif mode == "apex":
        monkeypatch.setattr(trainer, "use_apex", True)
    elif mode == "fp16":
        monkeypatch.setattr(trainer.args, "fp16", True)
    elif mode == "world_size":
        monkeypatch.setattr(
            type(trainer.args),
            "world_size",
            property(lambda _args: 2),
        )
    elif mode == "n_gpu":
        monkeypatch.setattr(trainer.args, "_n_gpu", 2)
    else:
        monkeypatch.setattr(trainer.args, "gradient_checkpointing", True)
        monkeypatch.setattr(trainer.args, "gradient_checkpointing_kwargs", None)

    with pytest.raises(ValueError, match=message):
        trainer.create_optimizer()


def test_gu_permits_nonreentrant_checkpointing_and_bf16(tmp_path, monkeypatch):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    monkeypatch.setattr(trainer.args, "gradient_checkpointing", True)
    monkeypatch.setattr(
        trainer.args,
        "gradient_checkpointing_kwargs",
        {"use_reentrant": False},
    )
    monkeypatch.setattr(trainer.args, "bf16", True)

    trainer.create_optimizer()

    assert isinstance(trainer.optimizer, torch.optim.AdamW)


def test_gu_creates_append_writable_diagnostics_path_under_output_dir(tmp_path):
    output_dir = tmp_path / "output"
    trainer = make_trainer(
        TinyCausalLM(),
        output_dir,
        gu=gu_config(diagnostics_path="diagnostics/gu.jsonl"),
    )

    trainer.create_optimizer()

    expected_path = (output_dir / "diagnostics/gu.jsonl").resolve()
    assert trainer._gu_diagnostics_path == expected_path
    assert expected_path.is_file()
    with expected_path.open("a", encoding="utf-8") as diagnostics_file:
        diagnostics_file.write("writable\n")


@pytest.mark.parametrize(
    "diagnostics_path",
    ["/tmp/gu.jsonl", "../gu.jsonl", "diagnostics/../../gu.jsonl"],
)
def test_gu_rejects_absolute_or_traversing_diagnostics_path(
    tmp_path,
    diagnostics_path,
):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path / "output",
        gu=gu_config(diagnostics_path=diagnostics_path),
    )

    with pytest.raises(ValueError, match="diagnostics_path"):
        trainer.create_optimizer()


def test_gu_rejects_symlink_diagnostics_directory(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output_dir / "linked").symlink_to(outside, target_is_directory=True)
    trainer = make_trainer(
        TinyCausalLM(),
        output_dir,
        gu=gu_config(diagnostics_path="linked/gu.jsonl"),
    )

    with pytest.raises(ValueError, match="symlink"):
        trainer.create_optimizer()


def test_gu_rejects_final_symlink_diagnostics_path(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("unchanged\n", encoding="utf-8")
    (output_dir / "gu.jsonl").symlink_to(outside)
    trainer = make_trainer(
        TinyCausalLM(),
        output_dir,
        gu=gu_config(diagnostics_path="gu.jsonl"),
    )

    with pytest.raises(ValueError, match="symlink"):
        trainer.create_optimizer()

    assert outside.read_text(encoding="utf-8") == "unchanged\n"


def test_gu_rejects_symlink_in_output_dir_ancestor(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(outside, target_is_directory=True)
    output_dir = linked_output / "nested"
    trainer = make_trainer(
        TinyCausalLM(),
        output_dir,
        gu=gu_config(diagnostics_path="gu.jsonl"),
    )

    with pytest.raises(ValueError, match="symlink"):
        trainer.create_optimizer()

    assert not (outside / "nested" / "gu.jsonl").exists()


def test_disabled_gu_matches_unmodified_objective_update(tmp_path):
    torch.manual_seed(123)
    control_model = TinyCausalLM()
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in control_model.named_parameters()
    }
    disabled_model = copy.deepcopy(control_model)
    batch = {
        "forget": {
            "input_ids": torch.tensor([[1, 4, 2, 8, 3], [7, 2, 6, 1, 5]]),
            "labels": torch.tensor([[1, 4, 2, 8, 3], [7, 2, 6, 1, 5]]),
        }
    }
    control = make_trainer(control_model, tmp_path / "control")
    disabled = make_trainer(
        disabled_model,
        tmp_path / "disabled",
        gu=gu_config(enabled=False),
    )

    control_loss = run_objective_update(control, copy.deepcopy(batch))
    disabled_loss = run_objective_update(disabled, copy.deepcopy(batch))

    assert any(
        not torch.equal(initial_parameters[name], parameter)
        for name, parameter in control.model.named_parameters()
    ), "control update was a no-op"
    assert torch.equal(control_loss, disabled_loss)
    control_parameters = list(control.model.named_parameters())
    disabled_parameters = list(disabled.model.named_parameters())
    assert [name for name, _ in control_parameters] == [
        name for name, _ in disabled_parameters
    ]
    for (_, control_parameter), (_, disabled_parameter) in zip(
        control_parameters,
        disabled_parameters,
    ):
        assert torch.equal(control_parameter, disabled_parameter)
    assert_ordered_optimizer_state_equal(control, disabled)


def test_gu_collects_normalized_answer_masked_retain_constraint(tmp_path):
    torch.manual_seed(321)
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    retain = {
        "input_ids": torch.tensor([[1, 4, 2, 8], [7, 2, 6, 1]]),
        "labels": torch.tensor([[-100, -100, 2, 8], [-100, 2, 6, 1]]),
    }
    batch = {
        "forget": {
            "input_ids": torch.tensor([[3, 1, 5, 9], [8, 4, 2, 6]]),
            "labels": torch.tensor([[3, 1, 5, 9], [8, 4, 2, 6]]),
        },
        "retain": retain,
    }
    retain_nll, _ = compute_batch_nll(trainer.model, retain)
    expected_gradient = torch.autograd.grad(
        retain_nll.mean(),
        trainer._gu_selected[0][1],
    )[0].detach()
    expected_gradient /= expected_gradient.double().square().sum().sqrt()

    trainer.training_step(trainer.model, batch)

    assert len(trainer._gu_constraints_used) == 1
    (current_constraint,) = trainer._gu_constraints_used
    assert len(current_constraint) == 1
    assert current_constraint[0].dtype == torch.float32
    assert not current_constraint[0].requires_grad
    assert torch.allclose(current_constraint[0], expected_gradient, atol=1.0e-6)


def test_gu_retain_and_objective_forwards_run_in_train_mode(tmp_path, monkeypatch):
    model = TinyCausalLM()
    trainer = make_trainer(model, tmp_path, gu=gu_config())
    trainer.create_optimizer()
    forward_modes = []
    optimizer_train_calls = []
    original_forward = model.forward

    def record_forward_mode(*args, **kwargs):
        forward_modes.append(model.training)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", record_forward_mode)
    monkeypatch.setattr(
        trainer.optimizer,
        "train",
        lambda: optimizer_train_calls.append(None),
        raising=False,
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
        "retain": {"input_ids": tokens.flip(dims=(1,)), "labels": tokens.clone()},
    }
    model.eval()

    trainer.training_step(model, batch)

    assert forward_modes == [True, True]
    assert optimizer_train_calls == [None]


def test_gu_exposes_pending_current_without_advancing_history(tmp_path):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(retain_history_rank=2),
    )
    trainer.create_optimizer()
    prior = (torch.ones_like(trainer._gu_selected[0][1]),)
    history = (prior,)
    trainer._gu_constraint_history = history
    tokens = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
        "retain": {"input_ids": tokens.flip(dims=(1,)), "labels": tokens.clone()},
    }

    trainer.training_step(trainer.model, batch)

    current, used_prior = trainer._gu_constraints_used
    assert used_prior is prior
    assert trainer._gu_pending_history_covector is current
    assert trainer._gu_constraint_history is history


@pytest.mark.parametrize(
    "retain",
    [pytest.param(None, id="missing"), pytest.param([], id="not-mapping")],
)
def test_gu_requires_retain_mapping(tmp_path, retain):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    tokens = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
    }
    if retain is not None:
        batch["retain"] = retain

    with pytest.raises(ValueError, match="retain.*mapping"):
        trainer.training_step(trainer.model, batch)


def test_gu_rejects_unused_selected_retain_gradient(tmp_path, monkeypatch):
    model = TinyCausalLM()
    trainer = make_trainer(model, tmp_path, gu=gu_config())
    trainer.create_optimizer()
    model.embed.weight.requires_grad_(True)

    def forward_without_selected_parameter(input_ids, labels=None):
        del labels
        hidden = model.embed(input_ids)
        logits = hidden.mean(dim=-1, keepdim=True).expand(-1, -1, 11)
        return CausalLMOutput(logits=logits)

    monkeypatch.setattr(model, "forward", forward_without_selected_parameter)
    tokens = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
        "retain": {"input_ids": tokens, "labels": tokens.clone()},
    }

    with pytest.raises(ValueError, match="unused"):
        trainer.training_step(trainer.model, batch)


def test_gu_rejects_nonfinite_retain_gradient(tmp_path):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    selected_parameter = trainer._gu_selected[0][1]
    hook = selected_parameter.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
        "retain": {"input_ids": tokens, "labels": tokens.clone()},
    }

    try:
        with pytest.raises(ValueError, match="finite"):
            trainer.training_step(trainer.model, batch)
    finally:
        hook.remove()


def test_gu_rejects_zero_global_retain_norm(tmp_path):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    selected_parameter = trainer._gu_selected[0][1]
    hook = selected_parameter.register_hook(torch.zeros_like)
    tokens = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
        "retain": {"input_ids": tokens, "labels": tokens.clone()},
    }

    try:
        with pytest.raises(ValueError, match="zero"):
            trainer.training_step(trainer.model, batch)
    finally:
        hook.remove()


def test_gu_accumulates_effective_batch_then_uses_one_global_block_norm(
    tmp_path,
):
    torch.manual_seed(987)
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gradient_accumulation_steps=2,
        gu=gu_config(
            parameter_regex=["embed[.]weight", "protected[.]weight"],
            retain_history_rank=0,
        ),
    )
    trainer.create_optimizer()
    retain_batches = [
        {
            "input_ids": torch.tensor([[1, 4, 2, 8], [7, 2, 6, 1]]),
            "labels": torch.tensor([[-100, -100, 2, 8], [-100, 2, 6, 1]]),
        },
        {
            "input_ids": torch.tensor([[3, 9, 5, 2], [6, 8, 4, 1]]),
            "labels": torch.tensor([[-100, 9, 5, 2], [-100, -100, 4, 1]]),
        },
    ]
    selected_parameters = tuple(
        parameter for _, parameter in trainer._gu_selected
    )
    microstep_gradients = []
    for retain in retain_batches:
        retain_nll, _ = compute_batch_nll(trainer.model, retain)
        microstep_gradients.append(
            torch.autograd.grad(retain_nll.mean(), selected_parameters)
        )
    expected_blocks = tuple(
        (
            microstep_gradients[0][index]
            + microstep_gradients[1][index]
        ).detach()
        / 2
        for index in range(len(selected_parameters))
    )
    expected_norm = sum(
        block.double().square().sum() for block in expected_blocks
    ).sqrt()
    expected_blocks = tuple(
        (block / expected_norm).to(torch.float32)
        for block in expected_blocks
    )

    for index, retain in enumerate(retain_batches):
        forget_tokens = torch.tensor(
            [[1 + index, 3 + index, 5 + index, 7 + index]]
        )
        batch = {
            "forget": {
                "input_ids": forget_tokens,
                "labels": forget_tokens.clone(),
            },
            "retain": retain,
        }
        with trainer.accelerator.accumulate(trainer.model):
            trainer.training_step(trainer.model, batch)
            if index == 0:
                assert not trainer.accelerator.sync_gradients
                assert len(trainer._gu_constraint_accumulator) == 2
                assert not hasattr(trainer, "_gu_constraints_used")
            else:
                assert trainer.accelerator.sync_gradients

    (current_constraint,) = trainer._gu_constraints_used
    assert len(current_constraint) == 2
    for actual_block, expected_block in zip(
        current_constraint,
        expected_blocks,
    ):
        assert torch.allclose(actual_block, expected_block, atol=1.0e-6)
    block_norms = [
        block.double().square().sum().sqrt().item()
        for block in current_constraint
    ]
    assert all(0.0 < block_norm < 1.0 for block_norm in block_norms)
    global_norm = sum(
        block.double().square().sum() for block in current_constraint
    ).sqrt()
    assert global_norm.item() == pytest.approx(1.0, abs=1.0e-6)
    assert not hasattr(trainer, "_gu_constraint_accumulator")


def test_gu_reuses_fixed_fp32_accumulator_blocks_without_stack(
    tmp_path,
    monkeypatch,
):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gradient_accumulation_steps=3,
        gu=gu_config(
            parameter_regex=["embed[.]weight", "protected[.]weight"],
        ),
    )
    trainer.create_optimizer()
    tokens = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
        "retain": {"input_ids": tokens.flip(dims=(1,)), "labels": tokens.clone()},
    }
    monkeypatch.setattr(
        torch,
        "stack",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("GU accumulation must not call torch.stack")
        ),
    )

    with trainer.accelerator.accumulate(trainer.model):
        trainer.training_step(trainer.model, copy.deepcopy(batch))
        accumulator = trainer._gu_constraint_accumulator
        block_ids = tuple(id(block) for block in accumulator)
        first_values = tuple(block.clone() for block in accumulator)
        assert all(block.dtype == torch.float32 for block in accumulator)
        assert all(not block.requires_grad for block in accumulator)
    with trainer.accelerator.accumulate(trainer.model):
        trainer.training_step(trainer.model, copy.deepcopy(batch))
        assert trainer._gu_constraint_accumulator is accumulator
        assert (
            tuple(id(block) for block in trainer._gu_constraint_accumulator)
            == block_ids
        )
        for block, first_value in zip(accumulator, first_values):
            assert torch.allclose(block, 2 * first_value)
    with trainer.accelerator.accumulate(trainer.model):
        trainer.training_step(trainer.model, copy.deepcopy(batch))

    assert not hasattr(trainer, "_gu_constraint_accumulator")


def test_gu_rank_zero_keeps_current_constraint_without_history(tmp_path):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(retain_history_rank=0),
    )
    trainer.create_optimizer()
    history = ()
    trainer._gu_constraint_history = history
    tokens = torch.tensor([[1, 3, 5, 7]])
    batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
        "retain": {"input_ids": tokens.flip(dims=(1,)), "labels": tokens.clone()},
    }

    trainer.training_step(trainer.model, batch)

    (current,) = trainer._gu_constraints_used
    assert trainer._gu_pending_history_covector is current
    assert trainer._gu_constraint_history is history


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(None, id="absent"),
        pytest.param(gu_config(enabled=False), id="disabled"),
        pytest.param({}, id="legacy-inert"),
    ],
)
def test_gu_inert_modes_delegate_inputs_and_result_unchanged(
    tmp_path,
    monkeypatch,
    config,
):
    sentinel = object()
    inputs = {"opaque": object()}
    calls = []

    def parent_training_step(self, model, parent_inputs):
        calls.append((self, model, parent_inputs))
        return sentinel

    monkeypatch.setattr(FinetuneTrainer, "training_step", parent_training_step)
    kwargs = {} if config is None else {"gu": config}
    trainer = make_trainer(TinyCausalLM(), tmp_path, **kwargs)

    result = trainer.training_step(trainer.model, inputs)

    assert result is sentinel
    assert calls == [(trainer, trainer.model, inputs)]


def test_gu_exception_before_parent_clears_accumulator_and_pending_state(tmp_path):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gradient_accumulation_steps=2,
        gu=gu_config(),
    )
    trainer.create_optimizer()
    tokens = torch.tensor([[1, 2, 3, 4]])
    valid_batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
        "retain": {"input_ids": tokens.flip(dims=(1,)), "labels": tokens.clone()},
    }

    with trainer.accelerator.accumulate(trainer.model):
        trainer.training_step(trainer.model, valid_batch)
        assert hasattr(trainer, "_gu_constraint_accumulator")
    trainer._gu_constraints_used = object()
    trainer._gu_pending_history_covector = object()
    with pytest.raises(ValueError, match="retain.*mapping"):
        with trainer.accelerator.accumulate(trainer.model):
            trainer.training_step(
                trainer.model,
                {"forget": valid_batch["forget"]},
            )

    assert not hasattr(trainer, "_gu_constraint_accumulator")
    assert not hasattr(trainer, "_gu_constraints_used")
    assert not hasattr(trainer, "_gu_pending_history_covector")


def test_gu_parent_failure_clears_accumulator_and_pending_state(
    tmp_path,
    monkeypatch,
):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gradient_accumulation_steps=2,
        gu=gu_config(),
    )
    trainer.create_optimizer()
    original_compute_loss = trainer.compute_loss
    parent_calls = 0

    def failing_second_compute_loss(
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        nonlocal parent_calls
        parent_calls += 1
        if parent_calls == 2:
            raise RuntimeError("configured objective failed")
        return original_compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    monkeypatch.setattr(trainer, "compute_loss", failing_second_compute_loss)
    tokens = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
        "retain": {"input_ids": tokens.flip(dims=(1,)), "labels": tokens.clone()},
    }

    with trainer.accelerator.accumulate(trainer.model):
        trainer.training_step(trainer.model, copy.deepcopy(batch))
        assert hasattr(trainer, "_gu_constraint_accumulator")
    trainer._gu_constraints_used = object()
    trainer._gu_pending_history_covector = object()
    with pytest.raises(RuntimeError, match="configured objective failed"):
        with trainer.accelerator.accumulate(trainer.model):
            trainer.training_step(trainer.model, copy.deepcopy(batch))

    assert not hasattr(trainer, "_gu_constraint_accumulator")
    assert not hasattr(trainer, "_gu_constraints_used")
    assert not hasattr(trainer, "_gu_pending_history_covector")


def test_gu_constrains_actual_multi_active_adamw_delta_and_reports_kkt(tmp_path):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(retain_history_rank=2, projection_eps=1.0e-12),
    )
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    constraints = tuple(
        coordinate_constraint(trainer, [(0, coordinate, -1.0)])
        for coordinate in range(3)
    )
    install_constraints(trainer, constraints)
    before = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)

    trainer.optimizer.step()

    actual_delta = parameter.detach().float() - before.float()
    for constraint in constraints:
        violation = sum(
            (block.double() * actual_delta.double()).sum()
            for block in constraint
        )
        assert violation.item() <= 1.0e-7
    diagnostics = trainer.gu_last_diagnostics
    assert diagnostics["constraint_count"] == 3
    assert diagnostics["active_constraints"] == [0, 1, 2]
    assert diagnostics["max_violation_before"] > 0.0
    assert diagnostics["max_violation_after"] <= 1.0e-12
    assert diagnostics["kkt_residual"] <= 1.0e-12
    assert diagnostics["projection_tolerance"] == 1.0e-12
    assert trainer.gu_projection_calls == 1
    assert trainer._gu_constraint_history == constraints[:2]
    for attribute in (
        "_gu_constraints_used",
        "_gu_pending_history_covector",
        "_gu_parameter_snapshot",
        "_gu_proposal_delta",
    ):
        assert not hasattr(trainer, attribute)


def test_gu_rejects_violating_delta_after_parameter_dtype_cast(
    tmp_path,
    monkeypatch,
):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(
            parameter_regex=["embed[.]weight", "protected[.]weight"],
            retain_history_rank=1,
        ),
    )
    trainer.create_optimizer()
    constraint = coordinate_constraint(
        trainer,
        [(0, 0, -1.0), (1, 0, -1.0)],
    )
    install_constraints(trainer, (constraint,))
    history = trainer._gu_constraint_history
    snapshots = tuple(
        parameter.detach().clone() for _, parameter in trainer._gu_selected
    )
    for _, parameter in trainer._gu_selected:
        parameter.grad = torch.ones_like(parameter)

    original_to = torch.Tensor.to
    correction_casts = 0

    def drop_second_parameter_correction(tensor, *args, **kwargs):
        nonlocal correction_casts
        result = original_to(tensor, *args, **kwargs)
        if tensor.dtype == torch.float64 and args == (torch.float32,):
            correction_casts += 1
            if correction_casts == 2:
                return torch.zeros_like(result)
        return result

    monkeypatch.setattr(torch.Tensor, "to", drop_second_parameter_correction)

    with pytest.raises(ValueError, match="applied"):
        trainer.optimizer.step()

    assert correction_casts == 2
    for (_, parameter), snapshot in zip(trainer._gu_selected, snapshots):
        assert torch.equal(parameter, snapshot)
        assert trainer.optimizer.state[parameter]["step"].item() == 1
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    assert trainer.gu_last_diagnostics is None


def test_gu_safe_proposal_is_bitwise_unchanged(tmp_path):
    model = TinyCausalLM()
    trainer = make_trainer(model, tmp_path, gu=gu_config())
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    expected = nn.Parameter(parameter.detach().clone())
    group = trainer.optimizer.param_groups[0]
    control = torch.optim.AdamW(
        [expected],
        lr=group["lr"],
        betas=group["betas"],
        eps=group["eps"],
        weight_decay=group["weight_decay"],
    )
    constraint = coordinate_constraint(trainer, [(0, 0, 1.0)])
    install_constraints(trainer, (constraint,))
    parameter.grad = torch.ones_like(parameter)
    expected.grad = torch.ones_like(expected)

    control.step()
    trainer.optimizer.step()

    assert torch.equal(parameter, expected)
    diagnostics = trainer.gu_last_diagnostics
    assert diagnostics["active_constraints"] == []
    assert diagnostics["correction_ratio"] == 0.0
    assert diagnostics["proposal_norm"] == diagnostics["corrected_norm"]
    assert not hasattr(trainer, "_gu_proposal_delta")


def test_gu_dependent_constraints_use_feasible_nonsingular_active_set(tmp_path):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    constraint = coordinate_constraint(trainer, [(0, 0, -1.0)])
    dependent = tuple(block.clone() for block in constraint)
    constraints = (constraint, dependent)
    install_constraints(trainer, constraints)
    before = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)

    trainer.optimizer.step()

    actual_delta = parameter.detach().float() - before.float()
    assert all(
        sum(
            (block.double() * actual_delta.double()).sum()
            for block in candidate
        ).item()
        <= 1.0e-7
        for candidate in constraints
    )
    assert trainer.gu_last_diagnostics["active_constraints"] == [0]
    assert trainer.gu_last_diagnostics["kkt_residual"] <= 1.0e-8


def test_gu_uses_one_global_dual_coefficient_across_parameter_blocks(tmp_path):
    model = TinyCausalLM()
    parameters = (model.embed.weight, model.protected.weight)
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3)
    proposal_parameters = []

    def capture_proposal(_optimizer, _args, _kwargs):
        proposal_parameters[:] = [
            parameter.detach().clone() for parameter in parameters
        ]

    optimizer.register_step_post_hook(capture_proposal)
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(parameter_regex=["embed[.]weight", "protected[.]weight"]),
        optimizers=(optimizer, None),
    )
    trainer.create_optimizer()
    constraint = coordinate_constraint(
        trainer,
        [(0, 0, -1.0), (1, 0, -1.0)],
    )
    install_constraints(trainer, (constraint,))
    before = [parameter.detach().clone() for parameter in parameters]
    parameters[0].grad = torch.ones_like(parameters[0])
    parameters[1].grad = torch.full_like(parameters[1], 2.0)

    optimizer.step()

    coefficients = []
    for parameter, proposed in zip(parameters, proposal_parameters):
        metric = optimizer.state[parameter]["exp_avg_sq"].double().sqrt()
        metric.add_(optimizer.param_groups[0]["eps"])
        adjustment = parameter.detach().double() - proposed.double()
        coefficients.append((adjustment.view(-1)[0] * metric.view(-1)[0]).item())
    assert coefficients[0] == pytest.approx(coefficients[1], rel=1.0e-4)
    applied = [
        parameter.detach().float() - old.float()
        for parameter, old in zip(parameters, before)
    ]
    violation = sum(
        (block.double() * delta.double()).sum()
        for block, delta in zip(constraint, applied)
    )
    assert violation.item() <= 1.0e-7


@pytest.mark.parametrize("solver_failure", ["rank", "negative", "nonfinite"])
def test_gu_infeasible_solver_result_does_not_correct_or_commit(
    tmp_path,
    monkeypatch,
    solver_failure,
):
    model = TinyCausalLM()
    parameter = model.protected.weight
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
    ordinary_proposal = []

    def capture_proposal(_optimizer, _args, _kwargs):
        ordinary_proposal.append(parameter.detach().clone())

    optimizer.register_step_post_hook(capture_proposal)
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(),
        optimizers=(optimizer, None),
    )
    trainer.create_optimizer()
    constraint = coordinate_constraint(trainer, [(0, 0, -1.0)])
    install_constraints(trainer, (constraint,))
    history = trainer._gu_constraint_history
    parameter.grad = torch.ones_like(parameter)
    real_lstsq = torch.linalg.lstsq

    def invalid_lstsq(matrix, right_hand_side, *, rcond):
        result = real_lstsq(matrix, right_hand_side, rcond=rcond)
        if solver_failure == "rank":
            return SimpleNamespace(
                solution=result.solution,
                rank=torch.zeros_like(result.rank),
            )
        fill = -1.0 if solver_failure == "negative" else float("nan")
        return SimpleNamespace(
            solution=torch.full_like(result.solution, fill),
            rank=result.rank,
        )

    monkeypatch.setattr(torch.linalg, "lstsq", invalid_lstsq)

    with pytest.raises(ValueError, match="feasible"):
        optimizer.step()

    assert len(ordinary_proposal) == 1
    assert torch.equal(parameter, ordinary_proposal[0])
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    for attribute in (
        "_gu_constraints_used",
        "_gu_pending_history_covector",
        "_gu_parameter_snapshot",
        "_gu_proposal_delta",
    ):
        assert not hasattr(trainer, attribute)


def test_gu_nonfinite_constraint_does_not_correct_or_commit(tmp_path):
    model = TinyCausalLM()
    parameter = model.protected.weight
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
    ordinary_proposal = []
    optimizer.register_step_post_hook(
        lambda _optimizer, _args, _kwargs: ordinary_proposal.append(
            parameter.detach().clone()
        )
    )
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(),
        optimizers=(optimizer, None),
    )
    trainer.create_optimizer()
    constraint = coordinate_constraint(trainer, [(0, 0, float("nan"))])
    install_constraints(trainer, (constraint,))
    history = trainer._gu_constraint_history
    parameter.grad = torch.ones_like(parameter)

    with pytest.raises(ValueError, match="finite"):
        optimizer.step()

    assert torch.equal(parameter, ordinary_proposal[0])
    assert trainer._gu_constraint_history is history
    assert not hasattr(trainer, "_gu_proposal_delta")


def test_gu_captures_full_realized_adamw_delta_without_persisting_metric(tmp_path):
    torch.manual_seed(2468)
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        adam_beta1=0.9,
        weight_decay=0.2,
        gu=gu_config(),
    )
    trainer.create_optimizer()
    assert not trainer.optimizer.state
    batches = [
        gu_batch(
            torch.tensor([[1, 4, 2, 8], [7, 2, 6, 1]]),
            torch.tensor([[3, 1, 5, 9], [8, 4, 2, 6]]),
        ),
        gu_batch(
            torch.tensor([[2, 5, 3, 9], [8, 3, 7, 2]]),
            torch.tensor([[4, 2, 6, 10], [9, 5, 3, 7]]),
        ),
    ]

    trainer.training_step(trainer.model, batches[0])
    assert not trainer.optimizer.state
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()

    before = tuple(
        parameter.detach().float().clone()
        for _, parameter in trainer._gu_selected
    )
    trainer.training_step(trainer.model, batches[1])
    raw_gradients = tuple(
        parameter.grad.detach().float().clone()
        for _, parameter in trainer._gu_selected
    )
    trainer.optimizer.step()

    observed = tuple(
        parameter.detach().float() - old
        for (_, parameter), old in zip(trainer._gu_selected, before)
    )
    assert any(
        not torch.allclose(
            actual,
            -trainer.args.learning_rate * raw_gradient,
        )
        for actual, raw_gradient in zip(observed, raw_gradients)
    )
    assert not hasattr(trainer, "_gu_proposal_delta")
    assert not hasattr(trainer, "_gu_metric_diagonal")


def test_gu_commits_history_only_after_post_hook_and_truncates_fifo(tmp_path):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(retain_history_rank=2),
    )
    trainer.create_optimizer()
    prior = (torch.ones_like(trainer._gu_selected[0][1]),)
    trainer._gu_constraint_history = (prior,)
    expected_history = (prior,)
    batches = [
        gu_batch(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[4, 3, 2, 1]]),
        ),
        gu_batch(
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[5, 4, 3, 2]]),
        ),
        gu_batch(
            torch.tensor([[3, 4, 5, 6]]),
            torch.tensor([[6, 5, 4, 3]]),
        ),
    ]

    committed = [prior]
    for batch in batches:
        trainer.training_step(trainer.model, batch)
        pending = trainer._gu_pending_history_covector
        assert trainer._gu_constraint_history == expected_history
        trainer._gu_constraint_accumulator = object()

        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

        committed.insert(0, pending)
        expected_history = tuple(committed[:2])
        assert trainer._gu_constraint_history == expected_history
        assert trainer._gu_constraint_history[0] is pending
        for attribute in (
            "_gu_constraint_accumulator",
            "_gu_constraints_used",
            "_gu_pending_history_covector",
            "_gu_parameter_snapshot",
        ):
            assert not hasattr(trainer, attribute)
        assert not hasattr(trainer, "_gu_proposal_delta")
        assert not hasattr(trainer, "_gu_metric_diagonal")


def test_gu_rank_zero_commits_empty_history_after_step(tmp_path):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(retain_history_rank=0),
    )
    trainer.create_optimizer()
    trainer.training_step(
        trainer.model,
        gu_batch(
            torch.tensor([[1, 3, 5, 7]]),
            torch.tensor([[7, 5, 3, 1]]),
        ),
    )

    trainer.optimizer.step()

    assert trainer._gu_constraint_history == ()


@pytest.mark.parametrize(
    ("prepared_state", "message"),
    [
        pytest.param("missing", "ready", id="missing"),
        pytest.param("stale", "stale", id="stale"),
        pytest.param("duplicate", "snapshot", id="duplicate-snapshot"),
    ],
)
def test_gu_pre_hook_rejects_unready_or_duplicate_state(
    tmp_path,
    prepared_state,
    message,
):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        weight_decay=0.2,
        gu=gu_config(),
    )
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    parameter.grad = torch.ones_like(parameter)
    before = parameter.detach().clone()
    covector = (torch.ones_like(parameter),)
    if prepared_state != "missing":
        trainer._gu_constraints_used = (covector,)
        trainer._gu_pending_history_covector = covector
    if prepared_state == "stale":
        trainer._gu_pending_history_covector = tuple(
            block.clone() for block in covector
        )
    elif prepared_state == "duplicate":
        trainer._gu_parameter_snapshot = (parameter.detach().float().clone(),)

    with pytest.raises(ValueError, match=message):
        trainer.optimizer.step()

    assert torch.equal(parameter, before)
    assert not trainer.optimizer.state


def test_gu_bf16_snapshot_uses_native_dtype_and_storage(tmp_path):
    model = TinyCausalLM().to(dtype=torch.bfloat16)
    trainer = make_trainer(model, tmp_path, gu=gu_config())
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    pending = (torch.ones_like(parameter),)
    trainer._gu_constraints_used = (pending,)
    trainer._gu_pending_history_covector = pending

    trainer._gu_optimizer_step_pre_hook(trainer.optimizer, (), {})

    (snapshot,) = trainer._gu_parameter_snapshot
    assert snapshot.dtype == torch.bfloat16
    assert snapshot.untyped_storage().nbytes() == (
        parameter.numel() * parameter.element_size()
    )


def test_gu_registers_bound_optimizer_hooks_exactly_once(tmp_path, monkeypatch):
    model = TinyCausalLM()
    optimizer = torch.optim.AdamW([model.protected.weight], lr=1.0e-3)
    pre_hooks = []
    post_hooks = []
    original_register_pre_hook = optimizer.register_step_pre_hook
    original_register_post_hook = optimizer.register_step_post_hook

    def register_pre_hook(hook):
        pre_hooks.append(hook)
        return original_register_pre_hook(hook)

    def register_post_hook(hook):
        post_hooks.append(hook)
        return original_register_post_hook(hook)

    monkeypatch.setattr(optimizer, "register_step_pre_hook", register_pre_hook)
    monkeypatch.setattr(optimizer, "register_step_post_hook", register_post_hook)
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(),
        optimizers=(optimizer, None),
    )

    trainer.create_optimizer()
    trainer.create_optimizer()

    assert len(pre_hooks) == 1
    assert len(post_hooks) == 1
    assert pre_hooks[0].__self__ is trainer
    assert pre_hooks[0].__func__ is UnlearnTrainer._gu_optimizer_step_pre_hook
    assert post_hooks[0].__self__ is trainer
    assert post_hooks[0].__func__ is UnlearnTrainer._gu_optimizer_step_post_hook


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        pytest.param("missing", "state", id="missing"),
        pytest.param("nan", "finite", id="nan"),
        pytest.param("negative", "nonnegative", id="negative"),
        pytest.param("shape", "shape", id="shape"),
        pytest.param("integer", "floating", id="integer"),
        pytest.param("metric", "metric diagonal", id="metric"),
    ],
)
def test_gu_invalid_second_moment_does_not_commit_or_rewind_proposal(
    tmp_path,
    corruption,
    message,
):
    model = TinyCausalLM()
    parameter = model.protected.weight
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3, weight_decay=0.2)
    after_ordinary_step = []

    def corrupt_state(stepped_optimizer, _args, _kwargs):
        after_ordinary_step.append(parameter.detach().clone())
        if corruption == "missing":
            stepped_optimizer.state.pop(parameter)
            return
        state = stepped_optimizer.state[parameter]
        if corruption == "nan":
            state["exp_avg_sq"].fill_(float("nan"))
        elif corruption == "negative":
            state["exp_avg_sq"].fill_(-1.0)
        elif corruption == "shape":
            state["exp_avg_sq"] = torch.zeros(1)
        elif corruption == "metric":
            stepped_optimizer.param_groups[0]["eps"] = float("inf")
        else:
            state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.int64)

    optimizer.register_step_post_hook(corrupt_state)
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(retain_history_rank=2),
        optimizers=(optimizer, None),
    )
    trainer.create_optimizer()
    prior = (torch.ones_like(parameter),)
    history = (prior,)
    trainer._gu_constraint_history = history
    trainer.training_step(
        trainer.model,
        gu_batch(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[4, 3, 2, 1]]),
        ),
    )

    with pytest.raises(ValueError, match=message):
        trainer.optimizer.step()

    assert len(after_ordinary_step) == 1
    assert torch.equal(parameter, after_ordinary_step[0])
    assert trainer._gu_constraint_history is history
    assert not hasattr(trainer, "_gu_proposal_delta")
    assert not hasattr(trainer, "_gu_metric_diagonal")
    for attribute in (
        "_gu_constraint_accumulator",
        "_gu_constraints_used",
        "_gu_pending_history_covector",
        "_gu_parameter_snapshot",
    ):
        assert not hasattr(trainer, attribute)


def test_gu_validates_state2_without_persisting_metric(tmp_path):
    model = TinyCausalLM()
    parameter = model.protected.weight
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3, eps=3.0e-7)

    def expose_state2(stepped_optimizer, _args, _kwargs):
        state = stepped_optimizer.state[parameter]
        state["state2"] = state.pop("exp_avg_sq")

    optimizer.register_step_post_hook(expose_state2)
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(),
        optimizers=(optimizer, None),
    )
    trainer.create_optimizer()
    trainer.training_step(
        trainer.model,
        gu_batch(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[4, 3, 2, 1]]),
        ),
    )
    pending = trainer._gu_pending_history_covector

    trainer.optimizer.step()

    assert torch.isfinite(optimizer.state[parameter]["state2"]).all()
    assert not hasattr(trainer, "_gu_metric_diagonal")
    assert trainer._gu_constraint_history[0] is pending


def test_gu_failed_adamw_step_does_not_commit_history(tmp_path):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    trainer.training_step(
        trainer.model,
        gu_batch(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[4, 3, 2, 1]]),
        ),
    )
    history = ()
    trainer._gu_constraint_history = history
    before = trainer._gu_selected[0][1].detach().clone()

    def failing_closure():
        raise RuntimeError("ordinary AdamW step failed")

    with pytest.raises(RuntimeError, match="ordinary AdamW step failed"):
        trainer.optimizer.step(failing_closure)

    assert trainer._gu_constraint_history is history
    assert torch.equal(trainer._gu_selected[0][1], before)
    assert not hasattr(trainer, "_gu_proposal_delta")
    assert not hasattr(trainer, "_gu_metric_diagonal")
    assert hasattr(trainer, "_gu_parameter_snapshot")


def test_disabled_gu_registers_no_hooks_or_transient_state(tmp_path, monkeypatch):
    model = TinyCausalLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    registrations = []
    original_register_pre_hook = optimizer.register_step_pre_hook
    original_register_post_hook = optimizer.register_step_post_hook

    def register_pre_hook(hook):
        registrations.append(("pre", hook))
        return original_register_pre_hook(hook)

    def register_post_hook(hook):
        registrations.append(("post", hook))
        return original_register_post_hook(hook)

    monkeypatch.setattr(optimizer, "register_step_pre_hook", register_pre_hook)
    monkeypatch.setattr(optimizer, "register_step_post_hook", register_post_hook)
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(enabled=False),
        optimizers=(optimizer, None),
    )

    trainer.create_optimizer()

    assert registrations == []
    for attribute in (
        "_gu_parameter_snapshot",
        "_gu_proposal_delta",
        "_gu_metric_diagonal",
        "_gu_constraint_history",
    ):
        assert not hasattr(trainer, attribute)
