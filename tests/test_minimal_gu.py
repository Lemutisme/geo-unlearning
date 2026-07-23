import copy
import os

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
        "projection_eps": 1.0e-12,
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
    **kwargs,
):
    args = TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=True,
        report_to=[],
        learning_rate=1.0e-3,
        optim="adamw_torch",
        weight_decay=0.0,
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


def test_gu_anchors_diagnostics_open_when_parent_is_temporarily_swapped(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir()
    parked_diagnostics_dir = output_dir / "parked-diagnostics"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = os.open
    swapped = False

    def swap_parent_on_final_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (
            not swapped
            and os.path.basename(os.fspath(path)) == "gu.jsonl"
            and flags & os.O_CREAT
        ):
            diagnostics_dir.rename(parked_diagnostics_dir)
            diagnostics_dir.symlink_to(outside, target_is_directory=True)
            try:
                file_descriptor = original_open(
                    path,
                    flags,
                    mode,
                    dir_fd=dir_fd,
                )
            finally:
                diagnostics_dir.unlink()
                parked_diagnostics_dir.rename(diagnostics_dir)
            swapped = True
            return file_descriptor
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_parent_on_final_open)
    trainer = make_trainer(
        TinyCausalLM(),
        output_dir,
        gu=gu_config(diagnostics_path="diagnostics/gu.jsonl"),
    )

    initialization_error = None
    try:
        trainer.create_optimizer()
    except Exception as error:
        initialization_error = error

    assert swapped
    assert not (outside / "gu.jsonl").exists()
    assert initialization_error is None
    assert (diagnostics_dir / "gu.jsonl").is_file()


def test_gu_exclusive_commit_rejects_and_preserves_concurrent_file(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "output"
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True)
    diagnostics_path = diagnostics_dir / "gu.jsonl"
    original_open = os.open
    concurrent_contents = b"concurrent actor\n"
    created_concurrently = False

    def create_file_before_diagnostics_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal created_concurrently
        if (
            not created_concurrently
            and os.path.basename(os.fspath(path)) == diagnostics_path.name
            and flags & os.O_CREAT
            and flags & os.O_EXCL
        ):
            concurrent_descriptor = original_open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode,
                dir_fd=dir_fd,
            )
            try:
                os.write(concurrent_descriptor, concurrent_contents)
            finally:
                os.close(concurrent_descriptor)
            created_concurrently = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", create_file_before_diagnostics_open)
    trainer = make_trainer(
        TinyCausalLM(),
        output_dir,
        gu=gu_config(diagnostics_path="diagnostics/gu.jsonl"),
    )

    with pytest.raises(ValueError, match="append-writable"):
        trainer.create_optimizer()

    assert created_concurrently
    assert diagnostics_path.read_bytes() == concurrent_contents


def test_gu_opens_final_diagnostics_with_nofollow_append_create(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "output"
    diagnostics_path = output_dir / "diagnostics" / "gu.jsonl"
    opened_calls = []
    original_open = os.open

    def track_open(path, flags, mode=0o777, *, dir_fd=None):
        if (
            os.path.basename(os.fspath(path)) == diagnostics_path.name
            and flags & os.O_CREAT
        ):
            opened_calls.append((flags, dir_fd))
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", track_open)
    trainer = make_trainer(
        TinyCausalLM(),
        output_dir,
        gu=gu_config(diagnostics_path="diagnostics/gu.jsonl"),
    )

    trainer.create_optimizer()

    assert opened_calls
    opened_flags, parent_descriptor = opened_calls[-1]
    assert parent_descriptor is not None
    assert opened_flags & os.O_WRONLY
    assert opened_flags & os.O_NOFOLLOW
    assert opened_flags & os.O_APPEND
    assert opened_flags & os.O_CREAT
    assert opened_flags & os.O_EXCL


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


def test_gu_uses_current_then_newest_bounded_constraint_history(tmp_path):
    torch.manual_seed(654)
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(retain_history_rank=2),
    )
    trainer.create_optimizer()
    observed_currents = []

    for offset in range(4):
        tokens = torch.tensor(
            [
                [1 + offset, 2 + offset, 3 + offset, 4 + offset],
                [6 + offset, 7 + offset, 8 + offset, 9 + offset],
            ]
        ) % 11
        batch = {
            "forget": {"input_ids": tokens, "labels": tokens.clone()},
            "retain": {
                "input_ids": tokens.flip(dims=(1,)),
                "labels": torch.cat(
                    (
                        torch.full((2, 1), -100),
                        tokens.flip(dims=(1,))[:, 1:],
                    ),
                    dim=1,
                ),
            },
        }

        trainer.training_step(trainer.model, batch)

        current = tuple(
            block.clone() for block in trainer._gu_constraints_used[0]
        )
        observed_currents.append(current)
        expected = [current]
        expected.extend(reversed(observed_currents[:-1]))
        expected = expected[:3]
        assert len(trainer._gu_constraints_used) == len(expected)
        for actual_constraint, expected_constraint in zip(
            trainer._gu_constraints_used,
            expected,
        ):
            for actual_block, expected_block in zip(
                actual_constraint,
                expected_constraint,
            ):
                assert torch.equal(actual_block, expected_block)


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


def test_gu_rejects_shape_mismatched_retain_gradient(
    tmp_path,
    monkeypatch,
):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    monkeypatch.setattr(
        torch.autograd,
        "grad",
        lambda *args, **kwargs: (torch.ones(1),),
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "forget": {"input_ids": tokens, "labels": tokens.clone()},
        "retain": {"input_ids": tokens, "labels": tokens.clone()},
    }

    with pytest.raises(ValueError, match="shape"):
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
                assert len(trainer._gu_constraint_microsteps) == 1
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
    assert trainer._gu_constraint_microsteps == []


def test_gu_rank_zero_keeps_current_constraint_without_history(tmp_path):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(retain_history_rank=0),
    )
    trainer.create_optimizer()
    currents = []

    for offset in range(2):
        tokens = torch.tensor([[1 + offset, 3 + offset, 5 + offset, 7 + offset]])
        batch = {
            "forget": {"input_ids": tokens, "labels": tokens.clone()},
            "retain": {
                "input_ids": tokens.flip(dims=(1,)),
                "labels": tokens.clone(),
            },
        }
        trainer.training_step(trainer.model, batch)
        assert len(trainer._gu_constraints_used) == 1
        assert trainer._gu_constraint_history == ()
        currents.append(trainer._gu_constraints_used[0][0].clone())

    assert not torch.equal(currents[0], currents[1])


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


def test_gu_parent_failure_clears_accumulated_microsteps(tmp_path, monkeypatch):
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
        assert len(trainer._gu_constraint_microsteps) == 1
    with pytest.raises(RuntimeError, match="configured objective failed"):
        with trainer.accelerator.accumulate(trainer.model):
            trainer.training_step(trainer.model, copy.deepcopy(batch))

    assert trainer._gu_constraint_microsteps == []
