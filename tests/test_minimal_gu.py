import copy
import inspect
import json
import math
import os
import re
import subprocess
from pathlib import Path
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


def test_gu_rejects_selected_float16_parameter_without_fp16_mode(tmp_path):
    model = TinyCausalLM()
    model.protected.to(dtype=torch.float16)
    trainer = make_trainer(model, tmp_path, gu=gu_config())
    assert trainer.args.fp16 is False

    with pytest.raises(ValueError, match="FP16"):
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


def test_gu_accumulates_unequal_microbatches_by_sequence_weight(
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
            "input_ids": torch.tensor([[3, 9, 5, 2]]),
            "labels": torch.tensor([[-100, 9, 5, 2]]),
        },
    ]
    selected_parameters = tuple(
        parameter for _, parameter in trainer._gu_selected
    )
    microstep_gradients = []
    for retain in retain_batches:
        retain_nll, _ = compute_batch_nll(trainer.model, retain)
        microstep_gradients.append(
            torch.autograd.grad(retain_nll.sum(), selected_parameters)
        )
    expected_blocks = tuple(
        (microstep_gradients[0][index] + microstep_gradients[1][index]).detach()
        for index in range(len(selected_parameters))
    )
    expected_norm = sum(
        block.double().square().sum() for block in expected_blocks
    ).sqrt()
    expected_blocks = tuple(
        (block / expected_norm).to(torch.float32)
        for block in expected_blocks
    )
    equal_microbatch_blocks = tuple(
        microstep_gradients[0][index] / 2 + microstep_gradients[1][index]
        for index in range(len(selected_parameters))
    )
    equal_microbatch_norm = sum(
        block.double().square().sum() for block in equal_microbatch_blocks
    ).sqrt()
    equal_microbatch_blocks = tuple(
        (block / equal_microbatch_norm).to(torch.float32)
        for block in equal_microbatch_blocks
    )
    assert any(
        not torch.allclose(sequence, equal, atol=1.0e-6)
        for sequence, equal in zip(expected_blocks, equal_microbatch_blocks)
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
        gu=gu_config(retain_filter="finite_step"),
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
    assert not hasattr(trainer, "_gu_retain_inputs")


def test_gu_parent_failure_clears_accumulator_and_pending_state(
    tmp_path,
    monkeypatch,
):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gradient_accumulation_steps=2,
        gu=gu_config(retain_filter="finite_step"),
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
    assert not hasattr(trainer, "_gu_retain_inputs")


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


def test_gu_rejects_primal_safe_dtype_cast_with_stationarity_error(
    tmp_path,
    monkeypatch,
):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        gu=gu_config(retain_history_rank=1),
    )
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    constraint = coordinate_constraint(trainer, [(0, 0, -1.0)])
    install_constraints(trainer, (constraint,))
    history = trainer._gu_constraint_history
    snapshot = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    original_to = torch.Tensor.to
    correction_casts = 0

    def perturb_unconstrained_correction(tensor, *args, **kwargs):
        nonlocal correction_casts
        result = original_to(tensor, *args, **kwargs)
        if tensor.dtype == torch.float64 and args == (torch.float32,):
            correction_casts += 1
            result = result.clone()
            result.view(-1)[1] += 1.0e-2
        return result

    monkeypatch.setattr(torch.Tensor, "to", perturb_unconstrained_correction)

    with pytest.raises(ValueError, match="stationarity"):
        trainer.optimizer.step()

    assert correction_casts == 1
    assert torch.equal(parameter, snapshot)
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    assert trainer.gu_last_diagnostics is None
    assert trainer._gu_diagnostics_path.read_text() == ""


@pytest.mark.parametrize(
    "corruption",
    ["missing_pending", "missing_constraints", "stale_pending", "stale_constraints"],
)
def test_gu_post_hook_entry_corruption_restores_snapshot_and_clears_state(
    tmp_path,
    corruption,
):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    pending = (torch.ones_like(parameter, dtype=torch.float32),)
    install_constraints(trainer, (pending,))
    history = trainer._gu_constraint_history
    trainer.optimizer.state[parameter]["exp_avg_sq"] = torch.ones_like(parameter)
    trainer._gu_optimizer_step_pre_hook(trainer.optimizer, (), {})
    snapshot = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1.0e-2)
    if corruption == "missing_pending":
        del trainer._gu_pending_history_covector
    elif corruption == "missing_constraints":
        del trainer._gu_constraints_used
    elif corruption == "stale_pending":
        trainer._gu_pending_history_covector = tuple(block.clone() for block in pending)
    else:
        trainer._gu_constraints_used = (
            tuple(block.clone() for block in pending),
        )

    with pytest.raises((AttributeError, TypeError, ValueError)):
        trainer._gu_optimizer_step_post_hook(trainer.optimizer, (), {})

    assert torch.equal(parameter, snapshot)
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    assert trainer.gu_last_diagnostics is None
    assert trainer._gu_diagnostics_path.read_text() == ""
    for attribute in (
        "_gu_constraints_used",
        "_gu_pending_history_covector",
        "_gu_parameter_snapshot",
    ):
        assert not hasattr(trainer, attribute)


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
def test_gu_infeasible_solver_result_restores_snapshot_without_commit(
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
    snapshot = parameter.detach().clone()
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
    assert not torch.equal(ordinary_proposal[0], snapshot)
    assert torch.equal(parameter, snapshot)
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    assert trainer.gu_last_diagnostics is None
    assert trainer._gu_diagnostics_path.read_text() == ""
    for attribute in (
        "_gu_constraints_used",
        "_gu_pending_history_covector",
        "_gu_parameter_snapshot",
        "_gu_proposal_delta",
    ):
        assert not hasattr(trainer, attribute)


def test_gu_nonfinite_constraint_restores_snapshot_without_commit(tmp_path):
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
    snapshot = parameter.detach().clone()

    with pytest.raises(ValueError, match="finite"):
        optimizer.step()

    assert not torch.equal(ordinary_proposal[0], snapshot)
    assert torch.equal(parameter, snapshot)
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    assert trainer.gu_last_diagnostics is None
    assert trainer._gu_diagnostics_path.read_text() == ""
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
def test_gu_invalid_second_moment_restores_snapshot_without_commit(
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
    snapshot = parameter.detach().clone()

    with pytest.raises(ValueError, match=message):
        trainer.optimizer.step()

    assert len(after_ordinary_step) == 1
    assert not torch.equal(after_ordinary_step[0], snapshot)
    assert torch.equal(parameter, snapshot)
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    assert trainer.gu_last_diagnostics is None
    assert trainer._gu_diagnostics_path.read_text() == ""
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


def test_finite_step_filters_all_gas_microbatches_by_sequence_count(tmp_path):
    torch.manual_seed(1703)
    model = TinyCausalLM()
    trainer = make_trainer(
        model,
        tmp_path,
        gradient_accumulation_steps=2,
        gu=gu_config(
            retain_filter="finite_step",
            retain_budget=1.0,
            backtracking_scales=[1.0, 0.5, 0.25],
        ),
    )
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    retain_batches = [
        {
            "input_ids": torch.tensor([[1, 4, 2, 8], [7, 2, 6, 1]]),
            "labels": torch.tensor([[-100, -100, 2, 8], [-100, 2, 6, 1]]),
        },
        {
            "input_ids": torch.tensor([[3, 9, 5, 2]]),
            "labels": torch.tensor([[-100, 9, 5, 2]]),
        },
    ]
    gradients = []
    for retain in retain_batches:
        nll, _ = compute_batch_nll(model, retain)
        gradients.append(torch.autograd.grad(nll.mean(), parameter)[0])
    direction = gradients[0] - (
        (gradients[0] * gradients[1]).sum() / gradients[1].square().sum()
    ) * gradients[1]
    delta = direction.detach() * (0.2 / direction.norm())
    snapshot = parameter.detach().clone()
    losses = []
    with torch.no_grad():
        for scale in (0.0, 1.0, 0.5):
            parameter.copy_(snapshot + scale * delta)
            losses.append([compute_batch_nll(model, batch)[0] for batch in retain_batches])
        parameter.copy_(snapshot)
    aggregate_changes = [
        torch.cat(candidate).mean().item() - torch.cat(losses[0]).mean().item()
        for candidate in losses[1:]
    ]
    microbatch_mean_change = sum(value.mean().item() for value in losses[1]) / 2 - sum(
        value.mean().item() for value in losses[0]
    ) / 2
    budget = (aggregate_changes[0] + microbatch_mean_change) / 2
    trainer.gu_config["retain_budget"] = budget
    assert losses[1][-1].mean().item() - losses[0][-1].mean().item() <= budget
    assert aggregate_changes[0] > budget
    assert microbatch_mean_change < budget
    assert aggregate_changes[1] <= budget

    for index, retain in enumerate(retain_batches):
        forget = torch.tensor([[1 + index, 3 + index, 5 + index, 7 + index]])
        with trainer.accelerator.accumulate(model):
            trainer.training_step(
                model,
                {
                    "forget": {"input_ids": forget, "labels": forget.clone()},
                    "retain": retain,
                },
            )
            if index == 0:
                assert isinstance(trainer._gu_retain_inputs, list)
                assert len(trainer._gu_retain_inputs) == 1
    assert isinstance(trainer._gu_retain_inputs, tuple)
    assert len(trainer._gu_retain_inputs) == 2

    pending = (-delta.to(torch.float32),)
    install_constraints(trainer, (pending,))
    trainer.optimizer.state[parameter]["exp_avg_sq"] = torch.ones_like(parameter)
    trainer._gu_optimizer_step_pre_hook(trainer.optimizer, (), {})
    with torch.no_grad():
        parameter.copy_(snapshot + delta)

    trainer._gu_optimizer_step_post_hook(trainer.optimizer, (), {})

    assert torch.allclose(parameter, snapshot + 0.5 * delta)
    assert trainer.gu_last_diagnostics["applied_scale"] == 0.5
    assert trainer.gu_last_diagnostics["retain_loss_before"] == pytest.approx(
        torch.cat(losses[0]).mean().item()
    )
    assert not hasattr(trainer, "_gu_retain_inputs")


def test_finite_step_uses_largest_actual_safe_scale_and_exact_record(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(1701)
    model = TinyCausalLM()
    model.unmatched.data = model.unmatched.data.to(torch.bfloat16)
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(
            retain_filter="finite_step",
            retain_budget=1.0,
            backtracking_scales=[1.0, 0.5, 0.25],
        ),
    )
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    retain = {
        "input_ids": torch.tensor([[1, 4, 2, 8], [7, 2, 6, 1]]),
        "labels": torch.tensor([[-100, -100, 2, 8], [-100, 2, 6, 1]]),
    }
    retain_loss, _ = compute_batch_nll(model, retain)
    gradient = torch.autograd.grad(retain_loss.mean(), parameter)[0]
    delta = gradient.detach() * (0.2 / gradient.norm())
    constraint = (-delta.to(torch.float32),)
    snapshot = parameter.detach().clone()

    model.eval()
    measured = []
    with torch.no_grad():
        for scale in (0.0, 1.0, 0.5, 0.25):
            parameter.copy_(snapshot + scale * delta)
            loss, _ = compute_batch_nll(model, retain)
            measured.append(loss.mean().item())
        parameter.copy_(snapshot)
    budget = (measured[1] + measured[2]) / 2 - measured[0]
    trainer.gu_config["retain_budget"] = budget
    assert measured[1] - measured[0] > budget
    assert measured[2] - measured[0] <= budget
    assert measured[3] - measured[0] <= budget

    model.forward_observations = []
    original_forward = model.forward

    def record_forward_output(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        model.forward_observations.append(
            (
                kwargs["input_ids"].detach().clone(),
                model.training,
                torch.is_autocast_enabled("cpu"),
                outputs.logits.dtype,
            )
        )
        return outputs

    monkeypatch.setattr(
        trainer,
        "compute_loss_context_manager",
        lambda: torch.autocast("cpu", dtype=torch.bfloat16),
    )
    monkeypatch.setattr(model, "forward", record_forward_output)
    forget = torch.tensor([[2, 5, 3, 9]])
    trainer.training_step(
        model,
        {
            "forget": {"input_ids": forget, "labels": forget.clone()},
            "retain": retain,
        },
    )
    install_constraints(trainer, (constraint,))
    trainer.optimizer.state[parameter]["exp_avg_sq"] = torch.ones_like(parameter)
    trainer._gu_optimizer_step_pre_hook(trainer.optimizer, (), {})
    with torch.no_grad():
        parameter.copy_(snapshot + delta)

    trainer._gu_optimizer_step_post_hook(trainer.optimizer, (), {})

    assert model.training
    assert [item[1] for item in model.forward_observations] == [
        True, True, False, False, False,
    ]
    assert all(item[2] for item in model.forward_observations)
    assert all(item[3] == torch.bfloat16 for item in model.forward_observations)
    assert torch.equal(model.forward_observations[0][0], retain["input_ids"])
    assert torch.equal(model.forward_observations[1][0], forget)
    assert all(
        torch.equal(item[0], retain["input_ids"])
        for item in model.forward_observations[2:]
    )
    assert torch.allclose(parameter, snapshot + 0.5 * delta)
    diagnostics = trainer.gu_last_diagnostics
    assert diagnostics["applied_scale"] == 0.5
    assert diagnostics["retain_loss_after"] - diagnostics["retain_loss_before"] <= budget
    actual_violation = (constraint[0].double() * (
        parameter.detach().float() - snapshot.float()
    ).double()).sum().item()
    assert diagnostics["max_violation_after"] == pytest.approx(actual_violation)
    assert diagnostics["max_violation_after"] <= diagnostics["projection_tolerance"]
    assert diagnostics["kkt_residual"] <= diagnostics["projection_tolerance"]
    expected_fields = {
        "step", "objective", "selected_parameter_count", "proposal_norm",
        "corrected_norm", "correction_ratio", "constraint_count",
        "active_constraints", "max_violation_before", "max_violation_after",
        "kkt_residual", "projection_tolerance", "applied_scale",
        "retain_loss_before", "retain_loss_after", "zero_step",
        "zero_step_reason", "optimizer_state_semantics", "projection_seconds",
        "filter_seconds",
    }
    records = [
        json.loads(line)
        for line in trainer._gu_diagnostics_path.read_text().splitlines()
    ]
    assert records == [diagnostics]
    assert set(diagnostics) == expected_fields
    assert diagnostics["optimizer_state_semantics"] == "proposal_state_committed"
    assert all(
        not isinstance(value, float) or math.isfinite(value)
        for value in diagnostics.values()
    )
    assert trainer.gu_projection_calls == 1
    assert not hasattr(trainer, "_gu_retain_inputs")


def test_finite_step_without_feasible_scale_commits_zero_step_history(tmp_path):
    torch.manual_seed(1702)
    model = TinyCausalLM()
    trainer = make_trainer(
        model,
        tmp_path,
        gu=gu_config(
            retain_filter="finite_step",
            retain_budget=0.0,
            backtracking_scales=[1.0, 0.5, 0.25],
        ),
    )
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    retain = {
        "input_ids": torch.tensor([[1, 3, 5, 7]]),
        "labels": torch.tensor([[-100, 3, 5, 7]]),
    }
    retain_loss, _ = compute_batch_nll(model, retain)
    gradient = torch.autograd.grad(retain_loss.mean(), parameter)[0]
    delta = gradient.detach() * (0.2 / gradient.norm())
    pending = (-delta.to(torch.float32),)
    install_constraints(trainer, (pending,))
    trainer._gu_retain_inputs = (
        {key: value.detach() for key, value in retain.items()},
    )
    snapshot = parameter.detach().clone()
    trainer.optimizer.state[parameter]["exp_avg_sq"] = torch.ones_like(parameter)
    trainer._gu_optimizer_step_pre_hook(trainer.optimizer, (), {})
    with torch.no_grad():
        parameter.copy_(snapshot + delta)

    trainer._gu_optimizer_step_post_hook(trainer.optimizer, (), {})

    assert torch.equal(parameter, snapshot)
    assert trainer.gu_last_diagnostics["applied_scale"] == 0.0
    assert trainer.gu_last_diagnostics["zero_step"] is True
    assert trainer.gu_last_diagnostics["zero_step_reason"] == "retain_budget_exceeded"
    assert trainer.gu_last_diagnostics["retain_loss_after"] == pytest.approx(
        trainer.gu_last_diagnostics["retain_loss_before"]
    )
    assert trainer._gu_constraint_history[0] is pending
    assert trainer.optimizer.state[parameter]["exp_avg_sq"].eq(1).all()
    assert trainer.gu_projection_calls == 1


def test_gu_reports_bf16_quantized_zero_applied_delta(tmp_path):
    model = TinyCausalLM()
    model.protected.to(dtype=torch.bfloat16)
    trainer = make_trainer(model, tmp_path, gu=gu_config())
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    pending = (torch.ones_like(parameter, dtype=torch.float32),)
    install_constraints(trainer, (pending,))
    trainer.optimizer.state[parameter]["exp_avg_sq"] = torch.ones_like(parameter)
    trainer._gu_optimizer_step_pre_hook(trainer.optimizer, (), {})
    snapshot = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(torch.full_like(parameter, 1.0e-8))
    assert torch.equal(parameter, snapshot)

    trainer._gu_optimizer_step_post_hook(trainer.optimizer, (), {})

    assert torch.equal(parameter, snapshot)
    assert trainer.gu_last_diagnostics["applied_scale"] == 1.0
    assert trainer.gu_last_diagnostics["zero_step"] is True
    assert trainer.gu_last_diagnostics["zero_step_reason"] == "quantized_zero"
    assert trainer._gu_constraint_history[0] is pending


def test_first_order_step_adds_no_retain_forward_and_writes_one_record(
    tmp_path,
    monkeypatch,
):
    model = TinyCausalLM()
    trainer = make_trainer(model, tmp_path, gu=gu_config())
    trainer.create_optimizer()
    calls = []
    original_forward = model.forward

    def record_forward(*args, **kwargs):
        calls.append(None)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", record_forward)
    trainer.training_step(
        model,
        gu_batch(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[4, 3, 2, 1]]),
        ),
    )
    assert len(calls) == 2
    assert not hasattr(trainer, "_gu_retain_inputs")

    writes = []
    real_write = os.write

    def record_write(descriptor, payload):
        writes.append(payload)
        return real_write(descriptor, payload)

    monkeypatch.setattr(os, "write", record_write)
    trainer.optimizer.step()

    assert len(calls) == 2
    contents = trainer._gu_diagnostics_path.read_bytes()
    records = contents.decode().splitlines()
    assert writes == [contents]
    assert len(records) == 1
    assert json.loads(records[0]) == trainer.gu_last_diagnostics
    assert trainer.gu_last_diagnostics["retain_loss_before"] is None
    assert trainer.gu_last_diagnostics["retain_loss_after"] is None
    assert trainer.gu_last_diagnostics["applied_scale"] == 1.0


def test_diagnostics_write_failure_restores_snapshot_without_commit(tmp_path):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    constraint = coordinate_constraint(trainer, [(0, 0, 1.0)])
    install_constraints(trainer, (constraint,))
    history = trainer._gu_constraint_history
    snapshot = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    trainer._gu_diagnostics_path.unlink()
    trainer._gu_diagnostics_path.mkdir()

    with pytest.raises(OSError):
        trainer.optimizer.step()

    assert torch.equal(parameter, snapshot)
    assert trainer.optimizer.state[parameter]["step"].item() == 1
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    assert trainer.gu_last_diagnostics is None
    for attribute in (
        "_gu_constraints_used",
        "_gu_pending_history_covector",
        "_gu_parameter_snapshot",
        "_gu_retain_inputs",
    ):
        assert not hasattr(trainer, attribute)


def test_diagnostics_fsync_failure_truncates_record_before_rollback(
    tmp_path,
    monkeypatch,
):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    constraint = coordinate_constraint(trainer, [(0, 0, 1.0)])
    install_constraints(trainer, (constraint,))
    history = trainer._gu_constraint_history
    snapshot = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    existing = b'{"prior": true}\n'
    trainer._gu_diagnostics_path.write_bytes(existing)

    fsync_calls = 0
    real_fsync = os.fsync

    def fail_first_fsync(descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            raise OSError("configured fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="configured fsync failure"):
        trainer.optimizer.step()

    assert trainer._gu_diagnostics_path.read_bytes() == existing
    assert torch.equal(parameter, snapshot)
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    assert trainer.gu_last_diagnostics is None
    assert fsync_calls == 2


def test_diagnostics_partial_write_leaves_no_partial_record(tmp_path, monkeypatch):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    constraint = coordinate_constraint(trainer, [(0, 0, 1.0)])
    install_constraints(trainer, (constraint,))
    history = trainer._gu_constraint_history
    snapshot = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    existing = b'{"prior": true}\n'
    trainer._gu_diagnostics_path.write_bytes(existing)
    real_write = os.write

    def partial_write(descriptor, payload):
        return real_write(descriptor, payload[: max(1, len(payload) // 2)])

    monkeypatch.setattr(os, "write", partial_write)
    with pytest.raises(OSError, match="incomplete"):
        trainer.optimizer.step()

    assert trainer._gu_diagnostics_path.read_bytes() == existing
    assert torch.equal(parameter, snapshot)
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    assert trainer.gu_last_diagnostics is None


def test_diagnostics_close_failure_after_fsync_keeps_committed_step(
    tmp_path,
    monkeypatch,
):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    constraint = coordinate_constraint(trainer, [(0, 0, 1.0)])
    install_constraints(trainer, (constraint,))
    parameter.grad = torch.ones_like(parameter)
    close_calls = 0
    real_close = os.close

    def fail_after_close(descriptor):
        nonlocal close_calls
        close_calls += 1
        real_close(descriptor)
        raise OSError("configured close failure")

    monkeypatch.setattr(os, "close", fail_after_close)
    trainer.optimizer.step()

    assert close_calls == 1
    records = trainer._gu_diagnostics_path.read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0]) == trainer.gu_last_diagnostics
    assert trainer.gu_projection_calls == 1
    assert trainer._gu_constraint_history[0] is constraint


def test_diagnostics_close_failure_before_fsync_does_not_commit(
    tmp_path,
    monkeypatch,
):
    trainer = make_trainer(TinyCausalLM(), tmp_path, gu=gu_config())
    trainer.create_optimizer()
    parameter = trainer._gu_selected[0][1]
    constraint = coordinate_constraint(trainer, [(0, 0, 1.0)])
    install_constraints(trainer, (constraint,))
    history = trainer._gu_constraint_history
    snapshot = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    existing = b'{"prior": true}\n'
    trainer._gu_diagnostics_path.write_bytes(existing)
    fsync_calls = 0
    real_fsync, real_close = os.fsync, os.close

    def fail_first_fsync(descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            raise OSError("configured fsync failure")
        return real_fsync(descriptor)

    def fail_after_close(descriptor):
        real_close(descriptor)
        raise OSError("configured close failure")

    monkeypatch.setattr(os, "fsync", fail_first_fsync)
    monkeypatch.setattr(os, "close", fail_after_close)
    with pytest.raises(OSError, match="configured close failure"):
        trainer.optimizer.step()

    assert trainer._gu_diagnostics_path.read_bytes() == existing
    assert torch.equal(parameter, snapshot)
    assert trainer._gu_constraint_history is history
    assert trainer.gu_projection_calls == 0
    assert trainer.gu_last_diagnostics is None


def test_finite_scale_write_keeps_delta_scaling_in_fp32():
    source = inspect.getsource(UnlearnTrainer._gu_optimizer_step_post_hook)

    assert "delta.mul(scale).to(parameter.dtype)" in source
    assert "delta.double() * scale" not in source


TASK7_GU_KEYS = {
    "enabled",
    "parameter_regex",
    "retain_history_rank",
    "projection_eps",
    "retain_filter",
    "retain_budget",
    "backtracking_scales",
    "diagnostics_path",
}
TASK7_ROOT = Path(__file__).resolve().parents[1]
TASK7_SCRIPTS = (
    TASK7_ROOT / "scripts/gu_adam_mvp_eval.sh",
    TASK7_ROOT / "scripts/gu_eval.sh",
    TASK7_ROOT / "scripts/pcgrad_smoke_arm.sh",
)


def task7_gu_keys_from_argv(argv):
    mapping_arguments = [
        argument
        for argument in argv
        if argument.startswith("+trainer.method_args.gu=")
    ]
    dotted_arguments = [
        argument
        for argument in argv
        if argument.startswith("+trainer.method_args.gu.")
    ]
    assert bool(mapping_arguments) != bool(dotted_arguments)
    if mapping_arguments:
        assert len(mapping_arguments) == 1
        return set(
            re.findall(r"(?:[{,])\s*([a-z_]+):", mapping_arguments[0])
        )
    return {
        argument.removeprefix("+trainer.method_args.gu.").split("=", 1)[0]
        for argument in dotted_arguments
    }


def task7_read_commands(path):
    if not path.exists():
        return []
    return [
        [value.decode() for value in record.split(b"\0") if value]
        for record in path.read_bytes().split(b"\x1e")
        if record
    ]


def task7_run_eval_script(tmp_path, accelerate_exit_code=0):
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    capture_path = tmp_path / "accelerate.argv"
    eval_capture_path = tmp_path / "eval.argv"
    python_stub = stub_dir / "python"
    python_stub.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = "-c" ]; then\n'
        "    printf '12345\\n'\n"
        "    exit 0\n"
        "fi\n"
        'printf \'%s\\0\' "$@" >> "$TASK7_EVAL_ARGV_CAPTURE"\n'
        'printf \'\\036\' >> "$TASK7_EVAL_ARGV_CAPTURE"\n'
    )
    python_stub.chmod(0o755)
    accelerate_stub = stub_dir / "accelerate"
    accelerate_stub.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\0\' "$@" >> "$TASK7_ARGV_CAPTURE"\n'
        'printf \'\\036\' >> "$TASK7_ARGV_CAPTURE"\n'
        'exit "$TASK7_ACCELERATE_EXIT_CODE"\n'
    )
    accelerate_stub.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{stub_dir}:{environment['PATH']}"
    environment["TASK7_ARGV_CAPTURE"] = str(capture_path)
    environment["TASK7_EVAL_ARGV_CAPTURE"] = str(eval_capture_path)
    environment["TASK7_ACCELERATE_EXIT_CODE"] = str(accelerate_exit_code)

    result = subprocess.run(
        ["bash", str(TASK7_SCRIPTS[1])],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    return (
        result,
        task7_read_commands(capture_path),
        task7_read_commands(eval_capture_path),
    )


def task7_stub_launch_environment(tmp_path, create_tofu_summary=False):
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    conda_base = tmp_path / "conda"
    profile = conda_base / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text("conda() { :; }\n")
    conda_stub = stub_dir / "conda"
    conda_stub.write_text('#!/bin/sh\nprintf \'%s\\n\' "$TASK7_CONDA_BASE"\n')
    conda_stub.chmod(0o755)
    capture_path = tmp_path / "launch.argv"
    accelerate_stub = stub_dir / "accelerate"
    accelerate_stub.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\0\' "$@" >> "$TASK7_ARGV_CAPTURE"\n'
        'printf \'\\036\' >> "$TASK7_ARGV_CAPTURE"\n'
        "output_dir=\n"
        'for argument in "$@"; do\n'
        '    case "$argument" in\n'
        '        paths.output_dir=*) output_dir=${argument#paths.output_dir=} ;;\n'
        "    esac\n"
        "done\n"
        'if [ -n "$output_dir" ]; then\n'
        '    mkdir -p "$output_dir/.hydra"\n'
        '    printf \'{}\\n\' > "$output_dir/.hydra/config.yaml"\n'
        "fi\n"
        'if [ "${TASK7_CREATE_TOFU_SUMMARY:-0}" = 1 ]; then\n'
        '    mkdir -p "$output_dir/checkpoint-10/evals"\n'
        '    printf \'{}\\n\' > "$output_dir/checkpoint-10/evals/TOFU_SUMMARY.json"\n'
        "fi\n"
    )
    accelerate_stub.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{stub_dir}:{environment['PATH']}"
    environment["CONDA_EXE"] = str(conda_stub)
    environment["TASK7_CONDA_BASE"] = str(conda_base)
    environment["TASK7_ARGV_CAPTURE"] = str(capture_path)
    environment["TASK7_CREATE_TOFU_SUMMARY"] = str(int(create_tofu_summary))
    environment["PCGRAD_LOCAL_ROOT"] = str(tmp_path / "local")
    return environment, capture_path


def task7_run_mvp_script(tmp_path):
    environment, capture_path = task7_stub_launch_environment(tmp_path)
    result = subprocess.run(
        ["bash", str(TASK7_SCRIPTS[0])],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, task7_read_commands(capture_path)


def task7_run_smoke_arms(tmp_path):
    environment, capture_path = task7_stub_launch_environment(
        tmp_path,
        create_tofu_summary=True,
    )
    results = []
    for method in ("control", "gu", "pcgrad"):
        results.append(
            subprocess.run(
                [
                    "bash",
                    str(TASK7_SCRIPTS[2]),
                    "tofu01",
                    method,
                    "0",
                    "task7",
                    "production",
                ],
                cwd=tmp_path,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
        )
    return results, task7_read_commands(capture_path)


def test_task7_legacy_gu_has_exact_migration_and_safe_legacy_controls(tmp_path):
    from tests.test_geometric_adam import make_geometric_trainer

    message = (
        "Legacy GU is retired; select the objective trainer and set "
        "trainer.method_args.gu.enabled=true."
    )
    with pytest.raises(ValueError) as error:
        make_geometric_trainer(
            tmp_path / "legacy-gu",
            geometric_overrides={"gradient_surgery": "gu"},
        )
    assert str(error.value) == message

    disabled, _, _ = make_geometric_trainer(
        tmp_path / "disabled",
        gu_enabled=False,
        geometric_overrides={"gradient_surgery": "gu"},
    )
    pcgrad, _, _ = make_geometric_trainer(
        tmp_path / "pcgrad",
        geometric_overrides={"gradient_surgery": "pcgrad"},
    )
    assert disabled.gu_enabled is False
    assert disabled.gradient_surgery == "gu"
    assert pcgrad.gradient_surgery == "pcgrad"

    source = inspect.getsource(type(pcgrad).__init__)
    assert "(_ for _ in ())" not in source
    assert "# noqa" not in source
    guard = (
        'if bool(getattr(self.geometric_config, "gu_enabled", True)) '
        'and self.gradient_surgery == "gu":\n'
    )
    assert guard in source
    guard_index = source.index(guard)
    assert source.index("self.gradient_surgery") < guard_index
    assert guard_index < source.index("self.simnpo_config")
    assert guard_index < source.index("super().__init__")
    assert guard_index < source.index("raise ValueError(", guard_index)
    assert source.index("super().__init__") < source.index("self.gu_enabled = bool(")


def test_task7_legacy_yaml_defaults_to_pcgrad():
    from omegaconf import OmegaConf

    config = OmegaConf.load(TASK7_ROOT / "configs/trainer/GeometricUnlearn.yaml")

    assert config.method_args.geometric_config.gradient_surgery == "pcgrad"


def test_task7_mvp_expands_both_arms_with_exact_common_gu_contract(tmp_path):
    result, commands = task7_run_mvp_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert len(commands) == 2
    runtime_knobs = (
        "trainer.args.max_steps=10",
        "trainer.args.learning_rate=1e-5",
        "trainer.args.optim=adamw_torch",
        "trainer.args.adam_beta1=0.0",
        "trainer.args.weight_decay=0.0",
        "trainer.args.fp16=false",
        "trainer.args.bf16=false",
        "trainer.args.bf16_full_eval=false",
        "trainer.args.gradient_checkpointing=true",
        "trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
        "trainer.args.save_strategy=no",
    )
    for argv, enabled in zip(commands, (False, True)):
        assert "trainer=SimNPO" in argv
        assert "trainer=GeometricUnlearn" not in argv
        assert task7_gu_keys_from_argv(argv) == TASK7_GU_KEYS
        gu_argument = next(
            argument
            for argument in argv
            if argument.startswith("+trainer.method_args.gu=")
        )
        assert f"enabled:{str(enabled).lower()}" in gu_argument
        assert "parameter_regex:[\"lm_head[.]weight\"]" in gu_argument
        for argument in runtime_knobs:
            assert argv.count(argument) == 1


def test_task7_eval_maps_supported_losses_to_objective_trainers_and_one_process():
    text = TASK7_SCRIPTS[1].read_text()
    mappings = {
        "simnpo": "SimNPO",
        "npo": "NPO",
        "dpo": "DPO",
        "undial": "UNDIAL",
        "ceu": "CEU",
        "wga": "WGA",
        "satimp": "SatImp",
    }

    for loss, trainer in mappings.items():
        assert re.search(
            rf"{loss}[)]\s+trainer_config={trainer}\s+;;",
            text,
        )
    assert "trainer=GeometricUnlearn" not in text
    assert '"trainer=${trainer_config}"' in text
    assert "NUM_GPUS=1" in text
    assert '--num_processes "${NUM_GPUS}"' in text


def test_task7_eval_preserves_exact_runtime_contract_at_every_launch(tmp_path):
    result, commands, eval_commands = task7_run_eval_script(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(commands) == 84
    assert len(eval_commands) == 84

    expected_gu_overrides = [
        "+trainer.method_args.gu.enabled=true",
        '+trainer.method_args.gu.parameter_regex=["lm_head[.]weight"]',
        "+trainer.method_args.gu.retain_history_rank=8",
        "+trainer.method_args.gu.projection_eps=1e-6",
        "+trainer.method_args.gu.retain_filter=first_order",
        "+trainer.method_args.gu.retain_budget=1e-4",
        "+trainer.method_args.gu.backtracking_scales=[1.0,0.5,0.25,0.125]",
        "+trainer.method_args.gu.diagnostics_path=gu_diagnostics.jsonl",
    ]
    expected_prefix = [
        "launch",
        "--config_file",
        "configs/accelerate/gu_single_gpu.yaml",
        "--main_process_port",
        "12345",
        "--num_processes",
        "1",
        "src/train.py",
        "--config-name=unlearn.yaml",
    ]
    expected_common_knobs = [
        "trainer.args.learning_rate=1e-5",
        "trainer.args.optim=adamw_torch",
        "+trainer.args.adam_beta1=0.0",
        "+trainer.args.fp16=false",
        "trainer.args.bf16=false",
        "trainer.args.bf16_full_eval=false",
        "trainer.args.gradient_checkpointing=true",
        "+trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
        "trainer.args.save_strategy=no",
    ]
    expected_eval_knobs = [
        "trainer.args.ddp_find_unused_parameters=true",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
    ]
    expected_benchmark_knobs = {
        "tofu": [
            "trainer.args.num_train_epochs=10",
            "+trainer.args.max_steps=-1",
            "trainer.args.weight_decay=0.01",
        ],
        "muse": [
            "trainer.args.num_train_epochs=10",
            "+trainer.args.max_steps=-1",
            "trainer.args.weight_decay=0.0",
        ],
        "wmdp": [
            "trainer.args.num_train_epochs=5",
            "trainer.args.max_steps=80",
            "trainer.args.weight_decay=0.0",
        ],
    }
    trainers = {"SimNPO", "NPO", "DPO", "UNDIAL", "CEU", "WGA", "SatImp"}
    site_trainers = {"tofu": set(), "muse": set(), "wmdp": set()}

    for argv in commands:
        assert argv[: len(expected_prefix)] == expected_prefix
        assert not any("\n" in argument or "\\\n" in argument for argument in argv)
        trainer = next(
            argument.removeprefix("trainer=")
            for argument in argv
            if argument.startswith("trainer=")
        )
        task_name = next(
            argument.removeprefix("task_name=")
            for argument in argv
            if argument.startswith("task_name=")
        )
        site = task_name.split("_", 1)[0]
        site_trainers[site].add(trainer)
        assert trainer in trainers
        for argument in (
            *expected_common_knobs,
            *expected_benchmark_knobs[site],
            *expected_eval_knobs,
            *expected_gu_overrides,
        ):
            assert argv.count(argument) == 1
        assert task7_gu_keys_from_argv(argv) == TASK7_GU_KEYS
        if site == "tofu":
            assert "trainer.args.per_device_train_batch_size=4" in argv
            assert "trainer.args.gradient_accumulation_steps=4" in argv
            assert any(argument.startswith("forget_split=") for argument in argv)
            assert any(argument.startswith("retain_split=") for argument in argv)
            assert any(argument.startswith("retain_logs_path=") for argument in argv)
            assert any(
                argument.startswith("model.model_args.pretrained_model_name_or_path=")
                for argument in argv
            )
        elif site == "muse":
            assert "trainer.args.per_device_train_batch_size=2" in argv
            assert "trainer.args.gradient_accumulation_steps=8" in argv
            assert any(argument.startswith("data_split=") for argument in argv)
            assert any(argument.startswith("retain_logs_path=") for argument in argv)
            assert any(
                argument.startswith("model.model_args.pretrained_model_name_or_path=")
                for argument in argv
            )
        else:
            assert "trainer.args.per_device_train_batch_size=2" in argv
            assert "trainer.args.gradient_accumulation_steps=8" in argv
            assert any(argument.startswith("data_split=") for argument in argv)

    assert site_trainers == {site: trainers for site in site_trainers}


def test_task7_eval_all_benchmark_objectives_compose_to_legacy_effective_configs(
    tmp_path,
):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    result, commands, _ = task7_run_eval_script(tmp_path)
    assert result.returncode == 0, result.stderr
    config_path = next(
        command[command.index("--config_file") + 1] for command in commands
    )
    accelerate_config = OmegaConf.load(TASK7_ROOT / config_path)
    assert accelerate_config.distributed_type == "NO"
    assert "deepspeed_config" not in accelerate_config

    representatives = {}
    for command in commands:
        trainer = next(
            argument.removeprefix("trainer=")
            for argument in command
            if argument.startswith("trainer=")
        )
        task_name = next(
            argument.removeprefix("task_name=")
            for argument in command
            if argument.startswith("task_name=")
        )
        representatives.setdefault((task_name.split("_", 1)[0], trainer), command)
    trainers = {"SimNPO", "NPO", "DPO", "UNDIAL", "CEU", "WGA", "SatImp"}
    assert set(representatives) == {
        (benchmark, trainer)
        for benchmark in ("tofu", "muse", "wmdp")
        for trainer in trainers
    }
    expected = {
        "tofu": (10, -1, 0.01),
        "muse": (10, -1, 0.0),
        "wmdp": (5, 80, 0.0),
    }
    with initialize_config_dir(
        config_dir=str(TASK7_ROOT / "configs"),
        version_base=None,
    ):
        for (benchmark, _), command in representatives.items():
            config_name_index = command.index("--config-name=unlearn.yaml")
            config = compose(
                config_name="unlearn.yaml",
                overrides=command[config_name_index + 1 :],
            )
            epochs, max_steps, weight_decay = expected[benchmark]
            assert "deepspeed" not in config.trainer.args
            assert config.trainer.args.learning_rate == pytest.approx(1.0e-5)
            assert config.trainer.args.num_train_epochs == epochs
            assert config.trainer.args.max_steps == max_steps
            assert config.trainer.args.optim == "adamw_torch"
            assert config.trainer.args.adam_beta1 == pytest.approx(0.0)
            assert config.trainer.args.weight_decay == pytest.approx(weight_decay)
            assert config.trainer.args.fp16 is False
            assert config.trainer.args.bf16 is False
            assert config.trainer.args.bf16_full_eval is False
            assert config.trainer.args.gradient_checkpointing is True
            assert (
                config.trainer.args.gradient_checkpointing_kwargs.use_reentrant
                is False
            )
            assert config.trainer.args.save_strategy == "no"
            assert set(config.trainer.method_args.gu) == TASK7_GU_KEYS


def test_task7_eval_stops_before_eval_and_success_after_training_failure(tmp_path):
    result, commands, eval_commands = task7_run_eval_script(
        tmp_path,
        accelerate_exit_code=42,
    )

    assert result.returncode == 42
    assert len(commands) == 1
    assert eval_commands == []
    assert "All benchmarks completed!" not in result.stdout


def test_task7_smoke_expands_control_gu_and_pcgrad_arms(tmp_path):
    results, commands = task7_run_smoke_arms(tmp_path)

    assert len(commands) == 3
    assert all(result.returncode == 0 for result in results), [
        result.stderr for result in results
    ]
    common_runtime_knobs = (
        "trainer.args.max_steps=10",
        "trainer.args.learning_rate=1e-5",
        "trainer.args.adam_beta1=0.0",
        "trainer.args.weight_decay=0.0",
        "trainer.args.fp16=false",
        "trainer.args.gradient_checkpointing=true",
        "trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
        "trainer.args.save_strategy=no",
    )
    by_method = {}
    for argv in commands:
        task_name = next(
            argument.removeprefix("task_name=")
            for argument in argv
            if argument.startswith("task_name=")
        )
        method = task_name.split("_")[3]
        by_method[method] = argv
        for argument in common_runtime_knobs:
            assert argv.count(argument) == 1

    assert set(by_method) == {"control", "gu", "pcgrad"}
    assert "trainer=SimNPO" in by_method["gu"]
    assert task7_gu_keys_from_argv(by_method["gu"]) == TASK7_GU_KEYS
    assert not any(
        "geometric_config" in argument for argument in by_method["gu"]
    )
    for method, enabled, surgery in (
        ("control", "false", "gu"),
        ("pcgrad", "true", "pcgrad"),
    ):
        argv = by_method[method]
        assert "trainer=GeometricUnlearn" in argv
        assert (
            f"trainer.method_args.geometric_config.gu_enabled={enabled}" in argv
        )
        assert (
            f"trainer.method_args.geometric_config.gradient_surgery={surgery}"
            in argv
        )
        assert not any(
            argument.startswith("+trainer.method_args.gu") for argument in argv
        )


def test_task7_shipped_gu_scripts_have_valid_bash_syntax():
    for path in TASK7_SCRIPTS:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


TASK8_DIAGNOSTIC_FIELDS = {
    "step",
    "objective",
    "selected_parameter_count",
    "proposal_norm",
    "corrected_norm",
    "correction_ratio",
    "constraint_count",
    "active_constraints",
    "max_violation_before",
    "max_violation_after",
    "kkt_residual",
    "projection_tolerance",
    "applied_scale",
    "retain_loss_before",
    "retain_loss_after",
    "zero_step",
    "zero_step_reason",
    "optimizer_state_semantics",
    "projection_seconds",
    "filter_seconds",
}


def task8_unbatch(batch):
    return [
        {
            component: {
                key: tensor[index]
                for key, tensor in component_batch.items()
            }
            for component, component_batch in batch.items()
        }
        for index in range(batch["forget"]["input_ids"].shape[0])
    ]


def task8_assert_no_trainer_artifacts(output_dir):
    forbidden_names = {
        "model.safetensors",
        "optimizer.pt",
        "pytorch_model.bin",
        "trainer_state.json",
        "training_args.bin",
    }
    paths = tuple(output_dir.rglob("*"))
    assert not forbidden_names.intersection(path.name for path in paths)
    assert not any(
        path.is_dir() and path.name.startswith("checkpoint-") for path in paths
    )


def test_task8_hydra_flag_reaches_common_gu_branch_and_diagnostic(tmp_path):
    from hydra import compose, initialize_config_dir

    from tests.helpers import TinyCausalLM as ShippedTinyCausalLM
    from tests.helpers import make_unlearn_batch, nested_collator
    from trainer import load_trainer

    output_dir = tmp_path / "hydra-trace"
    overrides = [
        "trainer=SimNPO",
        "task_name=task8_hydra_trace",
        f"paths.output_dir={output_dir}",
        "trainer.args.per_device_train_batch_size=1",
        "trainer.args.gradient_accumulation_steps=1",
        "+trainer.args.max_steps=1",
        "+trainer.args.use_cpu=true",
        "trainer.args.bf16=false",
        "trainer.args.bf16_full_eval=false",
        "trainer.args.optim=adamw_torch",
        "trainer.args.report_to=none",
        "+trainer.args.disable_tqdm=true",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        "+trainer.method_args.gu.enabled=true",
        "+trainer.method_args.gu.parameter_regex=[\"lm_head[.]weight\"]",
        "+trainer.method_args.gu.retain_history_rank=8",
        "+trainer.method_args.gu.projection_eps=1e-6",
        "+trainer.method_args.gu.retain_filter=first_order",
        "+trainer.method_args.gu.retain_budget=1e-4",
        "+trainer.method_args.gu.backtracking_scales=[1.0,0.5,0.25,0.125]",
        "+trainer.method_args.gu.diagnostics_path=gu_diagnostics.jsonl",
    ]
    with initialize_config_dir(
        config_dir=str(TASK7_ROOT / "configs"),
        version_base=None,
    ):
        config = compose(config_name="unlearn.yaml", overrides=overrides)

    dataset = task8_unbatch(
        make_unlearn_batch(batch_size=1, sequence_length=6, seed=71)
    )
    trainer, _ = load_trainer(
        trainer_cfg=config.trainer,
        model=ShippedTinyCausalLM(),
        train_dataset=dataset,
        data_collator=nested_collator,
    )
    assert config.trainer.method_args.gu.enabled is True
    assert trainer.gu_enabled is True

    trainer.train()

    records = [
        json.loads(line)
        for line in (output_dir / "gu_diagnostics.jsonl").read_text().splitlines()
    ]
    assert trainer.gu_projection_calls == trainer.state.global_step == 1
    assert len(records) == 1
    assert set(records[0]) == TASK8_DIAGNOSTIC_FIELDS
    assert records[0]["objective"] == "SimNPO"
    assert records[0]["max_violation_after"] <= records[0][
        "projection_tolerance"
    ]
    task8_assert_no_trainer_artifacts(output_dir)


def test_task8_registered_objectives_inherit_common_gu_without_dispatch(tmp_path):
    from trainer import TRAINER_REGISTRY
    from trainer.unlearn.geometric import GeometricUnlearn
    from trainer.unlearn.grad_ascent import GradAscent
    from trainer.unlearn.npo import NPO
    from trainer.unlearn.pdu import PDU
    from trainer.unlearn.rmu import RMU

    registered = {
        name: trainer_class
        for name, trainer_class in TRAINER_REGISTRY.items()
        if issubclass(trainer_class, UnlearnTrainer)
    }
    assert registered
    for trainer_class in registered.values():
        assert UnlearnTrainer in trainer_class.__mro__
        assert (
            trainer_class._gu_optimizer_step_pre_hook
            is UnlearnTrainer._gu_optimizer_step_pre_hook
        )
        assert (
            trainer_class._gu_optimizer_step_post_hook
            is UnlearnTrainer._gu_optimizer_step_post_hook
        )
        if trainer_class is not GeometricUnlearn:
            assert trainer_class.training_step is UnlearnTrainer.training_step

    common_source = "\n".join(
        inspect.getsource(method)
        for method in (
            UnlearnTrainer.create_optimizer,
            UnlearnTrainer.training_step,
            UnlearnTrainer._gu_optimizer_step_pre_hook,
            UnlearnTrainer._gu_optimizer_step_post_hook,
        )
    )
    assert "TRAINER_REGISTRY" not in common_source
    for objective_name in registered:
        assert objective_name not in common_source

    cases = (
        (GradAscent, {}),
        (NPO, {"beta": 1.0, "gamma": 1.0, "alpha": 1.0, "retain_loss_type": "NLL"}),
        (
            RMU,
            {
                "module_regex": "protected",
                "trainable_params_regex": ["protected[.]weight"],
                "gamma": 1.0,
                "alpha": 1.0,
                "retain_loss_type": "NLL",
            },
        ),
        (
            PDU,
            {
                "primal_dual": False,
                "gamma": 1.0,
                "alpha": 1.0,
                "retain_loss_type": "NLL",
            },
        ),
    )
    for trainer_class, method_args in cases:
        args = TrainingArguments(
            output_dir=str(tmp_path / trainer_class.__name__),
            use_cpu=True,
            report_to=[],
            save_strategy="no",
        )
        trainer = trainer_class(
            model=TinyCausalLM(),
            args=args,
            gu=gu_config(enabled=False),
            **method_args,
        )
        assert trainer.gu_enabled is False
        assert type(trainer).training_step is UnlearnTrainer.training_step


def test_rmu_common_gu_keeps_exact_scope_and_gradients(tmp_path):
    from tests.helpers import TinyCausalLM as ShippedTinyCausalLM
    from tests.helpers import make_unlearn_batch
    from trainer.unlearn.rmu import RMU

    model = ShippedTinyCausalLM()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        optim="adamw_torch",
        remove_unused_columns=False,
    )
    trainer = RMU(
        model=model,
        args=args,
        module_regex="lm_head",
        trainable_params_regex=["embed[.]weight", "lm_head[.]weight"],
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        gu=gu_config(parameter_regex=["lm_head[.]weight"]),
    )
    trainer.create_optimizer()
    optimizer = trainer.optimizer
    pre_hooks = tuple(optimizer._optimizer_step_pre_hooks.values())
    post_hooks = tuple(optimizer._optimizer_step_post_hooks.values())

    trainer.create_optimizer()

    assert trainer.optimizer is optimizer
    assert tuple(optimizer._optimizer_step_pre_hooks.values()) == pre_hooks
    assert tuple(optimizer._optimizer_step_post_hooks.values()) == post_hooks
    assert model.embed.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is True
    assert [parameter for group in trainer.optimizer.param_groups
        for parameter in group["params"]] == [model.lm_head.weight]

    trainer.training_step(model, make_unlearn_batch(batch_size=2, seed=83))

    assert model.embed.weight.grad is None
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_task8_bf16_trainer_applies_one_safe_delta_per_update(tmp_path):
    from tests.helpers import TinyCausalLM as ShippedTinyCausalLM
    from tests.helpers import make_unlearn_batch, nested_collator
    from trainer.unlearn.simnpo import SimNPO

    output_dir = tmp_path / "bf16-trainer"
    model = ShippedTinyCausalLM().to(device="cuda", dtype=torch.bfloat16)
    dataset = task8_unbatch(
        make_unlearn_batch(batch_size=4, sequence_length=6, seed=17)
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=2,
        learning_rate=1.0e-2,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        save_strategy="no",
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    trainer = SimNPO(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=nested_collator,
        beta=4.5,
        delta=0.0,
        gamma=0.125,
        alpha=1.0,
        retain_loss_type="NLL",
        gu=gu_config(
            parameter_regex=["lm_head[.]weight"],
            diagnostics_path="gu_diagnostics.jsonl",
        ),
    )
    trainer.create_optimizer()
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    before = tuple(
        parameter.detach().clone() for _, parameter in trainer._gu_selected
    )

    trainer.train()

    after = tuple(
        parameter.detach().clone() for _, parameter in trainer._gu_selected
    )
    records = [
        json.loads(line)
        for line in (output_dir / "gu_diagnostics.jsonl").read_text().splitlines()
    ]
    assert trainer.gu_projection_calls == trainer.state.global_step == 2
    assert any(
        not torch.equal(before_value, after_value)
        for before_value, after_value in zip(before, after)
    )
    assert len(records) == 2
    assert all(
        len(record) == 20 and set(record) == TASK8_DIAGNOSTIC_FIELDS
        for record in records
    )
    assert all(
        record["max_violation_after"] <= record["projection_tolerance"]
        for record in records
    )
    task8_assert_no_trainer_artifacts(output_dir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_task8_paged_adamw32_trainer_commits_finite_safe_delta(tmp_path):
    bitsandbytes = pytest.importorskip("bitsandbytes")
    from tests.helpers import TinyCausalLM as ShippedTinyCausalLM
    from tests.helpers import make_unlearn_batch, nested_collator
    from trainer.unlearn.simnpo import SimNPO

    output_dir = tmp_path / "paged-adamw32-trainer"
    model = ShippedTinyCausalLM().to(device="cuda", dtype=torch.float32)
    optimizer = bitsandbytes.optim.PagedAdamW32bit(
        [model.lm_head.weight],
        lr=1.0e-2,
    )
    dataset = task8_unbatch(
        make_unlearn_batch(batch_size=2, sequence_length=6, seed=37)
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        max_steps=1,
        learning_rate=1.0e-2,
        optim="paged_adamw_32bit",
        report_to=[],
        save_strategy="no",
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    trainer = SimNPO(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=nested_collator,
        optimizers=(optimizer, None),
        beta=4.5,
        delta=0.0,
        gamma=0.125,
        alpha=1.0,
        retain_loss_type="NLL",
        gu=gu_config(
            parameter_regex=["lm_head[.]weight"],
            diagnostics_path="gu_diagnostics.jsonl",
        ),
    )
    trainer.create_optimizer()
    before = tuple(
        parameter.detach().clone() for _, parameter in trainer._gu_selected
    )

    trainer.train()

    after = tuple(
        parameter.detach().clone() for _, parameter in trainer._gu_selected
    )
    assert trainer.gu_projection_calls == trainer.state.global_step == 1
    assert any(
        not torch.equal(before_value, after_value)
        for before_value, after_value in zip(before, after)
    )
    constraint = trainer._gu_constraint_history[0]
    actual_violation = sum(
        (block.double() * (after_value - before_value).double()).sum().item()
        for block, before_value, after_value in zip(constraint, before, after)
    )
    assert actual_violation <= trainer.gu_config["projection_eps"]
    for _, parameter in trainer._gu_selected:
        state2 = optimizer.state[parameter]["state2"]
        assert state2.shape == parameter.shape
        assert torch.isfinite(state2).all()
        assert torch.count_nonzero(state2).item() > 0
    records = [
        json.loads(line)
        for line in (output_dir / "gu_diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert set(records[0]) == TASK8_DIAGNOSTIC_FIELDS
    assert records[0]["max_violation_after"] <= records[0][
        "projection_tolerance"
    ]
    task8_assert_no_trainer_artifacts(output_dir)
