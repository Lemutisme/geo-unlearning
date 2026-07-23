import copy
import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

from trainer.unlearn.base import UnlearnTrainer


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


def make_trainer(model, output_dir, **kwargs):
    args = TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=True,
        report_to=[],
        learning_rate=1.0e-3,
        optim="adamw_torch",
        weight_decay=0.0,
        remove_unused_columns=False,
        disable_tqdm=True,
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


def test_gu_rollback_preserves_file_concurrently_created_before_open(
    tmp_path,
    monkeypatch,
):
    model = TinyCausalLM()
    optimizer = torch.optim.SGD([model.protected.weight], lr=1.0e-3)
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
        model,
        output_dir,
        gu=gu_config(diagnostics_path="diagnostics/gu.jsonl"),
        optimizers=(optimizer, None),
    )

    with pytest.raises(ValueError, match="AdamW"):
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
