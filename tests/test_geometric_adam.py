import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import TrainingArguments

import trainer.unlearn.geometric as geometric_module
from trainer.unlearn.geometric import GeometricUnlearn
from trainer.unlearn.optimizer_geometry import (
    make_optimizer_geometry_adapter,
    unwrap_optimizer,
)
from trainer.unlearn.simnpo import SimNPO
from tests.helpers import TinyCausalLM, make_unlearn_batch, nested_collator


def make_geometric_trainer(
    tmp_path,
    *,
    model=None,
    gu_enabled=True,
    gamma=0.125,
    alpha=1.0,
    train_dataset=None,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    max_steps=-1,
    geometric_overrides=None,
    common_gu=None,
):
    model = TinyCausalLM() if model is None else model
    geometric_config = SimpleNamespace(
        loss="simnpo",
        gu_enabled=gu_enabled,
        gradient_surgery="pcgrad",
        projection_eps=1e-12,
        trainable_params_regex=[".*"],
        auto_last_k_layers=1,
        null_k=1,
    )
    for key, value in (geometric_overrides or {}).items():
        setattr(geometric_config, key, value)
    simnpo_config = SimpleNamespace(
        delta=0.0,
        beta=4.5,
        alpha=alpha,
        gamma=gamma,
        retain_loss_type="NLL",
    )
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=1e-3,
        optim="adamw_torch",
        adam_beta1=0.0,
        weight_decay=0.0,
        remove_unused_columns=False,
        disable_tqdm=True,
        save_strategy="no",
    )
    trainer = GeometricUnlearn(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=nested_collator if train_dataset is not None else None,
        gamma=9.0,
        alpha=7.0,
        retain_loss_type="NLL",
        geometric_config=geometric_config,
        simnpo_config=simnpo_config,
        npo_config=SimpleNamespace(
            beta=0.1,
            alpha=1.0,
            gamma=1.0,
            retain_loss_type="NLL",
        ),
        dpo_config=None,
        undial_config=None,
        wga_config=None,
        satimp_config=None,
        gu=common_gu,
    )
    return trainer, model, simnpo_config


def unbatch(batch):
    batch_size = batch["forget"]["input_ids"].shape[0]
    return [
        {
            component: {key: tensor[index] for key, tensor in component_batch.items()}
            for component, component_batch in batch.items()
        }
        for index in range(batch_size)
    ]


def seed_nonuniform_adam_state(trainer):
    trainer.create_optimizer()
    optimizer = unwrap_optimizer(trainer.optimizer)
    for parameter_index, group in enumerate(optimizer.param_groups, start=1):
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            state["step"] = torch.tensor(3.0)
            state["exp_avg"] = torch.zeros_like(parameter)
            values = torch.arange(
                1,
                parameter.numel() + 1,
                device=parameter.device,
                dtype=torch.float32,
            ).reshape_as(parameter)
            state["exp_avg_sq"] = values.mul(1e-4 * parameter_index).to(parameter.dtype)


def test_unlearn_batch_has_forget_and_retain_components():
    batch = make_unlearn_batch(batch_size=2, sequence_length=6)
    assert set(batch) == {"forget", "retain"}
    assert batch["forget"]["labels"].shape == (2, 6)
    assert (batch["retain"]["labels"][:, :2] == -100).all()


def test_sqrt_adam_coordinate_round_trip():
    gradient = torch.tensor([2.0, -3.0])
    sqrt_h = torch.tensor([0.5, 4.0])

    coordinates = GeometricUnlearn._to_adam_coordinates(gradient, sqrt_h)
    restored = GeometricUnlearn._from_adam_coordinates(coordinates, sqrt_h)

    torch.testing.assert_close(restored, gradient)


def test_global_projection_is_orthogonal_across_tensors():
    forget = {"a": torch.tensor([2.0]), "b": torch.tensor([1.0])}
    retain = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}

    projected, coefficient = GeometricUnlearn._project_rank_one(
        forget,
        retain,
        1e-12,
    )

    assert coefficient.item() == pytest.approx(0.5)
    assert abs(GeometricUnlearn._global_dot(projected, retain).item()) < 1e-6


def test_global_projection_differs_from_blockwise_projection():
    forget = {"a": torch.tensor([2.0]), "b": torch.tensor([1.0])}
    retain = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}

    projected, _ = GeometricUnlearn._project_rank_one(forget, retain, 1e-12)

    assert projected["a"].item() == pytest.approx(1.5)
    assert projected["b"].item() == pytest.approx(-0.5)


def test_frozen_preconditioner_forget_update_is_retain_orthogonal():
    raw_forget = {
        "a": torch.tensor([2.0, -1.0]),
        "b": torch.tensor([0.5]),
    }
    raw_retain = {
        "a": torch.tensor([1.0, 3.0]),
        "b": torch.tensor([-2.0]),
    }
    sqrt_h = {
        "a": torch.tensor([0.5, 2.0]),
        "b": torch.tensor([4.0]),
    }
    transformed_forget = {
        name: GeometricUnlearn._to_adam_coordinates(tensor, sqrt_h[name])
        for name, tensor in raw_forget.items()
    }
    transformed_retain = {
        name: GeometricUnlearn._to_adam_coordinates(tensor, sqrt_h[name])
        for name, tensor in raw_retain.items()
    }

    projected, _ = GeometricUnlearn._project_rank_one(
        transformed_forget,
        transformed_retain,
        1e-12,
    )
    raw_projected = {
        name: GeometricUnlearn._from_adam_coordinates(tensor, sqrt_h[name])
        for name, tensor in projected.items()
    }
    delta = {
        name: -raw_projected[name] / sqrt_h[name].square() for name in raw_projected
    }
    derivative = GeometricUnlearn._global_dot(raw_retain, delta).abs()
    relative = derivative / (
        GeometricUnlearn._global_dot(raw_retain, raw_retain).sqrt()
        * GeometricUnlearn._global_dot(delta, delta).sqrt()
        + 1e-12
    )

    assert relative.item() < 1e-6


def test_zero_retain_gradient_leaves_forget_gradient_unchanged():
    forget = {"weight": torch.tensor([2.0, -3.0])}
    retain = {"weight": torch.zeros(2)}

    projected, coefficient = GeometricUnlearn._project_rank_one(
        forget,
        retain,
        1e-12,
    )

    torch.testing.assert_close(projected["weight"], forget["weight"])
    assert coefficient.item() == 0.0
    assert torch.isfinite(projected["weight"]).all()


@pytest.mark.parametrize(
    ("state_case", "message"),
    [
        ("partial_unstepped_state", "missing step"),
        ("missing_second_moment", "missing exp_avg_sq"),
        ("negative_second_moment", "negative values"),
    ],
)
def test_malformed_initialized_adam_state_fails_closed(
    tmp_path,
    state_case,
    message,
):
    trainer, model, _ = make_geometric_trainer(tmp_path)
    trainer.create_optimizer()
    optimizer = unwrap_optimizer(trainer.optimizer)
    adapter = make_optimizer_geometry_adapter(optimizer)
    parameter = next(model.parameters())
    group = adapter.groups_by_parameter()[id(parameter)]
    state = optimizer.state[parameter]
    state["exp_avg"] = torch.zeros_like(parameter)
    if state_case != "partial_unstepped_state":
        state["step"] = torch.tensor(1.0)
    if state_case == "negative_second_moment":
        state["exp_avg_sq"] = -torch.ones_like(parameter)

    with pytest.raises(RuntimeError, match=message):
        adapter.sqrt_denominator(parameter, group)


def test_unused_optimizer_step_hook_is_removed():
    assert "optimizer_step" not in GeometricUnlearn.__dict__


def test_obsolete_per_parameter_projector_is_removed():
    assert not hasattr(geometric_module, "RetainNullProjector")


def test_simnpo_components_are_separate_and_weights_resolve_once(tmp_path):
    trainer, model, simnpo_config = make_geometric_trainer(tmp_path)
    batch = make_unlearn_batch(batch_size=2, sequence_length=6)

    assert trainer.gamma == 0.125
    assert trainer.alpha == 1.0
    assert trainer.retain_loss_type == "NLL"

    forget_loss, retain_loss, outputs = trainer.compute_component_losses(
        model,
        batch,
    )
    total = trainer.compute_loss(model, batch)

    torch.testing.assert_close(total, 0.125 * forget_loss + retain_loss)
    assert outputs.logits.shape[:2] == batch["forget"]["labels"].shape

    simnpo_config.gamma = 4.0
    trainer.compute_loss(model, batch)
    assert trainer.gamma == 0.125


@pytest.mark.parametrize(
    ("gamma", "alpha", "message"),
    [
        (0.0, 1.0, "gamma > 0"),
        (0.125, -1.0, "alpha >= 0"),
    ],
)
def test_invalid_gu_coefficients_fail_during_initialization(
    tmp_path,
    gamma,
    alpha,
    message,
):
    with pytest.raises(ValueError, match=message):
        make_geometric_trainer(
            tmp_path,
            gamma=gamma,
            alpha=alpha,
        )


def test_component_gradient_buffers_accumulate_in_fp32_and_clear(tmp_path):
    trainer, model, _ = make_geometric_trainer(tmp_path)
    named_parameters = list(model.named_parameters())
    forget_grads = [torch.ones_like(parameter) for _, parameter in named_parameters]
    retain_grads = [
        torch.full_like(parameter, 2.0) for _, parameter in named_parameters
    ]

    trainer._accumulate_component_grads(
        named_parameters,
        forget_grads,
        retain_grads,
    )
    trainer._accumulate_component_grads(
        named_parameters,
        forget_grads,
        retain_grads,
    )

    for name, parameter in named_parameters:
        forget = trainer.component_buffers.tensor("forget", name, parameter.device)
        retain = trainer.component_buffers.tensor("retain", name, parameter.device)
        assert forget.dtype == torch.float32
        assert retain.dtype == torch.float32
        torch.testing.assert_close(
            forget,
            torch.full_like(forget, 2.0),
        )
        torch.testing.assert_close(
            retain,
            torch.full_like(retain, 4.0),
        )

    trainer._clear_gu_buffers()
    assert trainer.component_buffers.empty


def test_pcgrad_surgery_runs_once_per_optimizer_update(tmp_path):
    dataset = unbatch(make_unlearn_batch(batch_size=4, sequence_length=6))
    trainer, _, _ = make_geometric_trainer(
        tmp_path,
        train_dataset=dataset,
        per_device_train_batch_size=2,
        max_steps=2,
    )

    trainer.train()

    assert trainer.surgery_calls == trainer.state.global_step == 2
    assert trainer.component_buffers.empty
    assert trainer.last_surgery_diagnostics["mode"] == "pcgrad"


def test_short_final_accumulation_window_runs_pcgrad_surgery(tmp_path):
    dataset = unbatch(make_unlearn_batch(batch_size=2, sequence_length=6))
    trainer, _, _ = make_geometric_trainer(
        tmp_path,
        train_dataset=dataset,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1,
    )

    trainer.train()

    assert trainer.surgery_calls == trainer.state.global_step == 1
    assert trainer.component_buffers.empty


def test_gradient_accumulation_matches_full_effective_batch(tmp_path):
    torch.manual_seed(123)
    full_batch_model = TinyCausalLM()
    accumulated_model = copy.deepcopy(full_batch_model)
    dataset = unbatch(make_unlearn_batch(batch_size=8, sequence_length=6, seed=9))

    full_batch_trainer, _, _ = make_geometric_trainer(
        tmp_path / "full",
        model=full_batch_model,
        train_dataset=dataset,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        max_steps=1,
    )
    seed_nonuniform_adam_state(full_batch_trainer)
    full_batch_trainer.train()

    accumulated_trainer, _, _ = make_geometric_trainer(
        tmp_path / "accumulated",
        model=accumulated_model,
        train_dataset=dataset,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1,
    )
    seed_nonuniform_adam_state(accumulated_trainer)
    accumulated_trainer.train()

    assert (
        full_batch_trainer.last_surgery_diagnostics["identity_fallback_parameters"]
        == 0
    )
    assert (
        accumulated_trainer.last_surgery_diagnostics[
            "identity_fallback_parameters"
        ]
        == 0
    )
    assert full_batch_trainer.last_surgery_diagnostics[
        "coefficient"
    ] == pytest.approx(
        accumulated_trainer.last_surgery_diagnostics["coefficient"],
        rel=1e-5,
        abs=1e-6,
    )
    for full_parameter, accumulated_parameter in zip(
        full_batch_model.parameters(),
        accumulated_model.parameters(),
    ):
        torch.testing.assert_close(
            full_parameter,
            accumulated_parameter,
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("momentum", "beta1=0"),
        ("weight_decay", "weight_decay=0"),
        ("amsgrad", "AMSGrad"),
        ("sgd", "SGD"),
        ("reentrant", "use_reentrant=false"),
        ("fp16", "BF16/FP32"),
        ("apex", "Apex"),
        ("world_size", "one process and one GPU"),
        ("multi_gpu", "one process and one GPU"),
        ("deepspeed", "DeepSpeed"),
        ("fsdp", "FSDP"),
        ("empty_selection", "selected trainable parameter"),
    ],
)
def test_unsupported_runtime_modes_fail_closed(
    tmp_path,
    mutation,
    message,
):
    trainer, model, _ = make_geometric_trainer(tmp_path)
    trainer.create_optimizer()
    group = trainer.optimizer.param_groups[0]

    if mutation == "momentum":
        group["betas"] = (0.9, group["betas"][1])
    elif mutation == "weight_decay":
        group["weight_decay"] = 0.01
    elif mutation == "amsgrad":
        group["amsgrad"] = True
    elif mutation == "sgd":
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    elif mutation == "reentrant":
        trainer.args.gradient_checkpointing = True
        trainer.args.gradient_checkpointing_kwargs = {}
    elif mutation == "fp16":
        trainer.args.fp16 = True
    elif mutation == "apex":
        trainer.use_apex = True
    elif mutation == "world_size":
        trainer.args.distributed_state = SimpleNamespace(num_processes=2)
    elif mutation == "multi_gpu":
        trainer.args._n_gpu = 2
    elif mutation == "deepspeed":
        trainer.is_deepspeed_enabled = True
    elif mutation == "fsdp":
        trainer.is_fsdp_enabled = True
    elif mutation == "empty_selection":
        trainer.trainable_params_regex = ["does-not-match"]

    with pytest.raises((NotImplementedError, ValueError), match=message):
        trainer._validate_gu_runtime()


def test_standard_torch_adam_runtime_is_supported(tmp_path):
    trainer, model, _ = make_geometric_trainer(tmp_path)
    trainer.optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.0, 0.999),
        weight_decay=0.0,
    )

    trainer._validate_gu_runtime()


def test_geometry_disabled_recovers_native_simnpo_update(tmp_path):
    torch.manual_seed(456)
    native_model = TinyCausalLM()
    geometric_model = copy.deepcopy(native_model)
    dataset = unbatch(make_unlearn_batch(batch_size=4, sequence_length=6, seed=17))
    batch = make_unlearn_batch(batch_size=2, sequence_length=6, seed=23)
    native_args = TrainingArguments(
        output_dir=str(tmp_path / "native"),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=4,
        max_steps=1,
        learning_rate=1e-3,
        optim="adamw_torch",
        adam_beta1=0.0,
        weight_decay=0.0,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    native_trainer = SimNPO(
        model=native_model,
        args=native_args,
        train_dataset=dataset,
        data_collator=nested_collator,
        gamma=0.125,
        alpha=1.0,
        retain_loss_type="NLL",
        delta=0.0,
        beta=4.5,
    )
    native_loss = native_trainer.compute_loss(native_model, batch).detach()
    native_trainer.train()

    geometric_trainer, _, _ = make_geometric_trainer(
        tmp_path / "geometric",
        model=geometric_model,
        gu_enabled=False,
        train_dataset=dataset,
        per_device_train_batch_size=4,
        max_steps=1,
    )
    geometric_loss = geometric_trainer.compute_loss(
        geometric_model,
        batch,
    ).detach()
    geometric_trainer.train()

    torch.testing.assert_close(native_loss, geometric_loss, rtol=0, atol=0)
    for native_parameter, geometric_parameter in zip(
        native_model.parameters(),
        geometric_model.parameters(),
    ):
        torch.testing.assert_close(
            native_parameter,
            geometric_parameter,
            rtol=1e-7,
            atol=1e-8,
        )


def test_geometric_yaml_uses_supported_approximate_adam_contract():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs/trainer/GeometricUnlearn.yaml")

    assert config.args.optim == "adamw_torch"
    assert config.args.adam_beta1 == 0.0
    assert config.args.weight_decay == 0.0
    assert config.args.fp16 is False
    assert config.args.max_steps == -1
    assert config.args.gradient_checkpointing_kwargs.use_reentrant is False
    assert config.method_args.geometric_config.gu_enabled is True
    assert config.method_args.geometric_config.gradient_surgery == "pcgrad"
    assert config.method_args.geometric_config.sign_selective is False
    assert config.method_args.geometric_config.trainable_params_regex == [".*"]
    assert config.method_args.simnpo_config.gamma == 0.125
    assert config.method_args.simnpo_config.retain_loss_type == "NLL"
    assert "null_k" not in config.method_args.geometric_config


def test_training_phase_can_skip_all_checkpoint_writes():
    from train import run_training_phase

    class RecordingTrainer:
        def __init__(self):
            self.events = []

        def train(self):
            self.events.append("train")

        def save_state(self):
            self.events.append("save_state")

        def save_model(self, output_dir):
            self.events.append(("save_model", output_dir))

    trainer = RecordingTrainer()
    args = SimpleNamespace(do_train=True, output_dir="unused")

    run_training_phase(
        trainer,
        args,
        save_model_after_train=False,
    )

    assert trainer.events == ["train"]


def test_unlearn_config_defaults_to_preserving_existing_save_behavior():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs/unlearn.yaml")

    assert config.save_model_after_train is True
