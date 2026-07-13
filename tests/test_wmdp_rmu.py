import copy
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path
from transformers import TrainerCallback, TrainingArguments

from trainer import TRAINER_REGISTRY
from tests.helpers import nested_collator
from tests.test_wmdp_uam import (
    ToyWMDPCausalLM,
    make_batch,
    unbatch_wmdp_dataset,
)
from trainer.unlearn.wmdp_selection import EXPECTED_WMDP_PARAMETER_NAMES
from trainer.unlearn.wmdp_rmu import WMDPRMUUnlearn


def make_wmdp_rmu_trainer(tmp_path, *, model=None, **overrides):
    model = ToyWMDPCausalLM() if model is None else model
    train_dataset = overrides.pop("train_dataset", None)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=overrides.pop(
            "per_device_train_batch_size",
            1,
        ),
        gradient_accumulation_steps=overrides.pop(
            "gradient_accumulation_steps",
            1,
        ),
        max_steps=overrides.pop("max_steps", 1),
        learning_rate=5e-5,
        optim="adamw_torch",
        adam_beta1=0.0,
        weight_decay=0.0,
        max_grad_norm=0.0,
        remove_unused_columns=False,
        disable_tqdm=True,
        save_strategy="no",
    )
    geometric_config = SimpleNamespace(
        loss="rmu",
        gu_enabled=overrides.pop("gu_enabled", False),
        gradient_surgery="gu",
        projection_eps=1e-12,
        trainable_params_regex=[
            r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"
        ],
        component_buffer_device="parameter",
        diagnostics_path=None,
        actual_delta_mode="off",
        actual_delta_steps=[1, 80],
        actual_delta_sample_elements=1_000,
    )
    rmu_config = SimpleNamespace(
        module_name=overrides.pop("module_name", "model.layers.7"),
        steering_coeff=overrides.pop("steering_coeff", 20.0),
        seed=overrides.pop("seed", 42),
        require_paged_adamw=overrides.pop("require_paged_adamw", False),
    )
    gamma = overrides.pop("gamma", 1.0)
    alpha = overrides.pop("alpha", 100.0)
    assert not overrides, overrides
    trainer = WMDPRMUUnlearn(
        model=model,
        args=args,
        gamma=gamma,
        alpha=alpha,
        retain_loss_type="EMBED_DIFF",
        geometric_config=geometric_config,
        rmu_config=rmu_config,
        simnpo_config=None,
        npo_config=None,
        dpo_config=None,
        undial_config=None,
        wga_config=None,
        satimp_config=None,
        train_dataset=train_dataset,
        data_collator=nested_collator if train_dataset is not None else None,
    )
    return trainer, model


def test_wmdp_rmu_freezes_reference_and_selects_exact_parameters(tmp_path):
    trainer, model = make_wmdp_rmu_trainer(tmp_path)

    assert trainer.ref_model.training is False
    assert all(not parameter.requires_grad for parameter in trainer.ref_model.parameters())
    assert tuple(name for name, _ in trainer._selected_named_parameters()) == (
        EXPECTED_WMDP_PARAMETER_NAMES
    )
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == set(EXPECTED_WMDP_PARAMETER_NAMES)


def test_wmdp_rmu_is_registered_and_yaml_resolves_w2_contract():
    assert TRAINER_REGISTRY["WMDPRMUUnlearn"] is WMDPRMUUnlearn
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="unlearn",
            overrides=["trainer=WMDPRMUUnlearn"],
        )

    trainer = config.trainer
    assert trainer.handler == "WMDPRMUUnlearn"
    assert trainer.args.per_device_train_batch_size == 1
    assert trainer.args.gradient_accumulation_steps == 4
    assert trainer.args.max_steps == 80
    assert trainer.args.learning_rate == pytest.approx(5e-5)
    assert trainer.args.bf16 is True
    assert trainer.args.fp16 is False
    assert trainer.args.optim == "paged_adamw_32bit"
    assert trainer.args.adam_beta1 == 0.0
    assert trainer.args.weight_decay == 0.0
    assert trainer.args.save_strategy == "no"
    assert trainer.method_args.gamma == 1.0
    assert trainer.method_args.alpha == 100.0
    assert trainer.method_args.geometric_config.loss == "rmu"
    assert trainer.method_args.geometric_config.gu_enabled is False
    assert trainer.method_args.geometric_config.actual_delta_mode == "full"
    assert trainer.method_args.geometric_config.actual_delta_steps == [1, 80]
    assert trainer.method_args.rmu_config.steering_coeff == 20.0
    assert trainer.method_args.rmu_config.require_paged_adamw is True


def test_control_vector_is_seeded_unit_direction_scaled_by_twenty(tmp_path):
    torch.manual_seed(11)
    base = ToyWMDPCausalLM()
    first, first_model = make_wmdp_rmu_trainer(
        tmp_path / "first",
        model=copy.deepcopy(base),
        seed=91,
    )
    second, second_model = make_wmdp_rmu_trainer(
        tmp_path / "second",
        model=copy.deepcopy(base),
        seed=91,
    )
    batch = make_batch(seed=12)

    first_loss, _ = first.compute_forget_loss(first_model, batch)
    cached = first.control_vector.clone()
    repeated_loss, _ = first.compute_forget_loss(first_model, batch)
    second_loss, _ = second.compute_forget_loss(second_model, batch)

    assert torch.linalg.vector_norm(cached).item() == pytest.approx(20.0)
    assert torch.equal(first.control_vector, cached)
    assert torch.equal(first.control_vector, second.control_vector)
    torch.testing.assert_close(first_loss, repeated_loss, rtol=0, atol=0)
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)


def test_retain_representation_loss_is_zero_at_initialization(tmp_path):
    trainer, model = make_wmdp_rmu_trainer(tmp_path)

    loss = trainer.compute_retain_loss(model, make_batch(seed=13))

    assert loss.item() == pytest.approx(0.0, abs=0.0)
    assert loss.requires_grad is True


@pytest.mark.parametrize("gu_enabled", [False, True])
def test_component_loss_uses_rmu_plus_hundred_retain_weight(
    tmp_path,
    gu_enabled,
):
    trainer, model = make_wmdp_rmu_trainer(tmp_path, gu_enabled=gu_enabled)
    inputs = {"forget": make_batch(seed=14), "retain": make_batch(seed=15)}

    forget, retain, outputs = trainer.compute_component_losses(model, inputs)
    total = trainer.compute_loss(model, inputs)

    assert outputs.logits.shape[:2] == inputs["forget"]["input_ids"].shape
    torch.testing.assert_close(total, forget + 100.0 * retain)


def test_optimizer_contains_only_the_three_selected_parameters(tmp_path):
    trainer, _ = make_wmdp_rmu_trainer(tmp_path)

    trainer.create_optimizer()

    optimizer_parameters = [
        parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    ]
    selected = trainer._selected_named_parameters()
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter) for _, parameter in selected
    }


@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("gamma", 2.0, "gamma=1"),
        ("alpha", 1.0, "alpha=100"),
        ("steering_coeff", 2.0, "steering_coeff=20"),
        ("module_name", "model.layers.6", "model.layers.7"),
    ],
)
def test_wmdp_rmu_runtime_rejects_nonmatched_settings(
    tmp_path,
    override,
    value,
    message,
):
    trainer, _ = make_wmdp_rmu_trainer(tmp_path, **{override: value})
    trainer.create_optimizer()

    with pytest.raises((ValueError, NotImplementedError), match=message):
        trainer._validate_wmdp_runtime()


class CaptureSelectedGradient(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.gradients = None

    def on_pre_optimizer_step(self, _args, _state, control, **_kwargs):
        self.gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in self.trainer._selected_named_parameters()
        }
        return control


@pytest.mark.parametrize("gu_enabled", [False, True])
def test_wmdp_rmu_gas_matches_effective_batch_update(tmp_path, gu_enabled):
    torch.manual_seed(201)
    base = ToyWMDPCausalLM()
    full_model = copy.deepcopy(base)
    accumulated_model = copy.deepcopy(base)
    dataset = unbatch_wmdp_dataset(
        {
            "forget": make_batch(seed=202, batch_size=4),
            "retain": make_batch(seed=203, batch_size=4),
        }
    )
    full, _ = make_wmdp_rmu_trainer(
        tmp_path / "full",
        model=full_model,
        gu_enabled=gu_enabled,
        train_dataset=dataset,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
    )
    accumulated, _ = make_wmdp_rmu_trainer(
        tmp_path / "accumulated",
        model=accumulated_model,
        gu_enabled=gu_enabled,
        train_dataset=dataset,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
    )

    full.train()
    accumulated.train()

    expected_calls = 1 if gu_enabled else 0
    assert full.gu_projection_calls == accumulated.gu_projection_calls == expected_calls
    for name in EXPECTED_WMDP_PARAMETER_NAMES:
        torch.testing.assert_close(
            full_model.get_parameter(name),
            accumulated_model.get_parameter(name),
            rtol=1e-5,
            atol=1e-6,
        )


def test_wmdp_rmu_baseline_writes_component_sum_gradient(tmp_path):
    torch.manual_seed(204)
    model = ToyWMDPCausalLM()
    batch = {
        "forget": make_batch(seed=205, batch_size=4),
        "retain": make_batch(seed=206, batch_size=4),
    }
    trainer, _ = make_wmdp_rmu_trainer(
        tmp_path,
        model=model,
        gu_enabled=False,
        train_dataset=unbatch_wmdp_dataset(batch),
        per_device_train_batch_size=4,
    )
    forget, retain, _ = trainer.compute_component_losses(model, batch)
    named_params = trainer._selected_named_parameters()
    params = [parameter for _, parameter in named_params]
    forget_grads = torch.autograd.grad(forget, params, retain_graph=True)
    retain_grads = torch.autograd.grad(retain, params)
    expected = {
        name: gf + 100.0 * gr
        for (name, _), gf, gr in zip(
            named_params,
            forget_grads,
            retain_grads,
            strict=True,
        )
    }
    capture = CaptureSelectedGradient(trainer)
    trainer.add_callback(capture)

    trainer.train()

    assert capture.gradients is not None
    assert trainer.rmu_finalizer_calls == trainer.state.global_step == 1
    assert trainer.last_rmu_diagnostics["record_type"] == "rmu_geometry"
    assert trainer.last_rmu_diagnostics["finalizer_calls"] == 1
    for name in EXPECTED_WMDP_PARAMETER_NAMES:
        torch.testing.assert_close(
            capture.gradients[name],
            expected[name],
            rtol=1e-5,
            atol=1e-6,
        )


def test_wmdp_rmu_gu_writes_one_global_projected_gradient(tmp_path):
    torch.manual_seed(207)
    model = ToyWMDPCausalLM()
    batch = {
        "forget": make_batch(seed=208, batch_size=4),
        "retain": make_batch(seed=209, batch_size=4),
    }
    trainer, _ = make_wmdp_rmu_trainer(
        tmp_path,
        model=model,
        gu_enabled=True,
        train_dataset=unbatch_wmdp_dataset(batch),
        per_device_train_batch_size=4,
    )
    forget, retain, _ = trainer.compute_component_losses(model, batch)
    named_params = trainer._selected_named_parameters()
    params = [parameter for _, parameter in named_params]
    forget_grads = torch.autograd.grad(forget, params, retain_graph=True)
    retain_grads = torch.autograd.grad(retain, params)
    dot = sum((gf * gr).sum() for gf, gr in zip(forget_grads, retain_grads))
    retain_sq = sum(gr.square().sum() for gr in retain_grads)
    coefficient = dot / retain_sq.clamp_min(1e-12)
    expected = {
        name: gf - coefficient * gr + 100.0 * gr
        for (name, _), gf, gr in zip(
            named_params,
            forget_grads,
            retain_grads,
            strict=True,
        )
    }
    capture = CaptureSelectedGradient(trainer)
    trainer.add_callback(capture)

    trainer.train()

    assert trainer.gu_projection_calls == trainer.state.global_step == 1
    assert trainer.last_gu_diagnostics["relative_orthogonality_residual"] < 1e-6
    for name in EXPECTED_WMDP_PARAMETER_NAMES:
        torch.testing.assert_close(
            capture.gradients[name],
            expected[name],
            rtol=1e-5,
            atol=1e-6,
        )
