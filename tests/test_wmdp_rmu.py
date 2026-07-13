import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import TrainingArguments

from tests.test_wmdp_uam import ToyWMDPCausalLM, make_batch
from trainer.unlearn.wmdp_selection import EXPECTED_WMDP_PARAMETER_NAMES
from trainer.unlearn.wmdp_rmu import WMDPRMUUnlearn


def make_wmdp_rmu_trainer(tmp_path, *, model=None, **overrides):
    model = ToyWMDPCausalLM() if model is None else model
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
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
