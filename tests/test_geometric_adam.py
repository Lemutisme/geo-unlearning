from types import SimpleNamespace

import pytest
import torch
from transformers import TrainingArguments

from trainer.unlearn.geometric import GeometricUnlearn
from tests.helpers import TinyCausalLM, make_unlearn_batch


def make_geometric_trainer(
    tmp_path,
    *,
    model=None,
    gu_enabled=True,
    gamma=0.125,
    alpha=1.0,
):
    model = TinyCausalLM() if model is None else model
    geometric_config = SimpleNamespace(
        loss="simnpo",
        gu_enabled=gu_enabled,
        projection_eps=1e-12,
        trainable_params_regex=[".*"],
        auto_last_k_layers=1,
        null_k=1,
    )
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        adam_beta1=0.0,
        weight_decay=0.0,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    trainer = GeometricUnlearn(
        model=model,
        args=args,
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
    )
    return trainer, model, simnpo_config


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
        name: -raw_projected[name] / sqrt_h[name].square()
        for name in raw_projected
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


def test_unused_optimizer_step_hook_is_removed():
    assert "optimizer_step" not in GeometricUnlearn.__dict__


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
