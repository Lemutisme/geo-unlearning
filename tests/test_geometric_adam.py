import pytest
import torch

from trainer.unlearn.geometric import GeometricUnlearn
from tests.helpers import make_unlearn_batch


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
