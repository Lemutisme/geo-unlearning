import pytest
import torch

from trainer.unlearn.orthograd_geometry import GlobalRetainBasis


def test_projection_is_globally_orthogonal_to_each_retain_direction():
    basis = GlobalRetainBasis(rank_tolerance=1e-8)
    retain_a = {
        "left": torch.tensor([1.0, 0.0]),
        "right": torch.tensor([1.0]),
    }
    retain_b = {
        "left": torch.tensor([0.0, 1.0]),
        "right": torch.tensor([1.0]),
    }
    assert basis.add(retain_a)
    assert basis.add(retain_b)

    candidate = {
        "left": torch.tensor([3.0, -2.0]),
        "right": torch.tensor([4.0]),
    }
    projected, removed, stats = basis.project(candidate)

    for retain in (retain_a, retain_b):
        assert GlobalRetainBasis.global_dot(
            projected,
            retain,
        ).item() == pytest.approx(0.0, abs=1e-6)
    for name in candidate:
        torch.testing.assert_close(
            projected[name] + removed[name],
            candidate[name].float(),
        )
    assert stats.effective_rank == 2


def test_duplicate_retain_direction_is_dropped_and_storage_is_exact():
    basis = GlobalRetainBasis(rank_tolerance=1e-8)
    retain = {
        "left": torch.tensor([1.0, 2.0]),
        "right": torch.tensor([3.0]),
    }

    assert basis.add(retain)
    assert not basis.add({name: 2.0 * value for name, value in retain.items()})

    assert basis.requested_rank == 2
    assert basis.effective_rank == 1
    assert basis.dropped_directions == 1
    assert basis.basis_bytes == 3 * torch.finfo(torch.float32).bits // 8


def test_global_projection_differs_from_independent_blockwise_projection():
    basis = GlobalRetainBasis(rank_tolerance=1e-8)
    retain = {
        "left": torch.tensor([1.0, 0.0]),
        "right": torch.tensor([1.0, 1.0]),
    }
    candidate = {
        "left": torch.tensor([2.0, 3.0]),
        "right": torch.tensor([-1.0, 4.0]),
    }
    assert basis.add(retain)

    projected, _, _ = basis.project(candidate)
    blockwise = {}
    for name in candidate:
        coefficient = torch.dot(candidate[name], retain[name]) / torch.dot(
            retain[name], retain[name]
        )
        blockwise[name] = candidate[name] - coefficient * retain[name]

    assert any(
        not torch.allclose(projected[name], blockwise[name]) for name in candidate
    )


def test_empty_and_nonfinite_directions_are_rejected():
    basis = GlobalRetainBasis(rank_tolerance=1e-8)

    with pytest.raises(RuntimeError, match="empty"):
        basis.add({})
    with pytest.raises(RuntimeError, match="non-finite"):
        basis.add({"weight": torch.tensor([float("nan")])})


def test_clear_releases_every_basis_tensor():
    basis = GlobalRetainBasis(rank_tolerance=1e-8)
    assert basis.add({"weight": torch.tensor([1.0, 0.0])})

    basis.clear()

    assert basis.empty
    assert basis.effective_rank == 0
    assert basis.requested_rank == 0
    assert basis.dropped_directions == 0
