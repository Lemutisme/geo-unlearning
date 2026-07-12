import pytest
import torch

from trainer.unlearn.gradient_surgery import (
    apply_surgery_tensor,
    decide_global_surgery,
)


def test_negative_dot_pcgrad_exactly_matches_gu():
    dot = torch.tensor(-2.0)
    retain_sq = torch.tensor(4.0)

    gu = decide_global_surgery("gu", dot, retain_sq, 1e-12)
    pcgrad = decide_global_surgery("pcgrad", dot, retain_sq, 1e-12)

    assert gu.coefficient.item() == pytest.approx(-0.5)
    assert pcgrad.coefficient.item() == pytest.approx(-0.5)
    assert pcgrad.conflict is True


def test_nonnegative_dot_pcgrad_is_identity_while_gu_projects():
    dot = torch.tensor(2.0)
    retain_sq = torch.tensor(4.0)

    gu = decide_global_surgery("gu", dot, retain_sq, 1e-12)
    pcgrad = decide_global_surgery("pcgrad", dot, retain_sq, 1e-12)

    assert pcgrad.coefficient.item() == 0.0
    assert pcgrad.conflict is False
    assert gu.coefficient.item() == pytest.approx(0.5)


def test_pcgrad_uses_one_global_coefficient_across_tensors():
    forget = {
        "a": torch.tensor([2.0]),
        "b": torch.tensor([1.0]),
    }
    retain = {
        "a": torch.tensor([-1.0]),
        "b": torch.tensor([-3.0]),
    }
    dot = sum((forget[name] * retain[name]).sum() for name in forget)
    retain_sq = sum(retain[name].square().sum() for name in retain)

    decision = decide_global_surgery("pcgrad", dot, retain_sq, 1e-12)
    projected = {
        name: apply_surgery_tensor(forget[name], retain[name], decision)
        for name in forget
    }

    assert decision.coefficient.item() == pytest.approx(-0.5)
    assert projected["a"].item() == pytest.approx(1.5)
    assert projected["b"].item() == pytest.approx(-0.5)
    projected_dot = sum((projected[name] * retain[name]).sum() for name in retain)
    assert projected_dot.item() == pytest.approx(0.0, abs=1e-6)


def test_global_surgery_differs_from_blockwise_projection():
    forget = {
        "a": torch.tensor([2.0]),
        "b": torch.tensor([1.0]),
    }
    retain = {
        "a": torch.tensor([-1.0]),
        "b": torch.tensor([-3.0]),
    }
    dot = sum((forget[name] * retain[name]).sum() for name in forget)
    retain_sq = sum(retain[name].square().sum() for name in retain)
    decision = decide_global_surgery("pcgrad", dot, retain_sq, 1e-12)

    global_a = apply_surgery_tensor(forget["a"], retain["a"], decision)
    block_a = (
        forget["a"]
        - ((forget["a"] * retain["a"]).sum() / retain["a"].square().sum()) * retain["a"]
    )

    assert global_a.item() == pytest.approx(1.5)
    assert block_a.item() == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["gu", "pcgrad"])
def test_zero_retain_gradient_has_finite_zero_coefficient(mode):
    forget = torch.tensor([2.0, -3.0])
    retain = torch.zeros(2)

    decision = decide_global_surgery(
        mode,
        (forget * retain).sum(),
        retain.square().sum(),
        1e-12,
    )
    projected = apply_surgery_tensor(forget, retain, decision)

    assert decision.coefficient.item() == 0.0
    assert torch.isfinite(decision.coefficient)
    torch.testing.assert_close(projected, forget)


def test_unknown_surgery_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported gradient surgery mode"):
        decide_global_surgery("unknown", torch.tensor(0.0), torch.tensor(1.0), 1e-12)
