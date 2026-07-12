import pytest
import torch

from trainer.unlearn.uam_geometry import (
    PerturbationDecision,
    ResidualGUDecision,
    UAMDecision,
    apply_residual_gu_tensor,
    apply_uam_tensor,
    decide_perturbation,
    decide_residual_gu,
    decide_uam,
)


def test_fixed_loss_perturbation_has_requested_linearized_increase():
    forget = torch.tensor([3.0, 4.0])
    rho = 0.2

    decision = decide_perturbation(
        "fixed_loss",
        forget.square().sum(),
        torch.tensor(0.0),
        rho,
        1e-12,
    )
    perturbation = decision.coefficient * forget

    assert isinstance(decision, PerturbationDecision)
    assert decision.mode == "fixed_loss"
    assert torch.dot(forget, perturbation).item() == pytest.approx(rho)


def test_metric_trust_perturbation_has_requested_metric_radius():
    forget = torch.tensor([2.0, -1.0])
    denominator = torch.tensor([4.0, 0.25])
    optimizer_forget_sq = (forget / denominator).square().sum()
    rho = 0.3

    decision = decide_perturbation(
        "metric_trust",
        forget.square().sum(),
        optimizer_forget_sq,
        rho,
        1e-12,
    )
    perturbation = decision.coefficient * forget / denominator

    assert decision.mode == "metric_trust"
    assert perturbation.norm().item() == pytest.approx(rho)


def test_gamma_two_uam_is_a_householder_reflection():
    forget = torch.tensor([1.0, 0.0], dtype=torch.float16)
    perturbed_retain = torch.tensor([3.0, 4.0], dtype=torch.float16)

    decision = decide_uam(
        torch.tensor(3.0),
        torch.tensor(1.0),
        reflection_gamma=2.0,
        eps=1e-12,
    )
    reflected = apply_uam_tensor(
        perturbed_retain,
        forget,
        decision,
    )

    assert isinstance(decision, UAMDecision)
    assert reflected.dtype == torch.float32
    torch.testing.assert_close(reflected, torch.tensor([-3.0, 4.0]))


def test_residual_gu_uses_one_global_projection_and_negative_forget_gate():
    retain = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([0.0, 2.0]),
    }
    residual = {
        "a": torch.tensor([3.0, 4.0]),
        "b": torch.tensor([5.0, 6.0]),
    }
    forget = {
        "a": torch.tensor([0.0, -1.0]),
        "b": torch.tensor([-1.0, 0.0]),
    }

    residual_retain_dot = sum(
        torch.dot(residual[name], retain[name]) for name in retain
    )
    residual_forget_dot = sum(
        torch.dot(residual[name], forget[name]) for name in retain
    )
    forget_retain_dot = sum(torch.dot(forget[name], retain[name]) for name in retain)
    optimizer_retain_sq = sum(value.square().sum() for value in retain.values())
    decision = decide_residual_gu(
        residual_retain_dot,
        residual_forget_dot,
        forget_retain_dot,
        optimizer_retain_sq,
        residual_lambda=0.5,
        sign_tau=1e-8,
        eps=1e-12,
    )

    normal = {}
    final = {}
    for name in retain:
        final[name], normal[name] = apply_residual_gu_tensor(
            retain[name] + residual[name],
            retain[name],
            decision,
        )

    global_normal_retain_dot = sum(
        torch.dot(normal[name], retain[name]) for name in retain
    )
    assert isinstance(decision, ResidualGUDecision)
    assert decision.projection_coefficient.item() == pytest.approx(3.0)
    assert decision.gate_dot.item() == pytest.approx(-9.0)
    assert decision.keep is True
    assert global_normal_retain_dot.item() == pytest.approx(0.0, abs=1e-6)
    for name in retain:
        torch.testing.assert_close(final[name], retain[name] + 0.5 * normal[name])


def test_residual_gu_gate_off_returns_retain_gradient():
    retain = torch.tensor([1.0, 0.0])
    residual = torch.tensor([0.0, 2.0])
    forget = torch.tensor([0.0, 1.0])
    decision = decide_residual_gu(
        torch.dot(residual, retain),
        torch.dot(residual, forget),
        torch.dot(forget, retain),
        retain.square().sum(),
        residual_lambda=0.5,
        sign_tau=1e-8,
        eps=1e-12,
    )

    final, normal = apply_residual_gu_tensor(
        retain + residual,
        retain,
        decision,
    )

    assert decision.keep is False
    torch.testing.assert_close(normal, residual)
    torch.testing.assert_close(final, retain)


def test_residual_projection_preserves_retain_descent_that_total_projection_deletes():
    retain = torch.tensor([1.0, 0.0])
    uam = torch.tensor([3.0, 4.0])
    forget = torch.tensor([0.0, -1.0])
    residual = uam - retain

    direct_decision = decide_residual_gu(
        torch.dot(uam, retain),
        torch.dot(uam, forget),
        torch.dot(forget, retain),
        retain.square().sum(),
        residual_lambda=1.0,
        sign_tau=0.0,
        eps=1e-12,
    )
    direct_projection = uam - direct_decision.projection_coefficient * retain

    residual_decision = decide_residual_gu(
        torch.dot(residual, retain),
        torch.dot(residual, forget),
        torch.dot(forget, retain),
        retain.square().sum(),
        residual_lambda=1.0,
        sign_tau=0.0,
        eps=1e-12,
    )
    final, _ = apply_residual_gu_tensor(
        uam,
        retain,
        residual_decision,
    )

    assert torch.dot(direct_projection, retain).item() == pytest.approx(0.0)
    assert torch.dot(final, retain).item() == pytest.approx(
        retain.square().sum().item()
    )


@pytest.mark.parametrize("mode", ["fixed_loss", "metric_trust"])
def test_zero_forget_norm_fails_closed(mode):
    with pytest.raises(RuntimeError, match="UAM forget gradient has zero norm"):
        decide_perturbation(
            mode,
            torch.tensor(0.0),
            torch.tensor(0.0),
            rho=0.2,
            eps=1e-12,
        )


def test_zero_retain_norm_fails_closed():
    with pytest.raises(
        RuntimeError,
        match="Residual-GU-UAM retain gradient has zero norm",
    ):
        decide_residual_gu(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            residual_lambda=0.5,
            sign_tau=1e-8,
            eps=1e-12,
        )


def test_unknown_perturbation_normalization_fails_closed():
    with pytest.raises(
        ValueError,
        match="Unsupported UAM perturbation normalization",
    ):
        decide_perturbation(
            "unknown",
            torch.tensor(1.0),
            torch.tensor(1.0),
            rho=0.2,
            eps=1e-12,
        )
