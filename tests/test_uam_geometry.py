import pytest
import torch

from trainer.unlearn.uam_geometry import (
    PerturbationDecision,
    ResidualGUDecision,
    ResidualGUProjection,
    UAMDecision,
    apply_residual_gu_tensor,
    apply_uam_tensor,
    decide_perturbation,
    decide_residual_gu,
    decide_residual_gu_gate,
    decide_uam,
    project_residual_gu_tensor,
)


def true_relative_orthogonality(normal, retain):
    normal64 = normal.double()
    retain64 = retain.double()
    normal_norm = normal64.square().sum().sqrt()
    if normal_norm.item() == 0.0:
        return torch.zeros((), dtype=torch.float64)
    return (normal64 * retain64).sum().abs() / (
        normal_norm * retain64.square().sum().sqrt()
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
    optimizer_forget_sq = (forget / denominator.sqrt()).square().sum()
    rho = 0.3

    decision = decide_perturbation(
        "metric_trust",
        forget.square().sum(),
        optimizer_forget_sq,
        rho,
        1e-12,
    )
    perturbation = decision.coefficient * forget / denominator
    metric_radius = (denominator * perturbation.square()).sum().sqrt()

    assert decision.mode == "metric_trust"
    assert metric_radius.item() == pytest.approx(rho)


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


def test_uam_projection_uses_additive_epsilon_at_regularization_scale():
    eps = 1e-4

    decision = decide_uam(
        forget_perturbed_retain_dot=torch.tensor(eps),
        optimizer_forget_sq=torch.tensor(eps),
        reflection_gamma=1.0,
        eps=eps,
    )

    assert decision.coefficient.item() == pytest.approx(0.5)


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
    optimizer_retain_sq = sum(value.square().sum() for value in retain.values())
    projection = decide_residual_gu(
        residual_retain_dot,
        optimizer_retain_sq,
        residual_lambda=0.5,
        eps=1e-12,
    )

    normal = {
        name: project_residual_gu_tensor(
            retain[name] + residual[name],
            retain[name],
            projection,
        )
        for name in retain
    }
    normal_forget_dot = sum(torch.dot(normal[name], forget[name]) for name in retain)
    normal_retain_dot64 = sum(
        (normal[name].double() * retain[name].double()).sum() for name in retain
    )
    normal_sq64 = sum(normal[name].double().square().sum() for name in retain)
    retain_sq64 = sum(retain[name].double().square().sum() for name in retain)
    relative_orthogonality = normal_retain_dot64.abs() / (
        normal_sq64.sqrt() * retain_sq64.sqrt()
    )
    decision = decide_residual_gu_gate(
        projection,
        normal_forget_dot,
        relative_orthogonality,
        sign_tau=1e-8,
    )
    final = {}
    for name in retain:
        final[name], applied_normal = apply_residual_gu_tensor(
            retain[name] + residual[name],
            retain[name],
            decision,
        )
        torch.testing.assert_close(applied_normal, normal[name])

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


def test_residual_gu_projection_uses_additive_epsilon_at_regularization_scale():
    eps = 1e-4

    projection = decide_residual_gu(
        residual_retain_dot=torch.tensor(eps),
        optimizer_retain_sq=torch.tensor(eps),
        residual_lambda=0.5,
        eps=eps,
    )

    assert projection.projection_coefficient.item() == pytest.approx(0.5)


def test_residual_gu_gate_uses_dot_of_applied_fp32_normal():
    retain = torch.tensor([-7343.7060546875])
    uam = torch.tensor([-21389.1171875])
    forget = torch.tensor([-37464.50390625])
    residual = uam - retain
    residual_forget_dot = torch.dot(residual, forget)
    forget_retain_dot = torch.dot(forget, retain)
    projection = decide_residual_gu(
        residual_retain_dot=torch.dot(residual, retain),
        optimizer_retain_sq=retain.square().sum(),
        residual_lambda=1.0,
        eps=1e-12,
    )

    normal = project_residual_gu_tensor(uam, retain, projection)
    applied_normal_dot = torch.dot(normal, forget)
    decision = decide_residual_gu_gate(
        projection,
        applied_normal_dot,
        true_relative_orthogonality(normal, retain),
        sign_tau=0.0,
    )
    final, normal = apply_residual_gu_tensor(uam, retain, decision)
    expanded_gate = (
        residual_forget_dot - projection.projection_coefficient * forget_retain_dot
    )

    assert expanded_gate.item() == pytest.approx(-32.0)
    assert normal.item() == pytest.approx(-0.0009765625)
    assert applied_normal_dot.item() == pytest.approx(36.58643)
    assert decision.gate_dot.item() == pytest.approx(applied_normal_dot.item())
    assert decision.keep is False
    assert torch.equal(final, retain)


def test_residual_gu_gate_rejects_regularized_nonorthogonal_normal():
    retain = torch.tensor([1e-10])
    uam = torch.tensor([2e-10])
    forget = torch.tensor([-1e-10])
    residual = uam - retain
    projection = decide_residual_gu(
        residual_retain_dot=torch.dot(residual, retain),
        optimizer_retain_sq=retain.square().sum(),
        residual_lambda=1.0,
        eps=1e-12,
    )
    normal = project_residual_gu_tensor(uam, retain, projection)
    normal64 = normal.double()
    retain64 = retain.double()
    normal_retain_dot = (normal64 * retain64).sum()
    relative_orthogonality = normal_retain_dot.abs() / (
        normal64.square().sum().sqrt() * retain64.square().sum().sqrt()
    )
    decision = decide_residual_gu_gate(
        projection,
        (normal64 * forget.double()).sum(),
        relative_orthogonality,
        sign_tau=0.0,
    )

    final, _ = apply_residual_gu_tensor(uam, retain, decision)

    assert relative_orthogonality.item() == pytest.approx(1.0)
    assert decision.sign_gate_passed is True
    assert decision.orthogonality_safe is False
    assert decision.keep is False
    assert torch.equal(final, retain)


def test_residual_gu_gate_off_returns_retain_gradient():
    retain = torch.tensor([1.0, 0.0])
    residual = torch.tensor([0.0, 2.0])
    forget = torch.tensor([0.0, 1.0])
    projection = decide_residual_gu(
        torch.dot(residual, retain),
        retain.square().sum(),
        residual_lambda=0.5,
        eps=1e-12,
    )
    normal = project_residual_gu_tensor(retain + residual, retain, projection)
    decision = decide_residual_gu_gate(
        projection,
        torch.dot(normal, forget),
        true_relative_orthogonality(normal, retain),
        sign_tau=1e-8,
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

    direct_projection_decision = decide_residual_gu(
        torch.dot(uam, retain),
        retain.square().sum(),
        residual_lambda=1.0,
        eps=1e-12,
    )
    direct_projection = uam - direct_projection_decision.projection_coefficient * retain

    residual_projection = decide_residual_gu(
        torch.dot(residual, retain),
        retain.square().sum(),
        residual_lambda=1.0,
        eps=1e-12,
    )
    normal = project_residual_gu_tensor(uam, retain, residual_projection)
    residual_decision = decide_residual_gu_gate(
        residual_projection,
        torch.dot(normal, forget),
        true_relative_orthogonality(normal, retain),
        sign_tau=0.0,
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
            residual_lambda=0.5,
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


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
@pytest.mark.parametrize(
    "field",
    ["raw_forget_sq", "optimizer_forget_sq", "rho", "eps"],
)
def test_perturbation_non_finite_geometry_fails_closed(field, non_finite):
    values = {
        "raw_forget_sq": torch.tensor(1.0),
        "optimizer_forget_sq": torch.tensor(1.0),
        "rho": 0.2,
        "eps": 1e-12,
    }
    values[field] = (
        torch.tensor(non_finite)
        if field in {"raw_forget_sq", "optimizer_forget_sq"}
        else non_finite
    )

    with pytest.raises((ValueError, RuntimeError), match="finite"):
        decide_perturbation("metric_trust", **values)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
@pytest.mark.parametrize(
    "field",
    [
        "forget_perturbed_retain_dot",
        "optimizer_forget_sq",
        "reflection_gamma",
        "eps",
    ],
)
def test_uam_non_finite_geometry_fails_closed(field, non_finite):
    values = {
        "forget_perturbed_retain_dot": torch.tensor(1.0),
        "optimizer_forget_sq": torch.tensor(1.0),
        "reflection_gamma": 2.0,
        "eps": 1e-12,
    }
    values[field] = (
        torch.tensor(non_finite)
        if field
        in {
            "forget_perturbed_retain_dot",
            "optimizer_forget_sq",
        }
        else non_finite
    )

    with pytest.raises((ValueError, RuntimeError), match="finite"):
        decide_uam(**values)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
@pytest.mark.parametrize(
    "field",
    [
        "residual_retain_dot",
        "optimizer_retain_sq",
        "residual_lambda",
        "eps",
    ],
)
def test_residual_gu_non_finite_geometry_fails_closed(field, non_finite):
    values = {
        "residual_retain_dot": torch.tensor(1.0),
        "optimizer_retain_sq": torch.tensor(2.0),
        "residual_lambda": 0.5,
        "eps": 1e-12,
    }
    values[field] = (
        torch.tensor(non_finite)
        if field in {"residual_retain_dot", "optimizer_retain_sq"}
        else non_finite
    )

    with pytest.raises((ValueError, RuntimeError), match="finite"):
        decide_residual_gu(**values)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
@pytest.mark.parametrize(
    "field",
    ["normal_forget_dot", "relative_orthogonality", "sign_tau"],
)
def test_residual_gu_gate_non_finite_geometry_fails_closed(field, non_finite):
    projection = ResidualGUProjection(
        projection_coefficient=torch.tensor(0.5),
        residual_lambda=0.5,
    )
    values = {
        "projection": projection,
        "normal_forget_dot": torch.tensor(-1.0),
        "relative_orthogonality": torch.tensor(0.0),
        "sign_tau": 1e-8,
    }
    values[field] = (
        torch.tensor(non_finite)
        if field in {"normal_forget_dot", "relative_orthogonality"}
        else non_finite
    )

    with pytest.raises((ValueError, RuntimeError), match="finite"):
        decide_residual_gu_gate(**values)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["perturbed_retain", "forget"])
def test_apply_uam_non_finite_inputs_fail_closed(field, non_finite):
    tensors = {
        "perturbed_retain": torch.tensor([1.0, 2.0]),
        "forget": torch.tensor([3.0, 4.0]),
    }
    tensors[field] = torch.tensor([non_finite, 1.0])
    decision = UAMDecision(coefficient=torch.tensor(0.5))

    with pytest.raises((ValueError, RuntimeError), match="finite"):
        apply_uam_tensor(**tensors, decision=decision)


def test_apply_uam_non_finite_derived_tensor_fails_closed():
    decision = UAMDecision(coefficient=torch.tensor(1.0))

    with pytest.raises((ValueError, RuntimeError), match="finite"):
        apply_uam_tensor(
            torch.tensor([3e38]),
            torch.tensor([-3e38]),
            decision,
        )


@pytest.mark.parametrize("keep", [True, False])
@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["uam", "retain"])
def test_apply_residual_gu_non_finite_inputs_fail_closed(
    field,
    non_finite,
    keep,
):
    tensors = {
        "uam": torch.tensor([1.0, 2.0]),
        "retain": torch.tensor([3.0, 4.0]),
    }
    tensors[field] = torch.tensor([non_finite, 1.0])
    projection = ResidualGUProjection(
        projection_coefficient=torch.tensor(0.5),
        residual_lambda=0.5,
    )
    decision = ResidualGUDecision(
        projection=projection,
        gate_dot=torch.tensor(-1.0 if keep else 1.0),
        relative_orthogonality=torch.tensor(0.0),
        sign_gate_passed=keep,
        orthogonality_safe=True,
    )

    with pytest.raises((ValueError, RuntimeError), match="finite"):
        apply_residual_gu_tensor(**tensors, decision=decision)


@pytest.mark.parametrize("keep", [True, False])
def test_apply_residual_gu_non_finite_derived_tensor_fails_closed(keep):
    projection = ResidualGUProjection(
        projection_coefficient=torch.tensor(0.0),
        residual_lambda=0.5,
    )
    decision = ResidualGUDecision(
        projection=projection,
        gate_dot=torch.tensor(-1.0 if keep else 1.0),
        relative_orthogonality=torch.tensor(0.0),
        sign_gate_passed=keep,
        orthogonality_safe=True,
    )

    with pytest.raises((ValueError, RuntimeError), match="finite"):
        apply_residual_gu_tensor(
            torch.tensor([3e38]),
            torch.tensor([-3e38]),
            decision,
        )
