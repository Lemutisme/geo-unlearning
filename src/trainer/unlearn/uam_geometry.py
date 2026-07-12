import math
from dataclasses import dataclass

import torch


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be a finite tensor")


def _require_finite_tensor_scalar(name: str, value: torch.Tensor) -> None:
    if value.numel() != 1:
        raise ValueError(f"{name} must be a finite tensor scalar")
    _require_finite_tensor(name, value)


def _require_finite_float(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class PerturbationDecision:
    mode: str
    coefficient: torch.Tensor


def decide_perturbation(
    mode: str,
    raw_forget_sq: torch.Tensor,
    optimizer_forget_sq: torch.Tensor,
    rho: float,
    eps: float,
) -> PerturbationDecision:
    if mode not in {"fixed_loss", "metric_trust"}:
        raise ValueError(f"Unsupported UAM perturbation normalization: {mode}")

    _require_finite_tensor_scalar("raw_forget_sq", raw_forget_sq)
    _require_finite_tensor_scalar("optimizer_forget_sq", optimizer_forget_sq)
    _require_finite_float("rho", rho)
    _require_finite_float("eps", eps)
    if raw_forget_sq.item() == 0.0:
        raise RuntimeError("UAM forget gradient has zero norm")

    if mode == "fixed_loss":
        coefficient = rho / raw_forget_sq.clamp_min(eps)
    else:
        coefficient = rho / optimizer_forget_sq.clamp_min(eps).sqrt()
    _require_finite_tensor_scalar("perturbation coefficient", coefficient)
    return PerturbationDecision(mode=mode, coefficient=coefficient)


@dataclass(frozen=True)
class UAMDecision:
    coefficient: torch.Tensor


def decide_uam(
    forget_perturbed_retain_dot: torch.Tensor,
    optimizer_forget_sq: torch.Tensor,
    reflection_gamma: float,
    eps: float,
) -> UAMDecision:
    _require_finite_tensor_scalar(
        "forget_perturbed_retain_dot",
        forget_perturbed_retain_dot,
    )
    _require_finite_tensor_scalar("optimizer_forget_sq", optimizer_forget_sq)
    _require_finite_float("reflection_gamma", reflection_gamma)
    _require_finite_float("eps", eps)
    coefficient = (
        reflection_gamma * forget_perturbed_retain_dot / (optimizer_forget_sq + eps)
    )
    _require_finite_tensor_scalar("UAM coefficient", coefficient)
    return UAMDecision(coefficient=coefficient)


def apply_uam_tensor(
    perturbed_retain: torch.Tensor,
    forget: torch.Tensor,
    decision: UAMDecision,
) -> torch.Tensor:
    _require_finite_tensor("perturbed_retain", perturbed_retain)
    _require_finite_tensor("forget", forget)
    applied = perturbed_retain.float() - decision.coefficient.float() * forget.float()
    _require_finite_tensor("applied UAM tensor", applied)
    return applied


@dataclass(frozen=True)
class ResidualGUDecision:
    projection_coefficient: torch.Tensor
    residual_lambda: float
    gate_dot: torch.Tensor
    keep: bool


def decide_residual_gu(
    residual_retain_dot: torch.Tensor,
    residual_forget_dot: torch.Tensor,
    forget_retain_dot: torch.Tensor,
    optimizer_retain_sq: torch.Tensor,
    residual_lambda: float,
    sign_tau: float,
    eps: float,
) -> ResidualGUDecision:
    _require_finite_tensor_scalar("residual_retain_dot", residual_retain_dot)
    _require_finite_tensor_scalar("residual_forget_dot", residual_forget_dot)
    _require_finite_tensor_scalar("forget_retain_dot", forget_retain_dot)
    _require_finite_tensor_scalar("optimizer_retain_sq", optimizer_retain_sq)
    _require_finite_float("residual_lambda", residual_lambda)
    _require_finite_float("sign_tau", sign_tau)
    _require_finite_float("eps", eps)
    if optimizer_retain_sq.item() == 0.0:
        raise RuntimeError("Residual-GU-UAM retain gradient has zero norm")

    projection_coefficient = residual_retain_dot / (optimizer_retain_sq + eps)
    _require_finite_tensor_scalar(
        "Residual-GU-UAM projection coefficient",
        projection_coefficient,
    )
    gate_dot = residual_forget_dot - projection_coefficient * forget_retain_dot
    _require_finite_tensor_scalar("Residual-GU-UAM gate dot", gate_dot)
    keep = bool(gate_dot.item() < -sign_tau)
    return ResidualGUDecision(
        projection_coefficient=projection_coefficient,
        residual_lambda=residual_lambda,
        gate_dot=gate_dot,
        keep=keep,
    )


def apply_residual_gu_tensor(
    uam: torch.Tensor,
    retain: torch.Tensor,
    decision: ResidualGUDecision,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_finite_tensor("uam", uam)
    _require_finite_tensor("retain", retain)
    retain_fp32 = retain.float()
    residual = uam.float() - retain_fp32
    _require_finite_tensor("Residual-GU-UAM residual", residual)
    normal = residual - decision.projection_coefficient.float() * retain_fp32
    _require_finite_tensor("Residual-GU-UAM normal", normal)
    correction = (
        decision.residual_lambda * normal if decision.keep else torch.zeros_like(normal)
    )
    _require_finite_tensor("Residual-GU-UAM correction", correction)
    final = retain_fp32 + correction
    _require_finite_tensor("final Residual-GU-UAM tensor", final)
    return final, normal
