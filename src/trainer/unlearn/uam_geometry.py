from dataclasses import dataclass

import torch


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
    if raw_forget_sq.item() == 0.0:
        raise RuntimeError("UAM forget gradient has zero norm")

    if mode == "fixed_loss":
        coefficient = rho / raw_forget_sq.clamp_min(eps)
    else:
        coefficient = rho / optimizer_forget_sq.clamp_min(eps).sqrt()
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
    coefficient = (
        reflection_gamma
        * forget_perturbed_retain_dot
        / optimizer_forget_sq.clamp_min(eps)
    )
    return UAMDecision(coefficient=coefficient)


def apply_uam_tensor(
    perturbed_retain: torch.Tensor,
    forget: torch.Tensor,
    decision: UAMDecision,
) -> torch.Tensor:
    return perturbed_retain.float() - decision.coefficient.float() * forget.float()


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
    if optimizer_retain_sq.item() == 0.0:
        raise RuntimeError("Residual-GU-UAM retain gradient has zero norm")

    projection_coefficient = residual_retain_dot / optimizer_retain_sq.clamp_min(eps)
    gate_dot = residual_forget_dot - projection_coefficient * forget_retain_dot
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
    retain_fp32 = retain.float()
    residual = uam.float() - retain_fp32
    normal = residual - decision.projection_coefficient.float() * retain_fp32
    correction = decision.residual_lambda * normal if decision.keep else 0.0
    final = retain_fp32 + correction
    return final, normal
