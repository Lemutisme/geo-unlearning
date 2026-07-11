from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SurgeryDecision:
    mode: str
    coefficient: torch.Tensor
    raw_coefficient: torch.Tensor
    conflict: bool


def decide_global_surgery(
    mode: str,
    dot: torch.Tensor,
    retain_sq: torch.Tensor,
    eps: float,
) -> SurgeryDecision:
    if mode not in {"gu", "pcgrad"}:
        raise ValueError(f"Unsupported gradient surgery mode: {mode}")

    raw_coefficient = dot / retain_sq.clamp_min(eps)
    conflict = bool(dot.item() < 0)
    coefficient = (
        raw_coefficient
        if mode == "gu" or conflict
        else torch.zeros_like(raw_coefficient)
    )
    return SurgeryDecision(
        mode=mode,
        coefficient=coefficient,
        raw_coefficient=raw_coefficient,
        conflict=conflict,
    )


def apply_surgery_tensor(
    forget: torch.Tensor,
    retain: torch.Tensor,
    decision: SurgeryDecision,
) -> torch.Tensor:
    return forget.float() - decision.coefficient * retain.float()
