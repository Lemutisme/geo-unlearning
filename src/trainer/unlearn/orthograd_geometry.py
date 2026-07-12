from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class OrthogradProjectionStats:
    requested_rank: int
    effective_rank: int
    dropped_directions: int
    candidate_norm: float
    projected_norm: float
    preserved_signal_ratio: float
    maximum_relative_residual: float
    basis_bytes: int
    orthogonalization_flops: int
    estimated_projection_flops: int


class GlobalRetainBasis:
    def __init__(self, rank_tolerance: float = 1e-8):
        rank_tolerance = float(rank_tolerance)
        if rank_tolerance <= 0.0 or not math.isfinite(rank_tolerance):
            raise ValueError("rank_tolerance must be positive and finite.")
        self.rank_tolerance = rank_tolerance
        self.effective_rank_tolerance = max(
            rank_tolerance,
            8.0 * torch.finfo(torch.float32).eps,
        )
        self.vectors: list[dict[str, torch.Tensor]] = []
        self.requested_rank = 0
        self.dropped_directions = 0
        self.orthogonalization_flops = 0

    @staticmethod
    def global_dot(
        left: dict[str, torch.Tensor],
        right: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        result = None
        for name in sorted(left.keys() & right.keys()):
            value = (left[name].float() * right[name].float()).sum()
            result = value if result is None else result + value
        if result is None:
            raise RuntimeError("No common tensors for global dot product.")
        return result

    @classmethod
    def global_norm(cls, vector: dict[str, torch.Tensor]) -> torch.Tensor:
        return cls.global_dot(vector, vector).clamp_min(0.0).sqrt()

    @property
    def empty(self) -> bool:
        return not self.vectors

    @property
    def effective_rank(self) -> int:
        return len(self.vectors)

    @property
    def basis_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for vector in self.vectors
            for tensor in vector.values()
        )

    @staticmethod
    def _common_numel(left, right) -> int:
        return sum(left[name].numel() for name in left.keys() & right.keys())

    @torch.no_grad()
    def add(self, vector: dict[str, torch.Tensor]) -> bool:
        if not vector:
            raise RuntimeError("Cannot add an empty retain gradient direction.")
        self.requested_rank += 1
        result = {
            name: tensor.detach().to(dtype=torch.float32).clone()
            for name, tensor in vector.items()
        }
        original_norm = self.global_norm(result)
        if not torch.isfinite(original_norm):
            raise RuntimeError("Retain gradient direction has a non-finite norm.")
        if original_norm.item() == 0.0:
            self.dropped_directions += 1
            return False

        for _ in range(2):
            for direction in self.vectors:
                coefficient = self.global_dot(result, direction)
                for name in result.keys() & direction.keys():
                    result[name].add_(
                        direction[name],
                        alpha=-float(coefficient.item()),
                    )
                self.orthogonalization_flops += 4 * self._common_numel(
                    result,
                    direction,
                )

        norm = self.global_norm(result)
        if not torch.isfinite(norm):
            raise RuntimeError(
                "Orthogonalized retain gradient direction has a non-finite norm."
            )
        if norm <= self.effective_rank_tolerance * original_norm:
            self.dropped_directions += 1
            return False

        inverse_norm = norm.reciprocal()
        for tensor in result.values():
            tensor.mul_(inverse_norm)
        self.vectors.append(result)
        return True

    @torch.no_grad()
    def project(self, candidate: dict[str, torch.Tensor]):
        if not candidate:
            raise RuntimeError("Cannot project an empty candidate gradient.")
        if self.empty:
            raise RuntimeError("Cannot project without a retain gradient basis.")

        projected = {
            name: tensor.detach().to(dtype=torch.float32).clone()
            for name, tensor in candidate.items()
        }
        removed = {name: torch.zeros_like(tensor) for name, tensor in projected.items()}
        projection_flops = 0
        for direction in self.vectors:
            coefficient = self.global_dot(candidate, direction)
            for name in projected.keys() & direction.keys():
                contribution = direction[name] * coefficient
                projected[name].sub_(contribution)
                removed[name].add_(contribution)
            projection_flops += 4 * self._common_numel(projected, direction)

        candidate_norm = self.global_norm(candidate)
        projected_norm = self.global_norm(projected)
        if not torch.isfinite(candidate_norm) or not torch.isfinite(projected_norm):
            raise RuntimeError("Orthograd projection produced a non-finite norm.")
        eps = torch.finfo(torch.float32).eps
        residuals = [
            self.global_dot(projected, direction).abs() / (projected_norm + eps)
            for direction in self.vectors
        ]
        maximum_residual = max(float(value.item()) for value in residuals)
        candidate_norm_value = float(candidate_norm.item())
        projected_norm_value = float(projected_norm.item())
        preserved_ratio = projected_norm_value / max(candidate_norm_value, eps)
        stats = OrthogradProjectionStats(
            requested_rank=self.requested_rank,
            effective_rank=self.effective_rank,
            dropped_directions=self.dropped_directions,
            candidate_norm=candidate_norm_value,
            projected_norm=projected_norm_value,
            preserved_signal_ratio=preserved_ratio,
            maximum_relative_residual=maximum_residual,
            basis_bytes=self.basis_bytes,
            orthogonalization_flops=self.orthogonalization_flops,
            estimated_projection_flops=projection_flops,
        )
        return projected, removed, stats

    def clear(self) -> None:
        self.vectors.clear()
        self.requested_rank = 0
        self.dropped_directions = 0
        self.orthogonalization_flops = 0
