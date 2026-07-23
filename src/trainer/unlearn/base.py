import math
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate.utils import is_deepspeed_available
from packaging import version
from torch import nn
from trainer.base import FinetuneTrainer
from trainer.utils import compute_batch_nll
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_only,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_deepspeed_available():
    import deepspeed


class UnlearnTrainer(FinetuneTrainer):
    def __init__(self, *args, gu=None, **kwargs):
        self.gu_config = gu
        self.gu_enabled = gu is not None and gu.get("enabled") is True
        self.gu_projection_calls = 0
        self.gu_last_diagnostics = None
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.gu_config is None:
            return super().create_optimizer()

        original_requires_grad = tuple(
            (parameter, parameter.requires_grad)
            for parameter in self.model.parameters()
        )
        setup_required = not getattr(self, "_gu_setup_complete", False)
        try:
            required_keys = {
                "enabled",
                "parameter_regex",
                "retain_history_rank",
                "projection_eps",
                "retain_filter",
                "retain_budget",
                "backtracking_scales",
                "diagnostics_path",
            }
            if set(self.gu_config) != required_keys:
                raise ValueError(
                    "GU configuration keys do not match the public contract"
                )
            if not isinstance(self.gu_config["enabled"], bool):
                raise ValueError("GU enabled must be bool")
            if not self.gu_config["enabled"]:
                return super().create_optimizer()

            if setup_required:
                parameter_regex = self.gu_config["parameter_regex"]
                if (
                    not isinstance(parameter_regex, Sequence)
                    or isinstance(parameter_regex, (str, bytes))
                    or not parameter_regex
                    or any(
                        not isinstance(pattern, str) or not pattern
                        for pattern in parameter_regex
                    )
                ):
                    raise ValueError("GU parameter_regex must contain nonempty strings")
                try:
                    compiled_patterns = tuple(
                        re.compile(pattern) for pattern in parameter_regex
                    )
                except re.error as error:
                    raise ValueError(
                        "GU parameter_regex contains an invalid regex"
                    ) from error

                retain_history_rank = self.gu_config["retain_history_rank"]
                if (
                    isinstance(retain_history_rank, bool)
                    or not isinstance(retain_history_rank, int)
                    or not 0 <= retain_history_rank <= 8
                ):
                    raise ValueError(
                        "GU retain_history_rank must be an integer from 0 to 8"
                    )

                retain_filter = self.gu_config["retain_filter"]
                if not isinstance(retain_filter, str) or retain_filter not in {
                    "first_order",
                    "finite_step",
                }:
                    raise ValueError(
                        "GU retain_filter must be first_order or finite_step"
                    )

                for key, must_be_positive in (
                    ("projection_eps", True),
                    ("retain_budget", False),
                ):
                    value = self.gu_config[key]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, Real)
                        or not math.isfinite(value)
                        or value < 0
                        or (must_be_positive and value == 0)
                    ):
                        bound = "positive" if must_be_positive else "nonnegative"
                        raise ValueError(f"GU {key} must be finite and {bound}")

                backtracking_scales = self.gu_config["backtracking_scales"]
                valid_scales = (
                    isinstance(backtracking_scales, Sequence)
                    and not isinstance(backtracking_scales, (str, bytes))
                    and bool(backtracking_scales)
                )
                if valid_scales:
                    valid_scales = all(
                        isinstance(scale, Real)
                        and not isinstance(scale, bool)
                        and math.isfinite(scale)
                        and 0 < scale <= 1
                        for scale in backtracking_scales
                    )
                if valid_scales:
                    valid_scales = all(
                        left > right
                        for left, right in zip(
                            backtracking_scales,
                            backtracking_scales[1:],
                        )
                    )
                if not valid_scales:
                    raise ValueError(
                        "GU backtracking_scales must be finite, unique, and "
                        "strictly descending in (0, 1]"
                    )

                diagnostics_path = self.gu_config["diagnostics_path"]
                if (
                    not isinstance(diagnostics_path, str)
                    or not diagnostics_path.strip()
                ):
                    raise ValueError(
                        "GU diagnostics_path must be a nonempty relative path"
                    )
                relative_diagnostics_path = Path(diagnostics_path)
                if (
                    relative_diagnostics_path.is_absolute()
                    or relative_diagnostics_path == Path(".")
                    or ".." in relative_diagnostics_path.parts
                ):
                    raise ValueError("GU diagnostics_path must be a safe relative path")

                if getattr(self, "is_deepspeed_enabled", False):
                    raise ValueError("GU does not support DeepSpeed")
                if getattr(self, "is_fsdp_enabled", False) or getattr(
                    self, "is_fsdp_xla_enabled", False
                ):
                    raise ValueError("GU does not support FSDP")
                if getattr(self, "use_apex", False):
                    raise ValueError("GU does not support Apex")
                if self.args.fp16:
                    raise ValueError("GU does not support FP16")
                if self.args.world_size > 1 or self.args.n_gpu > 1:
                    raise ValueError("GU requires a single process and at most one GPU")
                checkpointing_kwargs = self.args.gradient_checkpointing_kwargs or {}
                if self.args.gradient_checkpointing and checkpointing_kwargs.get(
                    "use_reentrant", True
                ):
                    raise ValueError(
                        "GU does not support reentrant gradient checkpointing"
                    )

                selected = tuple(
                    (name, parameter)
                    for name, parameter in self.model.named_parameters()
                    if any(pattern.search(name) for pattern in compiled_patterns)
                )
                if not selected:
                    raise ValueError("GU parameter_regex did not select any parameters")
                selected_ids = {id(parameter) for _, parameter in selected}
                for parameter in self.model.parameters():
                    if id(parameter) not in selected_ids:
                        parameter.requires_grad_(False)
            else:
                selected = self._gu_selected

            optimizer = super().create_optimizer()
            optimizer_parameter_ids = [
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            selected_parameter_ids = {id(parameter) for _, parameter in selected}
            if (
                len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids))
                or set(optimizer_parameter_ids) != selected_parameter_ids
            ):
                raise ValueError(
                    "GU optimizer parameters must match selected parameters exactly "
                    "without duplicates"
                )

            supported_optimizer = isinstance(optimizer, torch.optim.AdamW)
            if not supported_optimizer:
                try:
                    import bitsandbytes
                except ImportError:
                    bitsandbytes = None
                if bitsandbytes is not None:
                    paged_adamw_types = (
                        bitsandbytes.optim.AdamW,
                        bitsandbytes.optim.PagedAdamW,
                        bitsandbytes.optim.PagedAdamW32bit,
                    )
                    supported_optimizer = (
                        isinstance(optimizer, paged_adamw_types)
                        and optimizer.is_paged
                        and optimizer.args.optim_bits == 32
                        and optimizer.optimizer_name == "adam"
                    )
            if not supported_optimizer:
                raise ValueError(
                    "GU requires torch.optim.AdamW or shipped bitsandbytes 32-bit "
                    "paged AdamW"
                )

            if getattr(self, "_gu_hooked_optimizer", None) is not optimizer:
                pre_hook_handle = optimizer.register_step_pre_hook(
                    self._gu_optimizer_step_pre_hook
                )
                try:
                    optimizer.register_step_post_hook(
                        self._gu_optimizer_step_post_hook
                    )
                except Exception:
                    pre_hook_handle.remove()
                    raise
                self._gu_hooked_optimizer = optimizer

            if setup_required:
                resolved_candidate = Path(
                    os.path.abspath(self.args.output_dir)
                ) / relative_diagnostics_path
                if any(
                    component.is_symlink()
                    for component in (
                        resolved_candidate,
                        *resolved_candidate.parents,
                    )
                ):
                    raise ValueError("GU diagnostics_path must not use symlinks")
                try:
                    resolved_candidate.parent.mkdir(parents=True, exist_ok=True)
                    file_descriptor = os.open(
                        resolved_candidate,
                        os.O_WRONLY
                        | os.O_APPEND
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_NOFOLLOW,
                        0o666,
                    )
                    os.close(file_descriptor)
                except OSError as error:
                    raise ValueError(
                        "GU diagnostics_path is not append-writable"
                    ) from error
                self._gu_parameter_patterns = compiled_patterns
                self._gu_selected = selected
                self._gu_diagnostics_path = resolved_candidate
                self._gu_setup_complete = True
        except Exception:
            for parameter, requires_grad in original_requires_grad:
                parameter.requires_grad_(requires_grad)
            raise
        return optimizer

    def _gu_optimizer_step_pre_hook(self, optimizer, _args, _kwargs):
        del optimizer
        if hasattr(self, "_gu_parameter_snapshot"):
            raise ValueError("GU optimizer step already has a pending snapshot")
        if not hasattr(self, "_gu_constraints_used") or not hasattr(
            self, "_gu_pending_history_covector"
        ):
            raise ValueError("GU optimizer step requires ready constraints")
        constraints = self._gu_constraints_used
        pending = self._gu_pending_history_covector
        if (
            not isinstance(constraints, tuple)
            or not constraints
            or constraints[0] is not pending
        ):
            raise ValueError("GU optimizer step received stale constraints")
        if (
            not isinstance(pending, tuple)
            or len(pending) != len(self._gu_selected)
            or any(
                not isinstance(block, torch.Tensor)
                or block.shape != parameter.shape
                for block, (_, parameter) in zip(pending, self._gu_selected)
            )
        ):
            raise ValueError("GU optimizer step received stale pending covector")

        self._gu_parameter_snapshot = tuple(
            parameter.detach().clone()
            for _, parameter in self._gu_selected
        )
        if hasattr(self, "_gu_proposal_delta"):
            del self._gu_proposal_delta

    def _gu_optimizer_step_post_hook(self, optimizer, _args, _kwargs):
        if not hasattr(self, "_gu_parameter_snapshot"):
            raise ValueError("GU optimizer step is missing its parameter snapshot")

        snapshot = self._gu_parameter_snapshot
        pending = self._gu_pending_history_covector
        constraints = self._gu_constraints_used
        try:
            if len(snapshot) != len(self._gu_selected) or not 1 <= len(
                constraints) <= 9 or any(not isinstance(item, tuple)
                or len(item) != len(self._gu_selected) for item in constraints):
                raise ValueError("GU constraints must align with selected parameters")
            proposal = tuple(parameter.detach().float() - before.float()
                for before, (_, parameter) in zip(snapshot, self._gu_selected))
            count, tolerance = len(constraints), float(self.gu_config["projection_eps"])
            violations_before = torch.zeros(count, dtype=torch.float64)
            gram = torch.zeros((count, count), dtype=torch.float64)
            proposal_squared = 0.0
            for block_index, (before, (name, parameter), delta) in enumerate(
                zip(snapshot, self._gu_selected, proposal)
            ):
                if before.shape != parameter.shape:
                    raise ValueError(f"GU parameter snapshot shape mismatch for {name}")
                if not torch.isfinite(delta).all():
                    raise ValueError(f"GU proposal must be finite for {name}")

                state = optimizer.state.get(parameter)
                if not isinstance(state, Mapping) or not (
                    "exp_avg_sq" in state or "state2" in state
                ):
                    raise ValueError(f"GU optimizer state is missing for {name}")
                state_key = "exp_avg_sq" if "exp_avg_sq" in state else "state2"
                second_moment = state[state_key]
                if not isinstance(second_moment, torch.Tensor) or not (
                    torch.is_floating_point(second_moment)
                ):
                    raise ValueError(f"GU second moment must be floating for {name}")
                if second_moment.shape != parameter.shape:
                    raise ValueError(f"GU second-moment shape mismatch for {name}")
                if not torch.isfinite(second_moment).all():
                    raise ValueError(f"GU second moment must be finite for {name}")
                if (second_moment < 0).any():
                    raise ValueError(f"GU second moment must be nonnegative for {name}")

                epsilon = next(
                    group["eps"]
                    for group in optimizer.param_groups
                    if any(
                        candidate is parameter for candidate in group["params"]
                    )
                )
                diagonal = second_moment.detach().double().sqrt()
                diagonal.add_(epsilon)
                if not torch.isfinite(diagonal).all() or (diagonal <= 0).any():
                    raise ValueError(f"GU metric diagonal must be finite for {name}")
                blocks = tuple(item[block_index] for item in constraints)
                if any(not isinstance(block, torch.Tensor) or block.shape != parameter.shape
                    or not torch.isfinite(block).all() for block in blocks):
                    raise ValueError(f"GU constraint must be finite for {name}")
                delta64 = delta.double()
                proposal_squared += delta64.square().sum().item()
                for left, left_block in enumerate(blocks):
                    left64 = left_block.double()
                    violations_before[left] += (left64 * delta64).sum().item()
                    for right, right_block in enumerate(blocks):
                        value = (left64 * right_block.double() / diagonal.double()).sum().item()
                        gram[left, right] += value
            if not torch.isfinite(violations_before).all() or not torch.isfinite(gram).all():
                raise ValueError("GU proposal violations and Gram matrix must be finite")
            multipliers, active_constraints = torch.zeros(count, dtype=torch.float64), []
            violations_after = violations_before.clone()
            if (violations_before > tolerance).any():
                best = None
                for mask in range(1, 1 << count):
                    active = [index for index in range(count) if mask & 1 << index]
                    result = torch.linalg.lstsq(gram[active][:, active],
                        violations_before[active], rcond=tolerance)
                    dual = result.solution
                    if result.rank.item() != len(active) or not torch.isfinite(
                        dual
                    ).all() or (dual < 0).any():
                        continue
                    candidate = torch.zeros_like(multipliers)
                    candidate[active] = dual
                    candidate_violations = violations_before - gram @ candidate
                    if not torch.isfinite(candidate_violations).all() or (
                        candidate_violations > tolerance
                    ).any() or (candidate_violations[active].abs() > tolerance).any():
                        continue
                    cost = (candidate @ gram @ candidate).item()
                    if math.isfinite(cost) and cost >= 0 and (best is None or cost < best[0]):
                        best = (cost, candidate, candidate_violations, active)
                if best is None:
                    raise ValueError("GU projection has no feasible active set")
                _, multipliers, violations_after, active_constraints = best
            kkt_residual = max(max(violations_after.max().item(), 0.0), max(
                (-multipliers).max().item(), 0.0), (multipliers *
                violations_after).abs().max().item(), max((abs(
                    violations_after[i].item()) for i in active_constraints), default=0.0))
            if not math.isfinite(kkt_residual) or kkt_residual > tolerance:
                raise ValueError("GU projection failed its KKT residual check")
            corrected_squared = correction_squared = 0.0
            violations_after.zero_()
            with torch.no_grad():
                for block_index, (delta, (_, parameter)) in enumerate(
                    zip(proposal, self._gu_selected)):
                    state = optimizer.state[parameter]
                    key = "exp_avg_sq" if "exp_avg_sq" in state else "state2"
                    metric = state[key].detach().double().sqrt()
                    metric.add_(next(group["eps"] for group in optimizer.param_groups
                        if any(item is parameter for item in group["params"])))
                    weighted = sum((item[block_index].double() * value.item()
                        for value, item in zip(multipliers, constraints)),
                        start=torch.zeros_like(delta, dtype=torch.float64))
                    corrected = delta.double() - weighted / metric
                    if not torch.isfinite(corrected).all():
                        raise ValueError("GU corrected delta must be finite")
                    parameter.add_((corrected - delta.double()).to(parameter.dtype))
                    applied = parameter.detach().float() - snapshot[block_index].float()
                    corrected_squared += applied.double().square().sum().item()
                    correction_squared += (applied.double() - delta.double()).square().sum().item()
                    for index, constraint in enumerate(constraints):
                        violations_after[index] += (constraint[block_index].double()
                            * applied.double()).sum().item()
            kkt_residual = max(max(violations_after.max().item(), 0.0), max(
                (-multipliers).max().item(), 0.0), (multipliers *
                violations_after).abs().max().item(), max((abs(
                    violations_after[i].item()) for i in active_constraints), default=0.0))
            if not torch.isfinite(violations_after).all() or not math.isfinite(
                kkt_residual) or kkt_residual > tolerance:
                with torch.no_grad():
                    for before, (_, parameter) in zip(snapshot, self._gu_selected):
                        parameter.copy_(before)
                raise ValueError("GU applied delta failed its KKT residual check")
            self.gu_last_diagnostics = {
                "proposal_norm": math.sqrt(proposal_squared),
                "corrected_norm": math.sqrt(corrected_squared),
                "correction_ratio": math.sqrt(correction_squared / proposal_squared)
                if proposal_squared else 0.0,
                "constraint_count": count,
                "active_constraints": active_constraints,
                "max_violation_before": violations_before.max().item(),
                "max_violation_after": violations_after.max().item(),
                "kkt_residual": kkt_residual,
                "projection_tolerance": tolerance,
            }
            self.gu_projection_calls += 1
            retain_history_rank = self.gu_config["retain_history_rank"]
            self._gu_constraint_history = (
                (pending, *getattr(self, "_gu_constraint_history", ()))[:retain_history_rank]
                if retain_history_rank
                else ()
            )
        finally:
            for attribute in (
                "_gu_constraint_accumulator",
                "_gu_constraints_used",
                "_gu_pending_history_covector",
                "_gu_parameter_snapshot",
                "_gu_proposal_delta",
            ):
                if hasattr(self, attribute):
                    delattr(self, attribute)

    def training_step(self, model, inputs):
        if not self.gu_enabled:
            return super().training_step(model, inputs)

        try:
            model.train()
            inputs = self._prepare_inputs(inputs)
            retain_inputs = inputs.get("retain")
            if not isinstance(retain_inputs, Mapping):
                raise ValueError("GU retain inputs must be a mapping")
            with self.compute_loss_context_manager():
                retain_nll, _ = compute_batch_nll(model, retain_inputs)
                retain_loss = retain_nll.mean()
            selected_parameters = tuple(
                parameter for _, parameter in self._gu_selected
            )
            retain_gradients = torch.autograd.grad(
                retain_loss,
                selected_parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            accumulator = getattr(self, "_gu_constraint_accumulator", None)
            if accumulator is None:
                accumulator = tuple(
                    torch.zeros_like(parameter, dtype=torch.float32)
                    for parameter in selected_parameters
                )
                self._gu_constraint_accumulator = accumulator
            scale = self.args.gradient_accumulation_steps
            with torch.no_grad():
                for (name, _), gradient, accumulator_block in zip(
                    self._gu_selected,
                    retain_gradients,
                    accumulator,
                ):
                    if gradient is None:
                        raise ValueError(
                            "GU retain gradient is unused for selected parameter "
                            f"{name}"
                        )
                    gradient_block = gradient.detach().to(torch.float32)
                    if not torch.isfinite(gradient_block).all():
                        raise ValueError(
                            "GU retain gradient must be finite for selected "
                            f"parameter {name}"
                        )
                    accumulator_block.add_(gradient_block, alpha=1.0 / scale)
            del retain_gradients, gradient, gradient_block

            loss = super().training_step(model, inputs)

            if self.accelerator.sync_gradients:
                squared_norm = sum(
                    block.double().square().sum() for block in accumulator
                )
                if not torch.isfinite(squared_norm):
                    raise ValueError("GU retain constraint norm must be finite")
                if squared_norm.item() == 0.0:
                    raise ValueError("GU retain constraint norm must be nonzero")
                norm = squared_norm.sqrt()
                current = tuple(
                    (block / norm).to(torch.float32)
                    for block in accumulator
                )
                history = getattr(self, "_gu_constraint_history", ())
                self._gu_constraints_used = (current, *history)
                self._gu_pending_history_covector = current
                del self._gu_constraint_accumulator

            return loss
        except Exception:
            for attribute in (
                "_gu_constraint_accumulator",
                "_gu_constraints_used",
                "_gu_pending_history_covector",
            ):
                if hasattr(self, attribute):
                    delattr(self, attribute)
            raise

    # Adapted from Huggingface DPO Trainer: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        The only change to this function is calling the Trainer's compute_loss, as it's often overridden by unlearning methods, and we want to maintain the Trainer's evaluation setup.
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        ### Call compute_loss of super class since overridden compute_loss is not be applicable to eval_dataset.
                        loss, outputs = super().compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
