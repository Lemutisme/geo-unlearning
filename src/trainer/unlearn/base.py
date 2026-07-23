import math
import os
import re
import stat
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
        diagnostics_parent_descriptor = None
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

                projection_eps = self.gu_config["projection_eps"]
                if (
                    isinstance(projection_eps, bool)
                    or not isinstance(projection_eps, Real)
                    or not math.isfinite(projection_eps)
                    or projection_eps <= 0
                ):
                    raise ValueError("GU projection_eps must be finite and positive")

                retain_filter = self.gu_config["retain_filter"]
                if not isinstance(retain_filter, str) or retain_filter not in {
                    "first_order",
                    "finite_step",
                }:
                    raise ValueError(
                        "GU retain_filter must be first_order or finite_step"
                    )

                retain_budget = self.gu_config["retain_budget"]
                if (
                    isinstance(retain_budget, bool)
                    or not isinstance(retain_budget, Real)
                    or not math.isfinite(retain_budget)
                    or retain_budget < 0
                ):
                    raise ValueError("GU retain_budget must be finite and nonnegative")

                backtracking_scales = self.gu_config["backtracking_scales"]
                if (
                    not isinstance(backtracking_scales, Sequence)
                    or isinstance(backtracking_scales, (str, bytes))
                    or not backtracking_scales
                ):
                    raise ValueError(
                        "GU backtracking_scales must be a nonempty sequence"
                    )
                previous_scale = math.inf
                for scale in backtracking_scales:
                    if (
                        isinstance(scale, bool)
                        or not isinstance(scale, Real)
                        or not math.isfinite(scale)
                        or not 0 < scale <= 1
                        or scale >= previous_scale
                    ):
                        raise ValueError(
                            "GU backtracking_scales must be finite, unique, and "
                            "strictly descending in (0, 1]"
                        )
                    previous_scale = scale

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
            optimizer_parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            optimizer_parameter_ids = [
                id(parameter) for parameter in optimizer_parameters
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

            if setup_required:
                output_dir = Path(os.path.abspath(self.args.output_dir))
                output_parts = output_dir.parts[1:]
                parent_parts = relative_diagnostics_path.parts[:-1]
                directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                directory_descriptor = os.open(
                    output_dir.anchor,
                    directory_flags,
                )
                try:
                    for index, part in enumerate(output_parts + parent_parts):
                        try:
                            try:
                                next_descriptor = os.open(
                                    part,
                                    directory_flags,
                                    dir_fd=directory_descriptor,
                                )
                            except FileNotFoundError:
                                try:
                                    os.mkdir(part, dir_fd=directory_descriptor)
                                except FileExistsError:
                                    pass
                                next_descriptor = os.open(
                                    part,
                                    directory_flags,
                                    dir_fd=directory_descriptor,
                                )
                        except OSError as error:
                            try:
                                component_stat = os.stat(
                                    part,
                                    dir_fd=directory_descriptor,
                                    follow_symlinks=False,
                                )
                            except OSError:
                                component_stat = None
                            if component_stat is not None and stat.S_ISLNK(
                                component_stat.st_mode
                            ):
                                raise ValueError(
                                    "GU diagnostics_path must not use symlinks"
                                ) from error
                            if index < len(output_parts):
                                raise ValueError(
                                    "GU output_dir must be a directory"
                                ) from error
                            raise ValueError(
                                "GU diagnostics_path parent must be a directory"
                            ) from error
                        previous_descriptor = directory_descriptor
                        directory_descriptor = next_descriptor
                        os.close(previous_descriptor)

                    diagnostics_parent_descriptor = directory_descriptor
                    directory_descriptor = None
                finally:
                    if directory_descriptor is not None:
                        os.close(directory_descriptor)

                diagnostics_name = relative_diagnostics_path.parts[-1]
                resolved_candidate = output_dir / relative_diagnostics_path
                diagnostics_flags = (
                    os.O_WRONLY
                    | os.O_APPEND
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                )
                try:
                    file_descriptor = os.open(
                        diagnostics_name,
                        diagnostics_flags,
                        0o666,
                        dir_fd=diagnostics_parent_descriptor,
                    )
                except OSError as error:
                    try:
                        candidate_stat = os.stat(
                            diagnostics_name,
                            dir_fd=diagnostics_parent_descriptor,
                            follow_symlinks=False,
                        )
                    except OSError:
                        candidate_stat = None
                    if candidate_stat is not None and stat.S_ISLNK(
                        candidate_stat.st_mode
                    ):
                        raise ValueError(
                            "GU diagnostics_path must not use symlinks"
                        ) from error
                    raise ValueError(
                        "GU diagnostics_path is not append-writable"
                    ) from error
                try:
                    os.close(file_descriptor)
                except OSError:
                    pass
                self._gu_parameter_patterns = compiled_patterns
                self._gu_selected = selected
                self._gu_diagnostics_path = resolved_candidate
                self._gu_setup_complete = True
        except Exception:
            for parameter, requires_grad in original_requires_grad:
                parameter.requires_grad_(requires_grad)
            for attribute in (
                "_gu_parameter_patterns",
                "_gu_selected",
                "_gu_diagnostics_path",
                "_gu_setup_complete",
            ):
                if hasattr(self, attribute):
                    delattr(self, attribute)
            raise
        finally:
            if diagnostics_parent_descriptor is not None:
                try:
                    os.close(diagnostics_parent_descriptor)
                except OSError:
                    pass
        return optimizer

    def training_step(self, model, inputs):
        if not self.gu_enabled:
            return super().training_step(model, inputs)

        inputs = self._prepare_inputs(inputs)
        retain_inputs = inputs.get("retain")
        if not isinstance(retain_inputs, Mapping):
            self._gu_constraint_microsteps = []
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
        scale = self.args.gradient_accumulation_steps
        blocks = []
        for (name, parameter), gradient in zip(
            self._gu_selected,
            retain_gradients,
        ):
            if gradient is None:
                self._gu_constraint_microsteps = []
                raise ValueError(
                    f"GU retain gradient is unused for selected parameter {name}"
                )
            if gradient.shape != parameter.shape:
                self._gu_constraint_microsteps = []
                raise ValueError(
                    f"GU retain gradient shape mismatch for selected parameter {name}"
                )
            block = gradient.detach().to(torch.float32) / scale
            if not torch.isfinite(block).all():
                self._gu_constraint_microsteps = []
                raise ValueError(
                    f"GU retain gradient must be finite for selected parameter {name}"
                )
            blocks.append(block)
        blocks = tuple(blocks)
        microsteps = getattr(self, "_gu_constraint_microsteps", [])
        microsteps.append(blocks)
        self._gu_constraint_microsteps = microsteps

        try:
            loss = super().training_step(model, inputs)
        except Exception:
            self._gu_constraint_microsteps = []
            raise

        if self.accelerator.sync_gradients:
            try:
                current_blocks = tuple(
                    torch.stack(
                        [microstep[index] for microstep in microsteps]
                    ).sum(dim=0)
                    for index in range(len(selected_parameters))
                )
                squared_norm = sum(
                    block.double().square().sum() for block in current_blocks
                )
                if not torch.isfinite(squared_norm):
                    raise ValueError("GU retain constraint norm must be finite")
                if squared_norm.item() == 0.0:
                    raise ValueError("GU retain constraint norm must be nonzero")
                norm = squared_norm.sqrt()
                current = tuple(
                    (block / norm).to(torch.float32)
                    for block in current_blocks
                )
                history = getattr(self, "_gu_constraint_history", ())
                self._gu_constraints_used = (current, *history)
                retain_history_rank = self.gu_config["retain_history_rank"]
                self._gu_constraint_history = (current, *history)[
                    :retain_history_rank
                ]
            finally:
                self._gu_constraint_microsteps = []

        return loss

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
