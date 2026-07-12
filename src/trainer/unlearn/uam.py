import math

from trainer.unlearn.component_buffers import ComponentGradientBuffers
from trainer.unlearn.geometric import GeometricUnlearn
from trainer.unlearn.optimizer_geometry import make_optimizer_geometry_adapter


class UAMUnlearn(GeometricUnlearn):
    def __init__(self, *args, **kwargs):
        self.uam_config = kwargs.pop("uam_config")
        super().__init__(*args, **kwargs)

        self.uam_mode = str(self.uam_config.mode).lower()
        self.forget_signal = str(self.uam_config.forget_signal).lower()
        normalization = str(self.uam_config.perturbation_normalization).lower()
        if normalization == "auto":
            normalization = "fixed_loss" if self.uam_mode == "uam" else "metric_trust"
        self.perturbation_normalization = normalization
        self.rho = float(self.uam_config.rho)
        self.reflection_gamma = float(self.uam_config.reflection_gamma)
        self.residual_lambda = float(self.uam_config.residual_lambda)
        self.sign_tau = float(self.uam_config.sign_tau)
        self.replay_device = str(self.uam_config.replay_device).lower()

        self.uam_calls = 0
        self.replay_calls = 0
        self.last_uam_diagnostics = {}
        self._uam_runtime_validated = False
        self._uam_microsteps = 0
        self._retain_replay_batches = []

    def _validate_uam_runtime(self):
        if self._uam_runtime_validated:
            return

        if self.uam_mode not in {"uam", "uam_gu"}:
            raise ValueError(f"Unsupported UAM mode: {self.uam_mode}")
        if self.forget_signal not in {"nll", "simnpo"}:
            raise ValueError(f"Unsupported UAM forget signal: {self.forget_signal}")
        if self.perturbation_normalization not in {"fixed_loss", "metric_trust"}:
            raise ValueError(
                "Unsupported UAM perturbation normalization: "
                f"{self.perturbation_normalization}"
            )
        if self.replay_device not in {"cpu", "parameter"}:
            raise ValueError(f"Unsupported UAM replay device: {self.replay_device}")
        if self.rho <= 0.0 or not math.isfinite(self.rho):
            raise ValueError("UAM rho must be positive and finite.")
        if not math.isfinite(self.reflection_gamma):
            raise ValueError("UAM reflection_gamma must be finite.")
        if self.residual_lambda < 0.0 or not math.isfinite(self.residual_lambda):
            raise ValueError("UAM residual_lambda must be non-negative and finite.")
        if self.sign_tau < 0.0 or not math.isfinite(self.sign_tau):
            raise ValueError("UAM sign_tau must be non-negative and finite.")
        if str(self.retain_loss_type).upper() != "NLL":
            raise NotImplementedError("UAM v1 requires NLL retain loss.")

        if self.is_deepspeed_enabled:
            raise NotImplementedError(
                "Approximate Adam UAM does not support DeepSpeed."
            )
        if self.is_fsdp_enabled:
            raise NotImplementedError("Approximate Adam UAM does not support FSDP.")
        if self.args.fp16:
            raise NotImplementedError("Approximate Adam UAM supports BF16/FP32 only.")
        if self.use_apex:
            raise NotImplementedError("Approximate Adam UAM does not support Apex.")
        if self.args.world_size != 1 or self.args.n_gpu > 1:
            raise NotImplementedError(
                "Approximate Adam UAM supports one process and one GPU only."
            )
        if self.args.gradient_checkpointing:
            checkpointing_kwargs = self.args.gradient_checkpointing_kwargs or {}
            if checkpointing_kwargs.get("use_reentrant", True):
                raise NotImplementedError(
                    "Approximate Adam UAM requires use_reentrant=false."
                )

        named_params = self._selected_named_parameters()
        if not named_params:
            raise ValueError("UAM requires at least one selected trainable parameter.")

        adapter = make_optimizer_geometry_adapter(self.optimizer)
        adapter.validate(named_params)

        if self.component_buffer_device == "cpu":
            ComponentGradientBuffers.validate_host_memory(
                sum(parameter.numel() for _, parameter in named_params),
                component_count=3,
            )

        self._optimizer_geometry_adapter = adapter
        self._uam_runtime_validated = True
