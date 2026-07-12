import math

from trainer.unlearn.component_buffers import ComponentGradientBuffers
from trainer.unlearn.geometric import GeometricUnlearn
from trainer.unlearn.optimizer_geometry import make_optimizer_geometry_adapter
from trainer.utils import compute_batch_nll


class UAMUnlearn(GeometricUnlearn):
    @staticmethod
    def _config_float(config, field):
        value = getattr(config, field)
        if isinstance(value, bool):
            raise ValueError(f"UAM {field} must be numeric, not bool.")
        return float(value)

    def __init__(self, *args, **kwargs):
        self.uam_config = kwargs.pop("uam_config")
        super().__init__(*args, **kwargs)

        self.uam_mode = str(self.uam_config.mode).lower()
        self.forget_signal = str(self.uam_config.forget_signal).lower()
        normalization = str(self.uam_config.perturbation_normalization).lower()
        if normalization == "auto":
            normalization = "fixed_loss" if self.uam_mode == "uam" else "metric_trust"
        self.perturbation_normalization = normalization
        self.rho = self._config_float(self.uam_config, "rho")
        self.reflection_gamma = self._config_float(
            self.uam_config,
            "reflection_gamma",
        )
        self.residual_lambda = self._config_float(
            self.uam_config,
            "residual_lambda",
        )
        self.sign_tau = self._config_float(self.uam_config, "sign_tau")
        self.replay_device = str(self.uam_config.replay_device).lower()

        self.uam_calls = 0
        self.replay_calls = 0
        self.last_uam_diagnostics = {}
        self._uam_runtime_validated = False
        self._uam_microsteps = 0
        self._retain_replay_batches = []

    def compute_uam_forget_signal(self, model, forget_inputs):
        if self.forget_signal == "nll":
            sequence_nll, outputs = compute_batch_nll(model, forget_inputs)
            answer_counts = forget_inputs["labels"][..., 1:].ne(-100).sum(-1)
            if (answer_counts == 0).any():
                raise RuntimeError("UAM NLL forget signal has an empty answer mask.")
            answer_counts = answer_counts.to(sequence_nll)
            return (sequence_nll / answer_counts).mean(), outputs
        if self.forget_signal == "simnpo":
            simnpo_loss, outputs = self.compute_forget_loss(model, forget_inputs)
            return -simnpo_loss, outputs
        raise ValueError(f"Unsupported UAM forget signal: {self.forget_signal}")

    def _validate_uam_runtime(self):
        if self._uam_runtime_validated:
            return

        if self.uam_mode not in {"uam", "uam_gu"}:
            raise ValueError(f"Unsupported UAM mode: {self.uam_mode}")
        if self.forget_signal not in {"nll", "simnpo"}:
            raise ValueError(f"Unsupported UAM forget signal: {self.forget_signal}")
        if self.forget_signal == "simnpo":
            if self.loss_name != "simnpo":
                raise ValueError(
                    "UAM SimNPO forget signal requires "
                    "geometric_config.loss='simnpo'."
                )
            if self.simnpo_config is None:
                raise ValueError(
                    "UAM SimNPO forget signal requires a non-null simnpo_config."
                )
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
            if checkpointing_kwargs.get("use_reentrant") is not False:
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
