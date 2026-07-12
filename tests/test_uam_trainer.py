from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import TrainingArguments

from trainer import TRAINER_REGISTRY
import trainer.unlearn.uam as uam_module
from trainer.unlearn.component_buffers import ComponentGradientBuffers
from trainer.unlearn.optimizer_geometry import TorchAdamGeometryAdapter
from trainer.unlearn.uam import UAMUnlearn
from tests.helpers import TinyCausalLM, nested_collator


def make_uam_trainer(tmp_path, **overrides):
    model = overrides.pop("model", TinyCausalLM())
    train_dataset = overrides.pop("train_dataset", None)
    retain_loss_type = overrides.pop("retain_loss_type", "NLL")
    uam_config = SimpleNamespace(
        mode=overrides.pop("mode", "uam"),
        forget_signal=overrides.pop("forget_signal", "nll"),
        perturbation_normalization=overrides.pop(
            "perturbation_normalization",
            "auto",
        ),
        rho=overrides.pop("rho", 5e-2),
        reflection_gamma=overrides.pop("reflection_gamma", 2.0),
        residual_lambda=overrides.pop("residual_lambda", 1.0),
        sign_tau=overrides.pop("sign_tau", 0.0),
        replay_device=overrides.pop("replay_device", "cpu"),
    )
    geometric_config = SimpleNamespace(
        loss="simnpo",
        gu_enabled=True,
        gradient_surgery="gu",
        projection_eps=1e-12,
        trainable_params_regex=overrides.pop("trainable_params_regex", [".*"]),
        component_buffer_device=overrides.pop(
            "component_buffer_device",
            "parameter",
        ),
        diagnostics_path=None,
        actual_delta_mode="off",
        actual_delta_steps=[1, 10],
        actual_delta_sample_elements=1000,
    )
    simnpo_config = SimpleNamespace(
        delta=0.0,
        beta=4.5,
        alpha=1.0,
        gamma=0.125,
        retain_loss_type=retain_loss_type,
    )
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=overrides.pop(
            "per_device_train_batch_size",
            2,
        ),
        gradient_accumulation_steps=overrides.pop(
            "gradient_accumulation_steps",
            1,
        ),
        max_steps=overrides.pop("max_steps", -1),
        learning_rate=1e-3,
        optim="adamw_torch",
        adam_beta1=0.0,
        weight_decay=0.0,
        remove_unused_columns=False,
        disable_tqdm=True,
        save_strategy="no",
    )
    assert not overrides, overrides
    trainer = UAMUnlearn(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=nested_collator if train_dataset is not None else None,
        gamma=0.125,
        alpha=1.0,
        retain_loss_type=retain_loss_type,
        geometric_config=geometric_config,
        uam_config=uam_config,
        simnpo_config=simnpo_config,
        npo_config=None,
        dpo_config=None,
        undial_config=None,
        wga_config=None,
        satimp_config=None,
    )
    return trainer, model


def test_uam_trainer_is_registered():
    assert TRAINER_REGISTRY["UAMUnlearn"] is UAMUnlearn


def test_uam_yaml_resolves_safe_defaults():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="unlearn",
            overrides=["trainer=UAMUnlearn"],
        )

    trainer = config.trainer
    assert trainer.handler == "UAMUnlearn"
    assert trainer.args.optim == "adamw_torch"
    assert trainer.args.adam_beta1 == 0.0
    assert trainer.args.weight_decay == 0.0
    assert trainer.args.fp16 is False
    assert trainer.args.save_strategy == "no"
    assert trainer.args.gradient_checkpointing_kwargs.use_reentrant is False
    assert trainer.method_args.retain_loss_type == "NLL"
    assert OmegaConf.to_container(trainer.method_args.uam_config) == {
        "mode": "uam",
        "forget_signal": "nll",
        "perturbation_normalization": "auto",
        "rho": 5e-5,
        "reflection_gamma": 2.0,
        "residual_lambda": 1.0,
        "sign_tau": 0.0,
        "replay_device": "cpu",
    }


def test_uam_initialization_resolves_values_and_empty_state(tmp_path):
    trainer, _ = make_uam_trainer(
        tmp_path,
        mode="UAM",
        forget_signal="NLL",
        perturbation_normalization="FIXED_LOSS",
        rho="0.25",
        reflection_gamma="-2.0",
        residual_lambda="3.0",
        sign_tau="4.0",
        replay_device="CPU",
    )

    assert trainer.uam_mode == "uam"
    assert trainer.forget_signal == "nll"
    assert trainer.perturbation_normalization == "fixed_loss"
    assert trainer.rho == 0.25
    assert trainer.reflection_gamma == -2.0
    assert trainer.residual_lambda == 3.0
    assert trainer.sign_tau == 4.0
    assert trainer.replay_device == "cpu"
    assert trainer.uam_calls == trainer.replay_calls == 0
    assert trainer.last_uam_diagnostics == {}
    assert trainer._uam_runtime_validated is False
    assert trainer._uam_microsteps == 0
    assert trainer._retain_replay_batches == []


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("uam", "fixed_loss"), ("uam_gu", "metric_trust")],
)
def test_auto_normalization_resolves_per_mode(tmp_path, mode, expected):
    trainer, _ = make_uam_trainer(
        tmp_path,
        mode=mode,
        perturbation_normalization="AUTO",
    )

    assert trainer.perturbation_normalization == expected


@pytest.mark.parametrize(
    ("attribute", "invalid", "exception", "message"),
    [
        ("uam_mode", "other", ValueError, "Unsupported UAM mode"),
        (
            "forget_signal",
            "other",
            ValueError,
            "Unsupported UAM forget signal",
        ),
        (
            "perturbation_normalization",
            "other",
            ValueError,
            "perturbation normalization",
        ),
        ("replay_device", "cuda", ValueError, "replay device"),
        ("rho", 0.0, ValueError, "rho must be positive"),
        ("rho", -1.0, ValueError, "rho must be positive"),
        ("rho", float("nan"), ValueError, "rho must be positive"),
        ("rho", float("inf"), ValueError, "rho must be positive"),
        (
            "reflection_gamma",
            float("nan"),
            ValueError,
            "reflection_gamma must be finite",
        ),
        (
            "reflection_gamma",
            float("inf"),
            ValueError,
            "reflection_gamma must be finite",
        ),
        (
            "residual_lambda",
            -1.0,
            ValueError,
            "residual_lambda must be non-negative",
        ),
        (
            "residual_lambda",
            float("nan"),
            ValueError,
            "residual_lambda must be non-negative",
        ),
        (
            "residual_lambda",
            float("inf"),
            ValueError,
            "residual_lambda must be non-negative",
        ),
        ("sign_tau", -1.0, ValueError, "sign_tau must be non-negative"),
        (
            "sign_tau",
            float("nan"),
            ValueError,
            "sign_tau must be non-negative",
        ),
        (
            "sign_tau",
            float("inf"),
            ValueError,
            "sign_tau must be non-negative",
        ),
        ("retain_loss_type", "KL", NotImplementedError, "requires NLL"),
    ],
)
def test_static_runtime_validation_fails_closed(
    tmp_path,
    attribute,
    invalid,
    exception,
    message,
):
    trainer, _ = make_uam_trainer(tmp_path)
    trainer.create_optimizer()
    setattr(trainer, attribute, invalid)

    with pytest.raises(exception, match=message):
        trainer._validate_uam_runtime()

    assert trainer._uam_runtime_validated is False


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("is_deepspeed_enabled", "DeepSpeed"),
        ("is_fsdp_enabled", "FSDP"),
        ("use_apex", "Apex"),
    ],
)
def test_unsupported_trainer_backends_fail_closed(
    tmp_path,
    attribute,
    message,
):
    trainer, _ = make_uam_trainer(tmp_path)
    trainer.create_optimizer()
    setattr(trainer, attribute, True)

    with pytest.raises(NotImplementedError, match=message):
        trainer._validate_uam_runtime()

    assert trainer._uam_runtime_validated is False


def test_fp16_runtime_fails_closed(tmp_path):
    trainer, _ = make_uam_trainer(tmp_path)
    trainer.create_optimizer()
    trainer.args.fp16 = True

    with pytest.raises(NotImplementedError, match="BF16/FP32"):
        trainer._validate_uam_runtime()


def test_multi_process_runtime_fails_closed(tmp_path, monkeypatch):
    trainer, _ = make_uam_trainer(tmp_path)
    trainer.create_optimizer()
    monkeypatch.setattr(trainer.args.distributed_state, "num_processes", 2)

    with pytest.raises(NotImplementedError, match="one process and one GPU"):
        trainer._validate_uam_runtime()


def test_multi_gpu_runtime_fails_closed(tmp_path, monkeypatch):
    trainer, _ = make_uam_trainer(tmp_path)
    trainer.create_optimizer()
    monkeypatch.setattr(trainer.args, "_n_gpu", 2)

    with pytest.raises(NotImplementedError, match="one process and one GPU"):
        trainer._validate_uam_runtime()


def test_reentrant_checkpointing_fails_closed(tmp_path):
    trainer, _ = make_uam_trainer(tmp_path)
    trainer.create_optimizer()
    trainer.args.gradient_checkpointing = True
    trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    with pytest.raises(NotImplementedError, match="use_reentrant=false"):
        trainer._validate_uam_runtime()


def test_empty_parameter_selection_fails_closed(tmp_path):
    trainer, _ = make_uam_trainer(
        tmp_path,
        trainable_params_regex=["does-not-match"],
    )
    trainer.create_optimizer()

    with pytest.raises(ValueError, match="at least one selected trainable parameter"):
        trainer._validate_uam_runtime()


def test_sgd_fails_closed(tmp_path):
    trainer, model = make_uam_trainer(tmp_path)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    with pytest.raises(NotImplementedError, match="SGD"):
        trainer._validate_uam_runtime()


@pytest.mark.parametrize(
    ("mutation", "exception", "message"),
    [
        ("beta1", NotImplementedError, "beta1=0"),
        ("weight_decay", NotImplementedError, "weight_decay=0"),
        ("amsgrad", NotImplementedError, "AMSGrad"),
        ("zero_eps", ValueError, "positive finite eps"),
        ("nan_eps", ValueError, "positive finite eps"),
        ("infinite_eps", ValueError, "positive finite eps"),
    ],
)
def test_optimizer_contract_failures_are_propagated(
    tmp_path,
    mutation,
    exception,
    message,
):
    trainer, _ = make_uam_trainer(tmp_path)
    trainer.create_optimizer()
    group = trainer.optimizer.param_groups[0]
    if mutation == "beta1":
        group["betas"] = (0.9, group["betas"][1])
    elif mutation == "weight_decay":
        group["weight_decay"] = 0.1
    elif mutation == "amsgrad":
        group["amsgrad"] = True
    elif mutation == "zero_eps":
        group["eps"] = 0.0
    elif mutation == "nan_eps":
        group["eps"] = float("nan")
    elif mutation == "infinite_eps":
        group["eps"] = float("inf")

    with pytest.raises(exception, match=message):
        trainer._validate_uam_runtime()

    assert trainer._uam_runtime_validated is False


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("reflection_gamma", [-3.0, 0.0, 2.0])
def test_torch_adam_variants_and_finite_gamma_are_supported(
    tmp_path,
    optimizer_type,
    reflection_gamma,
):
    trainer, model = make_uam_trainer(
        tmp_path,
        reflection_gamma=reflection_gamma,
        retain_loss_type="nll",
    )
    trainer.optimizer = optimizer_type(
        model.parameters(),
        lr=1e-3,
        betas=(0.0, 0.9),
        eps=1e-8,
        weight_decay=0.0,
    )

    trainer._validate_uam_runtime()

    assert trainer._uam_runtime_validated is True
    assert isinstance(trainer._optimizer_geometry_adapter, TorchAdamGeometryAdapter)


def test_runtime_validation_is_cached(tmp_path, monkeypatch):
    trainer, _ = make_uam_trainer(tmp_path)
    trainer.create_optimizer()
    real_factory = uam_module.make_optimizer_geometry_adapter
    calls = 0

    def counting_factory(optimizer):
        nonlocal calls
        calls += 1
        return real_factory(optimizer)

    monkeypatch.setattr(
        uam_module,
        "make_optimizer_geometry_adapter",
        counting_factory,
    )

    trainer._validate_uam_runtime()
    trainer._validate_uam_runtime()

    assert calls == 1


def test_cpu_buffers_validate_three_component_host_memory(tmp_path, monkeypatch):
    trainer, _ = make_uam_trainer(
        tmp_path,
        component_buffer_device="cpu",
    )
    trainer.create_optimizer()
    captured = {}

    def capture_host_memory(
        selected_numel,
        headroom=1.2,
        available_bytes=None,
        component_count=2,
    ):
        captured.update(
            selected_numel=selected_numel,
            headroom=headroom,
            available_bytes=available_bytes,
            component_count=component_count,
        )
        return 0

    monkeypatch.setattr(
        ComponentGradientBuffers,
        "validate_host_memory",
        staticmethod(capture_host_memory),
    )

    trainer._validate_uam_runtime()

    expected_numel = sum(
        parameter.numel() for _, parameter in trainer._selected_named_parameters()
    )
    assert captured["selected_numel"] == expected_numel
    assert captured["component_count"] == 3


def test_host_memory_failure_does_not_partially_cache_runtime(tmp_path, monkeypatch):
    trainer, _ = make_uam_trainer(
        tmp_path,
        component_buffer_device="cpu",
    )
    trainer.create_optimizer()

    def reject_host_memory(*args, **kwargs):
        raise RuntimeError("insufficient host memory")

    monkeypatch.setattr(
        ComponentGradientBuffers,
        "validate_host_memory",
        staticmethod(reject_host_memory),
    )

    with pytest.raises(RuntimeError, match="insufficient host memory"):
        trainer._validate_uam_runtime()

    assert trainer._optimizer_geometry_adapter is None
    assert trainer._uam_runtime_validated is False
