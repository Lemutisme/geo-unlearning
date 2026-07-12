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
from trainer.unlearn.uam import RetainReplayBuffer, UAMUnlearn
from trainer.utils import compute_batch_nll
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


def make_forget_inputs():
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
        ]
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
        "labels": torch.tensor(
            [
                [-100, -100, 3, 4, -100, -100],
                [-100, 5, -100, 3, 2, -100],
            ]
        ),
    }


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
    assert isinstance(trainer.replay_buffer, RetainReplayBuffer)
    assert trainer.replay_buffer.empty


def test_nll_forget_signal_is_mean_per_sequence_answer_nll(tmp_path):
    torch.manual_seed(7)
    trainer, model = make_uam_trainer(tmp_path, forget_signal="nll")
    forget_inputs = make_forget_inputs()

    sequence_nll, _ = compute_batch_nll(model, forget_inputs)
    answer_counts = forget_inputs["labels"][..., 1:].ne(-100).sum(-1)
    expected = (sequence_nll / answer_counts.to(sequence_nll)).mean()

    signal, _ = trainer.compute_uam_forget_signal(model, forget_inputs)

    torch.testing.assert_close(signal, expected)


def test_nll_forget_signal_returns_raw_forward_output(tmp_path):
    trainer, model = make_uam_trainer(tmp_path, forget_signal="nll")
    forget_inputs = make_forget_inputs()
    forward_outputs = []
    hook = model.register_forward_hook(
        lambda _module, _args, output: forward_outputs.append(output)
    )

    try:
        _, outputs = trainer.compute_uam_forget_signal(model, forget_inputs)
    finally:
        hook.remove()

    assert outputs is forward_outputs[-1]
    assert outputs.logits.shape[:2] == forget_inputs["labels"].shape


def test_nll_forget_signal_ignores_prompt_and_padding_labels(tmp_path):
    torch.manual_seed(11)
    trainer, model = make_uam_trainer(tmp_path, forget_signal="nll")
    forget_inputs = make_forget_inputs()

    signal, outputs = trainer.compute_uam_forget_signal(model, forget_inputs)

    shifted_labels = forget_inputs["labels"][..., 1:]
    answer_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~answer_mask, 0)
    token_nll = (
        -outputs.logits[..., :-1, :]
        .log_softmax(-1)
        .gather(
            -1,
            safe_labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    expected = ((token_nll * answer_mask).sum(-1) / answer_mask.sum(-1)).mean()

    torch.testing.assert_close(signal, expected)


def test_nll_forget_signal_rejects_any_empty_answer_mask(tmp_path):
    trainer, model = make_uam_trainer(tmp_path, forget_signal="nll")
    forget_inputs = make_forget_inputs()
    forget_inputs["labels"][0].fill_(-100)

    with pytest.raises(RuntimeError, match="empty answer mask"):
        trainer.compute_uam_forget_signal(model, forget_inputs)


def test_simnpo_forget_signal_rejects_shifted_empty_answer_mask(tmp_path):
    trainer, model = make_uam_trainer(tmp_path, forget_signal="simnpo")
    forget_inputs = make_forget_inputs()
    forget_inputs["labels"][0].fill_(-100)
    forget_inputs["labels"][0, 0] = forget_inputs["input_ids"][0, 0]

    with pytest.raises(RuntimeError, match="empty answer mask"):
        trainer.compute_uam_forget_signal(model, forget_inputs)


def test_simnpo_forget_signal_is_negative_existing_objective(tmp_path):
    torch.manual_seed(17)
    trainer, model = make_uam_trainer(tmp_path, forget_signal="simnpo")
    forget_inputs = make_forget_inputs()

    simnpo_loss, _ = trainer.compute_forget_loss(model, forget_inputs)
    signal, outputs = trainer.compute_uam_forget_signal(model, forget_inputs)

    torch.testing.assert_close(signal, -simnpo_loss)
    assert outputs.logits.shape[:2] == forget_inputs["labels"].shape


def test_simnpo_signal_gradients_negate_existing_objective_gradients(tmp_path):
    torch.manual_seed(19)
    trainer, model = make_uam_trainer(tmp_path, forget_signal="simnpo")
    forget_inputs = make_forget_inputs()
    parameters = [
        parameter for _, parameter in trainer._selected_named_parameters(model)
    ]

    simnpo_loss, _ = trainer.compute_forget_loss(model, forget_inputs)
    loss_grads = torch.autograd.grad(
        simnpo_loss,
        parameters,
        allow_unused=True,
    )
    signal, _ = trainer.compute_uam_forget_signal(model, forget_inputs)
    signal_grads = torch.autograd.grad(
        signal,
        parameters,
        allow_unused=True,
    )

    for loss_grad, signal_grad in zip(loss_grads, signal_grads, strict=True):
        if loss_grad is None or signal_grad is None:
            assert loss_grad is signal_grad is None
        else:
            torch.testing.assert_close(signal_grad, -loss_grad)


@pytest.mark.parametrize("forget_signal", ["nll", "simnpo"])
def test_small_step_along_forget_signal_gradient_increases_signal(
    tmp_path,
    forget_signal,
):
    torch.manual_seed(23)
    model = TinyCausalLM().double()
    trainer, model = make_uam_trainer(
        tmp_path,
        model=model,
        forget_signal=forget_signal,
    )
    trainer.simnpo_config.beta = 0.25
    forget_inputs = make_forget_inputs()
    named_parameters = trainer._selected_named_parameters(model)
    parameters = [parameter for _, parameter in named_parameters]
    originals = [parameter.detach().clone() for parameter in parameters]
    step_size = 1e-5

    signal, _ = trainer.compute_uam_forget_signal(model, forget_inputs)
    gradients = torch.autograd.grad(signal, parameters, allow_unused=True)
    directional_derivative = sum(
        gradient.square().sum() for gradient in gradients if gradient is not None
    )

    try:
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients, strict=True):
                if gradient is not None:
                    parameter.add_(gradient, alpha=step_size)
        stepped_signal, _ = trainer.compute_uam_forget_signal(
            model,
            forget_inputs,
        )
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)

    predicted_increase = step_size * directional_derivative
    assert directional_derivative > 0
    assert stepped_signal - signal > 0.5 * predicted_increase
    for parameter, original in zip(parameters, originals, strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


@pytest.mark.parametrize("forget_signal", ["nll", "simnpo"])
def test_forget_signal_rejects_non_finite_result(tmp_path, forget_signal):
    trainer, model = make_uam_trainer(
        tmp_path,
        forget_signal=forget_signal,
    )
    forget_inputs = make_forget_inputs()
    original_lm_head = model.lm_head.weight.detach().clone()

    try:
        with torch.no_grad():
            model.lm_head.weight[0, 0] = float("nan")
        with pytest.raises(RuntimeError, match="non-finite"):
            trainer.compute_uam_forget_signal(model, forget_inputs)
    finally:
        with torch.no_grad():
            model.lm_head.weight.copy_(original_lm_head)

    restored_signal, _ = trainer.compute_uam_forget_signal(model, forget_inputs)
    assert torch.isfinite(restored_signal)


def test_unsupported_forget_signal_direct_call_fails_closed(tmp_path):
    trainer, model = make_uam_trainer(tmp_path)
    trainer.forget_signal = "unsupported"

    with pytest.raises(ValueError, match="Unsupported UAM forget signal"):
        trainer.compute_uam_forget_signal(model, make_forget_inputs())


@pytest.mark.parametrize("loss_name", ["npo", "unknown"])
def test_simnpo_signal_direct_call_rejects_other_loss_names(
    tmp_path,
    loss_name,
):
    trainer, model = make_uam_trainer(tmp_path, forget_signal="simnpo")
    trainer.loss_name = loss_name

    with pytest.raises(ValueError, match="requires geometric_config.loss='simnpo'"):
        trainer.compute_uam_forget_signal(model, make_forget_inputs())


def test_simnpo_signal_direct_call_requires_config(tmp_path):
    trainer, model = make_uam_trainer(tmp_path, forget_signal="simnpo")
    trainer.simnpo_config = None

    with pytest.raises(ValueError, match="requires a non-null simnpo_config"):
        trainer.compute_uam_forget_signal(model, make_forget_inputs())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("beta", "4.5", "SimNPO beta"),
        ("beta", "not-a-number", "SimNPO beta"),
        ("delta", "0.0", "SimNPO delta"),
        ("delta", "not-a-number", "SimNPO delta"),
    ],
)
def test_simnpo_signal_direct_call_rejects_string_settings(
    tmp_path,
    field,
    value,
    message,
):
    trainer, model = make_uam_trainer(tmp_path, forget_signal="simnpo")
    setattr(trainer.simnpo_config, field, value)

    with pytest.raises(ValueError, match=message):
        trainer.compute_uam_forget_signal(model, make_forget_inputs())


@pytest.mark.parametrize(
    ("beta", "delta"),
    [(2, 0), (2.0, 0.5)],
)
def test_simnpo_signal_accepts_real_int_and_float_settings(
    tmp_path,
    beta,
    delta,
):
    trainer, model = make_uam_trainer(tmp_path, forget_signal="simnpo")
    trainer.simnpo_config.beta = beta
    trainer.simnpo_config.delta = delta

    signal, _ = trainer.compute_uam_forget_signal(model, make_forget_inputs())

    assert torch.isfinite(signal)
    assert trainer.simnpo_config.beta == beta
    assert trainer.simnpo_config.delta == delta
    assert type(trainer.simnpo_config.beta) is type(beta)
    assert type(trainer.simnpo_config.delta) is type(delta)


@pytest.mark.parametrize(
    ("loss_name", "drop_config", "message"),
    [
        ("npo", False, "requires geometric_config.loss='simnpo'"),
        ("simnpo", True, "requires a non-null simnpo_config"),
    ],
)
def test_simnpo_signal_runtime_rejects_incompatible_objective_config(
    tmp_path,
    loss_name,
    drop_config,
    message,
):
    trainer, _ = make_uam_trainer(tmp_path, forget_signal="simnpo")
    trainer.create_optimizer()
    trainer.loss_name = loss_name
    if drop_config:
        trainer.simnpo_config = None

    with pytest.raises(ValueError, match=message):
        trainer._validate_uam_runtime()

    assert trainer._uam_runtime_validated is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("beta", True, "SimNPO beta"),
        ("beta", False, "SimNPO beta"),
        ("beta", 0.0, "SimNPO beta"),
        ("beta", -1.0, "SimNPO beta"),
        ("beta", float("nan"), "SimNPO beta"),
        ("beta", float("inf"), "SimNPO beta"),
        ("beta", float("-inf"), "SimNPO beta"),
        ("delta", True, "SimNPO delta"),
        ("delta", False, "SimNPO delta"),
        ("delta", float("nan"), "SimNPO delta"),
        ("delta", float("inf"), "SimNPO delta"),
        ("delta", float("-inf"), "SimNPO delta"),
    ],
)
def test_simnpo_signal_runtime_rejects_invalid_objective_scalars(
    tmp_path,
    field,
    value,
    message,
):
    trainer, _ = make_uam_trainer(tmp_path, forget_signal="simnpo")
    trainer.create_optimizer()
    setattr(trainer.simnpo_config, field, value)

    with pytest.raises(ValueError, match=message):
        trainer._validate_uam_runtime()

    assert trainer._uam_runtime_validated is False


def test_nll_signal_runtime_is_independent_of_geometric_loss(tmp_path):
    trainer, _ = make_uam_trainer(tmp_path, forget_signal="nll")
    trainer.create_optimizer()
    trainer.loss_name = "npo"
    trainer.simnpo_config = None

    trainer._validate_uam_runtime()

    assert trainer._uam_runtime_validated is True


@pytest.mark.parametrize("forget_signal", ["nll", "simnpo"])
def test_forget_signal_does_not_mutate_input_tensors(tmp_path, forget_signal):
    trainer, model = make_uam_trainer(
        tmp_path,
        forget_signal=forget_signal,
    )
    forget_inputs = make_forget_inputs()
    originals = {name: tensor.clone() for name, tensor in forget_inputs.items()}

    trainer.compute_uam_forget_signal(model, forget_inputs)

    for name, original in originals.items():
        torch.testing.assert_close(forget_inputs[name], original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field",
    ["rho", "reflection_gamma", "residual_lambda", "sign_tau"],
)
@pytest.mark.parametrize("value", [True, False])
def test_boolean_numeric_config_is_rejected_before_coercion(
    tmp_path,
    field,
    value,
):
    with pytest.raises(ValueError, match=field):
        make_uam_trainer(tmp_path, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rho", 1),
        ("reflection_gamma", 0),
        ("residual_lambda", 0.5),
        ("sign_tau", 0),
    ],
)
def test_normal_numeric_config_remains_supported(tmp_path, field, value):
    trainer, _ = make_uam_trainer(tmp_path, **{field: value})

    assert getattr(trainer, field) == float(value)


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


@pytest.mark.parametrize(
    "checkpointing_kwargs",
    [{}, {"use_reentrant": None}, {"use_reentrant": True}],
)
def test_ambiguous_or_reentrant_checkpointing_fails_closed(
    tmp_path,
    checkpointing_kwargs,
):
    trainer, _ = make_uam_trainer(tmp_path)
    trainer.create_optimizer()
    trainer.args.gradient_checkpointing = True
    trainer.args.gradient_checkpointing_kwargs = checkpointing_kwargs

    with pytest.raises(NotImplementedError, match="use_reentrant=false"):
        trainer._validate_uam_runtime()


def test_explicit_nonreentrant_checkpointing_is_supported(tmp_path):
    trainer, _ = make_uam_trainer(tmp_path)
    trainer.create_optimizer()
    trainer.args.gradient_checkpointing = True
    trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    trainer._validate_uam_runtime()

    assert trainer._uam_runtime_validated is True


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
