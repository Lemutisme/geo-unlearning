from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from transformers import TrainingArguments

from trainer import TRAINER_REGISTRY
from trainer.unlearn.orthograd import OrthogradUnlearn
from tests.helpers import TinyCausalLM


def make_orthograd_trainer(tmp_path, **overrides):
    model = overrides.pop("model", TinyCausalLM())
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
        actual_delta_steps=[1, 2],
        actual_delta_sample_elements=1000,
    )
    orthograd_config = SimpleNamespace(
        forget_weight=overrides.pop("forget_weight", 0.125),
        retain_weight=overrides.pop("retain_weight", 1.0),
        rank_tolerance=overrides.pop("rank_tolerance", 1e-8),
        maximum_retain_batch_size=overrides.pop("maximum_retain_batch_size", 4),
        basis_device=overrides.pop("basis_device", "parameter"),
        diagnostics_path=None,
    )
    simnpo_config = SimpleNamespace(
        delta=0.0,
        beta=4.5,
        alpha=1.0,
        gamma=0.125,
        retain_loss_type="NLL",
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
        learning_rate=1e-3,
        optim="adamw_torch",
        adam_beta1=overrides.pop("adam_beta1", 0.0),
        weight_decay=overrides.pop("weight_decay", 0.0),
        remove_unused_columns=False,
        disable_tqdm=True,
        save_strategy="no",
    )
    assert not overrides, overrides
    trainer = OrthogradUnlearn(
        model=model,
        args=args,
        gamma=0.125,
        alpha=1.0,
        retain_loss_type="NLL",
        geometric_config=geometric_config,
        orthograd_config=orthograd_config,
        simnpo_config=simnpo_config,
        npo_config=None,
        dpo_config=None,
        undial_config=None,
        wga_config=None,
        satimp_config=None,
    )
    return trainer, model


def test_orthograd_trainer_is_registered():
    assert TRAINER_REGISTRY["OrthogradUnlearn"] is OrthogradUnlearn


def test_orthograd_yaml_has_mvp_defaults():
    config = OmegaConf.load("configs/trainer/OrthogradUnlearn.yaml")

    assert config.handler == "OrthogradUnlearn"
    assert config.args.save_strategy == "no"
    assert config.args.gradient_accumulation_steps == 1
    assert config.args.adam_beta1 == 0.0
    assert config.args.weight_decay == 0.0
    orthograd = config.method_args.orthograd_config
    assert orthograd.forget_weight == 0.125
    assert orthograd.retain_weight == 1.0
    assert orthograd.maximum_retain_batch_size == 4
    assert orthograd.basis_device == "parameter"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gradient_accumulation_steps": 2}, "gradient_accumulation_steps=1"),
        ({"basis_device": "disk"}, "basis device"),
        ({"adam_beta1": 0.9}, "beta1=0"),
        ({"weight_decay": 0.01}, "weight_decay=0"),
        ({"trainable_params_regex": ["embedding.*"]}, "all trainable parameters"),
    ],
)
def test_orthograd_runtime_validation_rejects_unsupported_contract(
    tmp_path,
    overrides,
    message,
):
    trainer, _ = make_orthograd_trainer(tmp_path, **overrides)
    trainer.create_optimizer()

    with pytest.raises((ValueError, NotImplementedError), match=message):
        trainer._validate_orthograd_runtime(retain_batch_size=2)


def test_orthograd_runtime_validation_accepts_cpu_and_parameter_basis(tmp_path):
    for basis_device in ("cpu", "parameter"):
        trainer, _ = make_orthograd_trainer(
            tmp_path / basis_device,
            basis_device=basis_device,
        )
        trainer.create_optimizer()

        trainer._validate_orthograd_runtime(retain_batch_size=2)
