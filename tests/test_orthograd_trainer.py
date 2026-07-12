from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import TrainingArguments

from trainer import TRAINER_REGISTRY
from trainer.unlearn.orthograd import OrthogradUnlearn
from tests.helpers import TinyCausalLM, make_unlearn_batch


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


def test_nested_batch_slicing_preserves_leading_dimension_and_metadata(tmp_path):
    trainer, _ = make_orthograd_trainer(tmp_path)
    batch = make_unlearn_batch(batch_size=3, sequence_length=6, seed=17)["retain"]
    batch["nested"] = (batch["attention_mask"].clone(), ["constant"])
    batch["metadata"] = "retain"

    samples = list(trainer._iter_batch_samples(batch))

    assert len(samples) == 3
    assert samples[0]["input_ids"].shape == (1, 6)
    assert samples[0]["nested"][0].shape == (1, 6)
    assert samples[0]["nested"][1] == ["constant"]
    assert samples[0]["metadata"] == "retain"
    assert batch["input_ids"].shape == (3, 6)


def test_per_sample_retain_mean_reconstructs_token_weighted_batch_gradient(tmp_path):
    trainer, model = make_orthograd_trainer(tmp_path)
    retain = make_unlearn_batch(batch_size=3, sequence_length=6, seed=19)["retain"]
    retain["labels"][1, 2:4] = -100
    named_params = trainer._selected_named_parameters()
    params = [parameter for _, parameter in named_params]

    batch_loss = trainer.compute_retain_loss(model, retain)
    batch_gradients = torch.autograd.grad(
        batch_loss,
        params,
        allow_unused=True,
    )
    reconstructed_loss, sample_count = trainer._collect_retain_gradients(
        model,
        retain,
        named_params,
    )

    assert sample_count == 3
    torch.testing.assert_close(reconstructed_loss, batch_loss.detach())
    for (name, parameter), expected in zip(named_params, batch_gradients):
        observed = trainer.component_buffers.tensor("retain", name, parameter.device)
        if expected is None:
            assert observed is None
        else:
            torch.testing.assert_close(observed, expected.float(), rtol=1e-5, atol=1e-6)
    assert trainer.retain_basis.requested_rank == 3


def test_per_sample_collection_rejects_empty_answer_masks_and_clears_state(tmp_path):
    trainer, model = make_orthograd_trainer(tmp_path)
    retain = make_unlearn_batch(batch_size=2, sequence_length=6, seed=23)["retain"]
    retain["labels"][1].fill_(-100)
    named_params = trainer._selected_named_parameters()

    with pytest.raises(RuntimeError, match="empty answer mask"):
        trainer._collect_retain_gradients(model, retain, named_params)

    assert trainer.component_buffers.empty
    assert trainer.retain_basis.empty
