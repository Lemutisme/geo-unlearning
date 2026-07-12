from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import TrainingArguments

from trainer import TRAINER_REGISTRY
from trainer.unlearn.orthograd import OrthogradUnlearn
from trainer.unlearn.orthograd_geometry import GlobalRetainBasis
from tests.helpers import (
    TinyCausalLM,
    make_unlearn_batch,
    nested_collator,
)


def make_orthograd_trainer(tmp_path, **overrides):
    model = overrides.pop("model", TinyCausalLM())
    train_dataset = overrides.pop("train_dataset", None)
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
        max_steps=overrides.pop("max_steps", -1),
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
        train_dataset=train_dataset,
        data_collator=nested_collator if train_dataset is not None else None,
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


def unbatch(batch):
    batch_size = batch["forget"]["input_ids"].shape[0]
    return [
        {
            component: {key: tensor[index] for key, tensor in component_batch.items()}
            for component, component_batch in batch.items()
        }
        for index in range(batch_size)
    ]


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


def test_public_training_path_finalizes_once_per_optimizer_update(tmp_path):
    dataset = unbatch(make_unlearn_batch(batch_size=4, sequence_length=6, seed=29))
    trainer, model = make_orthograd_trainer(
        tmp_path,
        train_dataset=dataset,
        per_device_train_batch_size=2,
        max_steps=2,
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]

    trainer.train()

    assert trainer.orthograd_calls == trainer.state.global_step == 2
    assert trainer.retain_basis.empty
    assert trainer.component_buffers.empty
    assert trainer.last_orthograd_diagnostics["requested_rank"] == 2
    assert trainer.last_orthograd_diagnostics["effective_rank"] >= 1
    assert trainer.last_orthograd_diagnostics["maximum_relative_residual"] < 1e-5
    assert any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, model.parameters())
    )


def test_final_gradient_matches_retain_plus_weighted_projected_forget(tmp_path):
    trainer, _ = make_orthograd_trainer(tmp_path)
    trainer.create_optimizer()
    trainer._validate_orthograd_runtime(retain_batch_size=2)
    named_params = trainer._selected_named_parameters()
    forget = [torch.zeros_like(parameter) for _, parameter in named_params]
    retain_mean = [torch.zeros_like(parameter) for _, parameter in named_params]
    sample_a = [torch.zeros_like(parameter) for _, parameter in named_params]
    sample_b = [torch.zeros_like(parameter) for _, parameter in named_params]
    forget[0].reshape(-1)[:3] = torch.tensor([3.0, -2.0, 4.0])
    retain_mean[0].reshape(-1)[:3] = torch.tensor([0.5, 0.5, 1.0])
    sample_a[0].reshape(-1)[:3] = torch.tensor([1.0, 0.0, 1.0])
    sample_b[0].reshape(-1)[:3] = torch.tensor([0.0, 1.0, 1.0])
    trainer.component_buffers.add("forget", named_params, forget)
    trainer.component_buffers.add("retain", named_params, retain_mean)
    for sample in (sample_a, sample_b):
        trainer.retain_basis.add(trainer._basis_vector(named_params, sample))

    reference = GlobalRetainBasis(trainer.rank_tolerance)
    for sample in (sample_a, sample_b):
        reference.add(trainer._basis_vector(named_params, sample))
    candidate = trainer._component_vector("forget", named_params)
    projected, _, _ = reference.project(candidate)
    expected = {
        name: trainer.component_buffers.tensor("retain", name, parameter.device)
        + trainer.forget_weight * projected[name].to(parameter.device)
        for name, parameter in named_params
    }

    trainer._finalize_orthograd_gradients(named_params)

    for name, parameter in named_params:
        torch.testing.assert_close(parameter.grad, expected[name])
    assert trainer.orthograd_calls == 1
    assert trainer.retain_basis.empty
    assert trainer.component_buffers.empty


def test_diagnostics_failure_clears_written_gradients_and_transient_state(tmp_path):
    class FailingWriter:
        def write_step(self, _record):
            raise OSError("injected diagnostics failure")

    trainer, _ = make_orthograd_trainer(tmp_path)
    trainer.create_optimizer()
    trainer._validate_orthograd_runtime(retain_batch_size=1)
    named_params = trainer._selected_named_parameters()
    gradients = [torch.ones_like(parameter) for _, parameter in named_params]
    trainer.component_buffers.add("forget", named_params, gradients)
    trainer.component_buffers.add("retain", named_params, gradients)
    trainer.retain_basis.add(trainer._basis_vector(named_params, gradients))
    trainer.diagnostics_writer = FailingWriter()

    with pytest.raises(OSError, match="injected diagnostics failure"):
        trainer._finalize_orthograd_gradients(named_params)

    assert all(parameter.grad is None for _, parameter in named_params)
    assert trainer.orthograd_calls == 0
    assert trainer.retain_basis.empty
    assert trainer.component_buffers.empty
