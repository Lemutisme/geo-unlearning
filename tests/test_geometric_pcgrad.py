import copy
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from trainer.unlearn.component_buffers import ComponentGradientBuffers
from trainer.unlearn.gu_diagnostics import (
    ActualDeltaCallback,
    SurgeryDiagnosticsWriter,
    aggregate_surgery_records,
)
from tests.helpers import TinyCausalLM, make_unlearn_batch
from tests.test_geometric_adam import make_geometric_trainer, unbatch


def make_finalized_gradients(tmp_path, model, mode, dot_sign):
    trainer, _, _ = make_geometric_trainer(tmp_path, model=model)
    trainer.gradient_surgery = mode
    trainer.create_optimizer()
    trainer._validate_gu_runtime()
    named_params = trainer._selected_named_parameters()
    forget_grads = [torch.zeros_like(parameter) for _, parameter in named_params]
    retain_grads = [torch.zeros_like(parameter) for _, parameter in named_params]
    forget_grads[0].reshape(-1)[0] = 2.0
    retain_grads[0].reshape(-1)[0] = float(dot_sign)
    trainer.component_buffers.add("forget", named_params, forget_grads)
    trainer.component_buffers.add("retain", named_params, retain_grads)

    trainer._finalize_gu_gradients(named_params)

    gradients = [parameter.grad.detach().clone() for _, parameter in named_params]
    return trainer, gradients


def test_negative_dot_pcgrad_and_gu_write_identical_gradients(tmp_path):
    torch.manual_seed(10)
    gu_model = TinyCausalLM()
    pcgrad_model = copy.deepcopy(gu_model)

    gu_trainer, gu_gradients = make_finalized_gradients(
        tmp_path / "gu",
        gu_model,
        "gu",
        dot_sign=-1,
    )
    pcgrad_trainer, pcgrad_gradients = make_finalized_gradients(
        tmp_path / "pcgrad",
        pcgrad_model,
        "pcgrad",
        dot_sign=-1,
    )

    for gu_gradient, pcgrad_gradient in zip(gu_gradients, pcgrad_gradients):
        torch.testing.assert_close(gu_gradient, pcgrad_gradient, rtol=0, atol=0)
    assert gu_trainer.last_surgery_diagnostics["conflict"] is True
    assert pcgrad_trainer.last_surgery_diagnostics["conflict"] is True
    assert pcgrad_trainer.last_surgery_diagnostics[
        "relative_pcgrad_gu_distance"
    ] == pytest.approx(0.0)


def test_positive_dot_pcgrad_leaves_forget_component_unprojected(tmp_path):
    torch.manual_seed(11)
    gu_model = TinyCausalLM()
    pcgrad_model = copy.deepcopy(gu_model)

    gu_trainer, gu_gradients = make_finalized_gradients(
        tmp_path / "gu",
        gu_model,
        "gu",
        dot_sign=1,
    )
    pcgrad_trainer, pcgrad_gradients = make_finalized_gradients(
        tmp_path / "pcgrad",
        pcgrad_model,
        "pcgrad",
        dot_sign=1,
    )

    assert pcgrad_trainer.last_surgery_diagnostics["coefficient"] == 0.0
    assert pcgrad_trainer.last_surgery_diagnostics["raw_coefficient"] == pytest.approx(
        2.0
    )
    assert pcgrad_trainer.last_surgery_diagnostics["conflict"] is False
    assert pcgrad_trainer.last_surgery_diagnostics["relative_pcgrad_gu_distance"] > 0.0
    assert not torch.equal(gu_gradients[0], pcgrad_gradients[0])
    expected_first_coordinate = 0.125 * 2.0 + 1.0
    assert pcgrad_gradients[0].reshape(-1)[0].item() == pytest.approx(
        expected_first_coordinate
    )
    assert gu_gradients[0].reshape(-1)[0].item() == pytest.approx(1.0)


def test_zero_retain_norm_is_explicit_in_surgery_diagnostics(tmp_path):
    trainer, _ = make_finalized_gradients(
        tmp_path,
        TinyCausalLM(),
        "pcgrad",
        dot_sign=0,
    )

    assert trainer.last_surgery_diagnostics["zero_retain_norm"] is True


def test_pcgrad_lifecycle_runs_once_per_optimizer_update(tmp_path):
    dataset = unbatch(make_unlearn_batch(batch_size=4, sequence_length=6, seed=31))
    trainer, _, _ = make_geometric_trainer(
        tmp_path,
        train_dataset=dataset,
        per_device_train_batch_size=2,
        max_steps=2,
    )
    trainer.gradient_surgery = "pcgrad"

    trainer.train()

    assert trainer.surgery_calls == trainer.state.global_step == 2
    assert trainer.last_surgery_diagnostics["mode"] == "pcgrad"
    assert "relative_pcgrad_gu_distance" in trainer.last_surgery_diagnostics
    assert trainer.component_buffers.empty


def test_legacy_test_helper_defaults_to_pcgrad(tmp_path):
    trainer, _, _ = make_geometric_trainer(tmp_path)

    assert trainer.gradient_surgery == "pcgrad"


def test_unknown_gradient_surgery_fails_during_initialization(tmp_path):
    trainer, _, _ = make_geometric_trainer(tmp_path)
    trainer.gradient_surgery = "unknown"
    trainer.create_optimizer()

    with pytest.raises(ValueError, match="Unsupported gradient surgery mode"):
        trainer._validate_gu_runtime()


def test_geometric_config_exposes_pcgrad_and_streaming_defaults():
    config = OmegaConf.load("configs/trainer/GeometricUnlearn.yaml")
    geometric = config.method_args.geometric_config

    assert geometric.gradient_surgery == "pcgrad"
    assert geometric.component_buffer_device == "parameter"
    assert geometric.diagnostics_path is None
    assert geometric.actual_delta_mode == "off"
    assert geometric.actual_delta_steps == [1, 10]
    assert geometric.actual_delta_sample_elements == 1_000_000
    assert config.args.save_strategy == "no"


def test_surgery_record_aggregation_ignores_missing_optional_distances():
    records = [
        {
            "record_type": "geometry",
            "conflict": True,
            "relative_orthogonality_residual": 2e-7,
            "relative_surgery_magnitude": 0.2,
            "relative_pcgrad_gu_distance": None,
        },
        {
            "record_type": "geometry",
            "conflict": False,
            "relative_orthogonality_residual": 5e-7,
            "relative_surgery_magnitude": 0.4,
            "relative_pcgrad_gu_distance": 0.3,
        },
    ]

    summary = aggregate_surgery_records(records)

    assert summary["surgery_count"] == 2
    assert summary["conflict_rate"] == pytest.approx(0.5)
    assert summary["maximum_relative_orthogonality_residual"] == pytest.approx(5e-7)
    assert summary["mean_relative_surgery_magnitude"] == pytest.approx(0.3)
    assert summary["mean_relative_pcgrad_gu_distance"] == pytest.approx(0.3)
    assert summary["pcgrad_gu_distance_count"] == 1


def test_diagnostics_writer_appends_sorted_jsonl_and_atomically_summarizes(tmp_path):
    path = tmp_path / "geometry.jsonl"
    writer = SurgeryDiagnosticsWriter(path)
    writer.write_step(
        {
            "record_type": "geometry",
            "update_step": 1,
            "conflict": True,
            "relative_orthogonality_residual": 1e-8,
            "relative_surgery_magnitude": 0.1,
            "relative_pcgrad_gu_distance": 0.0,
        }
    )
    writer.write_step(
        {
            "record_type": "geometry",
            "update_step": 2,
            "conflict": False,
            "relative_orthogonality_residual": 2e-8,
            "relative_surgery_magnitude": 0.3,
        }
    )

    summary = writer.write_summary()
    lines = path.read_text().splitlines()
    persisted = json.loads(writer.summary_path.read_text())

    assert len(lines) == 2
    assert lines[0].startswith('{"conflict"')
    assert summary == persisted
    assert summary["mean_relative_pcgrad_gu_distance"] == 0.0
    assert summary["pcgrad_gu_distance_count"] == 1


def test_full_actual_delta_callback_reports_exact_component_derivatives():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    named_params = [("weight", parameter)]
    buffers = ComponentGradientBuffers("parameter", pin_memory=False)
    buffers.add("forget", named_params, [torch.tensor([2.0, -1.0, 0.5])])
    buffers.add("retain", named_params, [torch.tensor([-3.0, 4.0, 1.0])])
    callback = ActualDeltaCallback(mode="full", steps=[1], sample_elements=2)
    callback.prepare_step(1, named_params, buffers)
    state = SimpleNamespace(global_step=0)
    control = SimpleNamespace()

    returned = callback.on_pre_optimizer_step(None, state, control)
    delta = torch.tensor([0.1, -0.2, 0.3])
    with torch.no_grad():
        parameter.add_(delta)
    returned_after = callback.on_optimizer_step(None, state, control)

    assert returned is control
    assert returned_after is control
    assert callback.last_record["coverage"] == "full"
    assert callback.last_record["forget_directional_derivative"] == pytest.approx(
        (torch.tensor([2.0, -1.0, 0.5]) * delta).sum().item()
    )
    assert callback.last_record["retain_directional_derivative"] == pytest.approx(
        (torch.tensor([-3.0, 4.0, 1.0]) * delta).sum().item()
    )
    assert callback._snapshots == {}
    assert callback._component_probes == {}


def test_sampled_actual_delta_indices_are_deterministic_and_labeled():
    first_parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    second_parameter = torch.nn.Parameter(torch.arange(8, 20, dtype=torch.float32))
    named_params = [("first", first_parameter), ("second", second_parameter)]
    buffers = ComponentGradientBuffers("parameter", pin_memory=False)
    buffers.add(
        "forget",
        named_params,
        [torch.ones_like(first_parameter), torch.ones_like(second_parameter)],
    )
    buffers.add(
        "retain",
        named_params,
        [torch.ones_like(first_parameter), torch.ones_like(second_parameter)],
    )
    callbacks = [
        ActualDeltaCallback(mode="sampled", steps=[1], sample_elements=5)
        for _ in range(2)
    ]
    for callback in callbacks:
        callback.prepare_step(1, named_params, buffers)

    first_indices = {
        name: probe["indices"].clone()
        for name, probe in callbacks[0]._component_probes.items()
    }
    second_indices = {
        name: probe["indices"].clone()
        for name, probe in callbacks[1]._component_probes.items()
    }
    assert first_indices.keys() == second_indices.keys()
    for name in first_indices:
        torch.testing.assert_close(first_indices[name], second_indices[name])
    assert sum(indices.numel() for indices in first_indices.values()) == 5

    state = SimpleNamespace(global_step=0)
    control = SimpleNamespace()
    callbacks[0].on_pre_optimizer_step(None, state, control)
    with torch.no_grad():
        first_parameter.add_(1.0)
        second_parameter.add_(1.0)
    callbacks[0].on_optimizer_step(None, state, control)

    assert callbacks[0].last_record["coverage"] == "sampled"
    assert callbacks[0].last_record["sampled_elements"] == 5


def test_trainer_persists_geometry_and_post_step_delta_without_checkpoint(tmp_path):
    diagnostics_path = tmp_path / "diagnostics" / "steps.jsonl"
    dataset = unbatch(make_unlearn_batch(batch_size=2, sequence_length=6, seed=41))
    trainer, _, _ = make_geometric_trainer(
        tmp_path / "trainer",
        train_dataset=dataset,
        per_device_train_batch_size=2,
        max_steps=1,
        geometric_overrides={
            "gradient_surgery": "pcgrad",
            "diagnostics_path": str(diagnostics_path),
            "actual_delta_mode": "full",
            "actual_delta_steps": [1],
            "actual_delta_sample_elements": 10,
        },
    )

    trainer.train()

    records = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
    assert [record["record_type"] for record in records] == [
        "geometry",
        "actual_delta",
    ]
    assert records[0]["update_step"] == records[1]["update_step"] == 1
    assert records[1]["coverage"] == "full"
    summary = json.loads(diagnostics_path.with_suffix(".summary.json").read_text())
    assert summary["surgery_count"] == 1
    assert not list(tmp_path.rglob("checkpoint-*"))
