import copy

import pytest
import torch
from omegaconf import OmegaConf

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
    assert pcgrad_trainer.last_surgery_diagnostics[
        "relative_pcgrad_gu_distance"
    ] > 0.0
    assert not torch.equal(gu_gradients[0], pcgrad_gradients[0])
    expected_first_coordinate = 0.125 * 2.0 + 1.0
    assert pcgrad_gradients[0].reshape(-1)[0].item() == pytest.approx(
        expected_first_coordinate
    )
    assert gu_gradients[0].reshape(-1)[0].item() == pytest.approx(1.0)


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


def test_historical_config_without_strategy_defaults_to_gu(tmp_path):
    trainer, _, _ = make_geometric_trainer(tmp_path)

    assert trainer.gradient_surgery == "gu"


def test_unknown_gradient_surgery_fails_during_initialization(tmp_path):
    trainer, _, _ = make_geometric_trainer(tmp_path)
    trainer.gradient_surgery = "unknown"
    trainer.create_optimizer()

    with pytest.raises(ValueError, match="Unsupported gradient surgery mode"):
        trainer._validate_gu_runtime()


def test_geometric_config_exposes_pcgrad_and_streaming_defaults():
    config = OmegaConf.load("configs/trainer/GeometricUnlearn.yaml")
    geometric = config.method_args.geometric_config

    assert geometric.gradient_surgery == "gu"
    assert geometric.component_buffer_device == "parameter"
    assert geometric.diagnostics_path is None
    assert geometric.actual_delta_mode == "off"
    assert geometric.actual_delta_steps == [1, 10]
    assert geometric.actual_delta_sample_elements == 1_000_000
