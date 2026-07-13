import pytest
import torch

from trainer.unlearn.wmdp_selection import (
    EXPECTED_WMDP_PARAMETER_NAMES,
    configure_wmdp_trainable_parameters,
    select_wmdp_parameters,
    validate_wmdp_optimizer,
)


class ToyMLP(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.down_proj = torch.nn.Linear(width, width, bias=False)
        self.up_proj = torch.nn.Linear(width, width, bias=False)


class ToyLayer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = ToyMLP(width)


class ToyWMDPModel(torch.nn.Module):
    def __init__(self, layer_count=9):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [ToyLayer(4) for _ in range(layer_count)]
        )


def test_configure_and_select_exact_layer_five_to_seven_down_projections():
    model = ToyWMDPModel()

    selected = configure_wmdp_trainable_parameters(model)

    assert tuple(name for name, _ in selected) == EXPECTED_WMDP_PARAMETER_NAMES
    assert select_wmdp_parameters(model) == selected
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == set(EXPECTED_WMDP_PARAMETER_NAMES)


def test_selection_rejects_a_missing_required_tensor():
    model = ToyWMDPModel(layer_count=7)

    with pytest.raises(ValueError, match="missing"):
        select_wmdp_parameters(model)


def test_selection_rejects_unexpected_trainable_tensor_after_configuration():
    model = ToyWMDPModel()
    configure_wmdp_trainable_parameters(model)
    model.embed.weight.requires_grad_(True)

    with pytest.raises(ValueError, match="unexpected trainable"):
        select_wmdp_parameters(model)


def test_optimizer_validation_accepts_exact_selected_identity_set():
    model = ToyWMDPModel()
    selected = configure_wmdp_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected],
        lr=1e-3,
    )

    validate_wmdp_optimizer(optimizer, selected)


def test_optimizer_validation_rejects_missing_and_extra_parameters():
    model = ToyWMDPModel()
    selected = configure_wmdp_trainable_parameters(model)
    missing = torch.optim.AdamW(
        [selected[0][1], selected[1][1]],
        lr=1e-3,
    )
    with pytest.raises(ValueError, match="missing"):
        validate_wmdp_optimizer(missing, selected)

    extra = torch.optim.AdamW(
        [parameter for _, parameter in selected] + [model.embed.weight],
        lr=1e-3,
    )
    with pytest.raises(ValueError, match="extra"):
        validate_wmdp_optimizer(extra, selected)


def test_optimizer_validation_rejects_duplicate_selected_parameter():
    model = ToyWMDPModel()
    selected = configure_wmdp_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected],
        lr=1e-3,
    )
    optimizer.param_groups[0]["params"].append(selected[0][1])

    with pytest.raises(ValueError, match="duplicate"):
        validate_wmdp_optimizer(optimizer, selected)
