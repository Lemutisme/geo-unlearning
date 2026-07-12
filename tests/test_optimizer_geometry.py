from types import SimpleNamespace

import pytest
import torch

from trainer.unlearn.optimizer_geometry import (
    PagedAdamW32GeometryAdapter,
    TorchAdamGeometryAdapter,
    make_optimizer_geometry_adapter,
)


def make_adamw(parameter=None):
    parameter = (
        torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        if parameter is None
        else parameter
    )
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=1e-3,
        betas=(0.0, 0.9),
        eps=1e-8,
        weight_decay=0.0,
    )
    return parameter, optimizer


def seed_torch_state(optimizer, parameter, *, step=3, exp_avg_sq=None):
    if exp_avg_sq is None:
        exp_avg_sq = torch.tensor([0.09, 0.36])
    optimizer.state[parameter].update(
        {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": exp_avg_sq,
        }
    )


def test_torch_empty_state_uses_identity_coordinates():
    parameter, optimizer = make_adamw()
    adapter = make_optimizer_geometry_adapter(optimizer)

    adapter.validate([("weight", parameter)])

    assert isinstance(adapter, TorchAdamGeometryAdapter)
    assert adapter.name == "torch_adam"
    assert adapter.sqrt_denominator(parameter, optimizer.param_groups[0]) is None


def test_torch_initialized_state_uses_frozen_bias_corrected_denominator():
    parameter, optimizer = make_adamw()
    seed_torch_state(optimizer, parameter, step=3)
    adapter = make_optimizer_geometry_adapter(optimizer)

    actual = adapter.sqrt_denominator(parameter, optimizer.param_groups[0])
    v_hat = torch.tensor([0.09, 0.36]) / (1.0 - 0.9**3)
    expected = (v_hat.sqrt() + 1e-8).sqrt()

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_step", "missing step"),
        ("missing_moment", "missing exp_avg_sq"),
        ("negative", "negative values"),
        ("nonfinite", "non-finite values"),
        ("wrong_shape", "shape does not match"),
        ("zero_step", "non-positive step"),
    ],
)
def test_torch_malformed_initialized_state_fails_closed(mutation, message):
    parameter, optimizer = make_adamw()
    seed_torch_state(optimizer, parameter)
    state = optimizer.state[parameter]

    if mutation == "missing_step":
        del state["step"]
    elif mutation == "missing_moment":
        del state["exp_avg_sq"]
    elif mutation == "negative":
        state["exp_avg_sq"] = -torch.ones_like(parameter)
    elif mutation == "nonfinite":
        state["exp_avg_sq"] = torch.full_like(parameter, float("nan"))
    elif mutation == "wrong_shape":
        state["exp_avg_sq"] = torch.ones(3)
    elif mutation == "zero_step":
        state["step"] = torch.tensor(0.0)

    adapter = make_optimizer_geometry_adapter(optimizer)
    with pytest.raises(RuntimeError, match=message):
        adapter.sqrt_denominator(parameter, optimizer.param_groups[0])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("momentum", "beta1=0"),
        ("weight_decay", "weight_decay=0"),
        ("amsgrad", "AMSGrad"),
        ("zero_eps", "positive finite eps"),
        ("nan_eps", "positive finite eps"),
        ("missing_parameter", "absent from the optimizer"),
    ],
)
def test_adapter_group_and_selection_validation_fails_closed(mutation, message):
    parameter, optimizer = make_adamw()
    named_params = [("weight", parameter)]
    group = optimizer.param_groups[0]

    if mutation == "momentum":
        group["betas"] = (0.9, group["betas"][1])
    elif mutation == "weight_decay":
        group["weight_decay"] = 0.01
    elif mutation == "amsgrad":
        group["amsgrad"] = True
    elif mutation == "zero_eps":
        group["eps"] = 0.0
    elif mutation == "nan_eps":
        group["eps"] = float("nan")
    elif mutation == "missing_parameter":
        named_params = [("other", torch.nn.Parameter(torch.ones(1)))]

    adapter = make_optimizer_geometry_adapter(optimizer)
    with pytest.raises((NotImplementedError, ValueError), match=message):
        adapter.validate(named_params)


def test_accelerate_style_optimizer_wrapper_is_unwrapped():
    _, optimizer = make_adamw()
    wrapper = SimpleNamespace(optimizer=SimpleNamespace(optimizer=optimizer))

    adapter = make_optimizer_geometry_adapter(wrapper)

    assert adapter.optimizer is optimizer


def test_unsupported_optimizer_names_its_class():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)

    with pytest.raises(NotImplementedError, match="SGD"):
        make_optimizer_geometry_adapter(optimizer)


class FakePagedAdamW:
    __module__ = "bitsandbytes.optim.adamw"

    def __init__(self, parameter, *, optim_bits=32, is_paged=True):
        self.is_paged = is_paged
        self.args = SimpleNamespace(optim_bits=optim_bits)
        self.param_groups = [
            {
                "params": [parameter],
                "betas": (0.0, 0.9),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
            }
        ]
        self.state = {parameter: {}}
        self.prefetched = []

    def prefetch_state(self, parameter):
        self.prefetched.append(parameter)


class FakePagedLion(FakePagedAdamW):
    __module__ = "bitsandbytes.optim.lion"


def test_paged_adamw32_reads_state2_after_prefetch():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = FakePagedAdamW(parameter)
    optimizer.state[parameter] = {
        "step": 3,
        "state1": torch.zeros_like(parameter),
        "state2": torch.tensor([0.09, 0.36]),
    }

    adapter = make_optimizer_geometry_adapter(optimizer)
    adapter.validate([("weight", parameter)])
    actual = adapter.sqrt_denominator(parameter, optimizer.param_groups[0])

    expected = ((torch.tensor([0.09, 0.36]) / (1.0 - 0.9**3)).sqrt() + 1e-8).sqrt()
    assert isinstance(adapter, PagedAdamW32GeometryAdapter)
    assert adapter.name == "paged_adamw_32bit"
    assert optimizer.prefetched == [parameter]
    torch.testing.assert_close(actual, expected)


def test_paged_adamw32_empty_state_uses_identity_without_prefetch():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = FakePagedAdamW(parameter)
    adapter = make_optimizer_geometry_adapter(optimizer)

    assert adapter.sqrt_denominator(parameter, optimizer.param_groups[0]) is None
    assert optimizer.prefetched == []


def test_paged_uint8_state_is_rejected_as_paged_adamw8():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = FakePagedAdamW(parameter)
    optimizer.state[parameter] = {
        "step": 1,
        "state1": torch.zeros(2, dtype=torch.uint8),
        "state2": torch.zeros(2, dtype=torch.uint8),
    }
    adapter = make_optimizer_geometry_adapter(optimizer)

    with pytest.raises(NotImplementedError, match="PagedAdamW8"):
        adapter.sqrt_denominator(parameter, optimizer.param_groups[0])


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16])
def test_paged_adamw32_rejects_non_fp32_state2(state_dtype):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = FakePagedAdamW(parameter)
    optimizer.state[parameter] = {
        "step": 1,
        "state1": torch.zeros(2, dtype=state_dtype),
        "state2": torch.zeros(2, dtype=state_dtype),
    }
    adapter = make_optimizer_geometry_adapter(optimizer)

    with pytest.raises(NotImplementedError, match="FP32 state2"):
        adapter.sqrt_denominator(parameter, optimizer.param_groups[0])


def test_paged_non_adamw_optimizer_is_rejected_before_training():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = FakePagedLion(parameter)

    with pytest.raises(NotImplementedError, match="FakePagedLion"):
        make_optimizer_geometry_adapter(optimizer)


@pytest.mark.parametrize(
    ("optim_bits", "is_paged"),
    [(8, True), (32, False)],
)
def test_other_bitsandbytes_adam_modes_are_rejected(optim_bits, is_paged):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = FakePagedAdamW(
        parameter,
        optim_bits=optim_bits,
        is_paged=is_paged,
    )

    with pytest.raises(NotImplementedError, match="FakePagedAdamW"):
        make_optimizer_geometry_adapter(optimizer)
