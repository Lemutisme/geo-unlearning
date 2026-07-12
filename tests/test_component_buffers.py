import math

import pytest
import torch

from trainer.unlearn.component_buffers import ComponentGradientBuffers


def make_named_parameters(device="cpu"):
    return [
        ("a", torch.nn.Parameter(torch.zeros(2, device=device))),
        ("b", torch.nn.Parameter(torch.zeros(3, device=device))),
    ]


@pytest.mark.parametrize("device_mode", ["parameter", "cpu"])
def test_component_buffers_accumulate_and_stream_fp32(device_mode):
    named_params = make_named_parameters()
    buffers = ComponentGradientBuffers(device_mode=device_mode, pin_memory=False)
    forget = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
    retain = [torch.tensor([-1.0, 1.0]), None]

    buffers.add("forget", named_params, forget)
    buffers.add("forget", named_params, forget)
    buffers.add("retain", named_params, retain)

    assert buffers.has_component("forget")
    assert buffers.has_component("retain")
    assert not buffers.empty
    torch.testing.assert_close(
        buffers.tensor("forget", "a", named_params[0][1].device),
        torch.tensor([2.0, 4.0]),
    )
    torch.testing.assert_close(
        buffers.tensor("forget", "b", named_params[1][1].device),
        torch.tensor([6.0, 8.0, 10.0]),
    )
    assert buffers.tensor("retain", "b", named_params[1][1].device) is None
    assert buffers.tensor("forget", "a", torch.device("cpu")).dtype == torch.float32


def test_parameter_and_cpu_buffers_are_equivalent():
    named_params = make_named_parameters()
    gradients = [
        torch.tensor([1.25, -2.5]),
        torch.tensor([0.5, 1.5, -3.0]),
    ]
    parameter_buffers = ComponentGradientBuffers("parameter", pin_memory=False)
    cpu_buffers = ComponentGradientBuffers("cpu", pin_memory=False)

    for buffers in (parameter_buffers, cpu_buffers):
        buffers.add("forget", named_params, gradients)
        buffers.add("retain", named_params, [gradient * 2 for gradient in gradients])

    for name, parameter in named_params:
        for component in ("forget", "retain"):
            torch.testing.assert_close(
                parameter_buffers.tensor(component, name, parameter.device),
                cpu_buffers.tensor(component, name, parameter.device),
            )


def test_perturbed_retain_buffer_is_independent_from_existing_components():
    named_params = make_named_parameters()
    buffers = ComponentGradientBuffers("cpu", pin_memory=False)

    buffers.add(
        "forget",
        named_params,
        [torch.full_like(parameter, 1.0) for _, parameter in named_params],
    )
    buffers.add(
        "retain",
        named_params,
        [torch.full_like(parameter, 2.0) for _, parameter in named_params],
    )
    buffers.add(
        "perturbed_retain",
        named_params,
        [torch.full_like(parameter, 3.0) for _, parameter in named_params],
    )

    assert buffers.has_component("perturbed_retain")
    torch.testing.assert_close(
        buffers.tensor("forget", "a", torch.device("cpu")),
        torch.full((2,), 1.0),
    )
    torch.testing.assert_close(
        buffers.tensor("retain", "a", torch.device("cpu")),
        torch.full((2,), 2.0),
    )
    torch.testing.assert_close(
        buffers.tensor("perturbed_retain", "a", torch.device("cpu")),
        torch.full((2,), 3.0),
    )


def test_component_buffer_lifecycle_releases_entries():
    named_params = make_named_parameters()
    buffers = ComponentGradientBuffers("cpu", pin_memory=False)
    gradients = [torch.ones_like(parameter) for _, parameter in named_params]
    buffers.add("forget", named_params, gradients)
    buffers.add("retain", named_params, gradients)

    buffers.clear_component("forget")

    assert not buffers.has_component("forget")
    assert buffers.has_component("retain")
    assert buffers.tensor("forget", "a", torch.device("cpu")) is None

    buffers.clear()
    assert buffers.empty


def test_storage_tensor_exposes_existing_storage_without_device_copy():
    named_params = make_named_parameters()
    buffers = ComponentGradientBuffers("parameter", pin_memory=False)
    buffers.add(
        "forget",
        named_params,
        [torch.ones_like(parameter) for _, parameter in named_params],
    )

    storage = buffers.storage_tensor("forget", "a")

    assert storage is buffers._data["forget"]["a"]
    assert buffers.storage_tensor("retain", "a") is None


def test_perturbed_retain_buffer_lifecycle_is_independent():
    named_params = make_named_parameters()
    buffers = ComponentGradientBuffers("cpu", pin_memory=False)
    gradients = [torch.ones_like(parameter) for _, parameter in named_params]
    for component in ("forget", "retain", "perturbed_retain"):
        buffers.add(component, named_params, gradients)

    buffers.clear_component("perturbed_retain")

    assert buffers.has_component("forget")
    assert buffers.has_component("retain")
    assert not buffers.has_component("perturbed_retain")
    assert buffers.tensor("perturbed_retain", "a", torch.device("cpu")) is None

    buffers.add("perturbed_retain", named_params, gradients)
    buffers.clear()

    assert buffers.empty
    assert not buffers.has_component("forget")
    assert not buffers.has_component("retain")
    assert not buffers.has_component("perturbed_retain")


def test_host_memory_requirement_accounts_for_two_fp32_components():
    selected_numel = 101
    expected = math.ceil(2 * selected_numel * 4 * 1.2)

    assert ComponentGradientBuffers.required_host_bytes(selected_numel) == expected


def test_host_memory_requirement_accounts_for_three_fp32_components():
    assert (
        ComponentGradientBuffers.required_host_bytes(
            selected_numel=10,
            headroom=1,
            component_count=3,
        )
        == 120
    )


def test_host_memory_requirement_defaults_to_two_components():
    default = ComponentGradientBuffers.required_host_bytes(
        selected_numel=101,
        headroom=1.2,
    )
    explicit = ComponentGradientBuffers.required_host_bytes(
        selected_numel=101,
        headroom=1.2,
        component_count=2,
    )

    assert default == explicit


@pytest.mark.parametrize(
    "component_count",
    [
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param(1.5, id="fractional"),
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="positive-infinity"),
        pytest.param(-math.inf, id="negative-infinity"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative-one"),
        pytest.param(-3, id="negative-three"),
    ],
)
def test_host_memory_requirement_rejects_invalid_component_count(
    component_count,
):
    with pytest.raises(
        ValueError,
        match="component_count must be a positive integer",
    ):
        ComponentGradientBuffers.required_host_bytes(
            selected_numel=10,
            component_count=component_count,
        )


def test_host_memory_preflight_fails_one_byte_below_requirement():
    selected_numel = 1_000
    required = ComponentGradientBuffers.required_host_bytes(selected_numel)

    with pytest.raises(
        RuntimeError,
        match="Insufficient host memory for GU component buffers",
    ):
        ComponentGradientBuffers.validate_host_memory(
            selected_numel,
            available_bytes=required - 1,
        )


def test_host_memory_preflight_accepts_exact_requirement():
    selected_numel = 1_000
    required = ComponentGradientBuffers.required_host_bytes(selected_numel)

    assert (
        ComponentGradientBuffers.validate_host_memory(
            selected_numel,
            available_bytes=required,
        )
        == required
    )


def test_host_memory_preflight_uses_three_component_requirement():
    required = 120

    with pytest.raises(
        RuntimeError,
        match=f"requires {required} bytes, found {required - 1} bytes",
    ):
        ComponentGradientBuffers.validate_host_memory(
            selected_numel=10,
            headroom=1,
            available_bytes=required - 1,
            component_count=3,
        )

    assert (
        ComponentGradientBuffers.validate_host_memory(
            selected_numel=10,
            headroom=1,
            available_bytes=required,
            component_count=3,
        )
        == required
    )


@pytest.mark.parametrize("device_mode", ["gpu", "disk", ""])
def test_unknown_buffer_device_is_rejected(device_mode):
    with pytest.raises(ValueError, match="Unsupported component buffer device"):
        ComponentGradientBuffers(device_mode)


def test_unknown_component_is_rejected():
    named_params = make_named_parameters()
    buffers = ComponentGradientBuffers("cpu", pin_memory=False)

    with pytest.raises(ValueError, match="Unsupported gradient component"):
        buffers.add("other", named_params, [torch.ones(2), torch.ones(3)])


def test_gradient_and_parameter_count_must_match():
    named_params = make_named_parameters()
    buffers = ComponentGradientBuffers("cpu", pin_memory=False)

    with pytest.raises(ValueError, match="same length"):
        buffers.add("forget", named_params, [torch.ones(2)])


@pytest.mark.gpu
def test_cpu_mode_uses_pinned_storage_when_cuda_is_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    named_params = make_named_parameters("cuda")
    buffers = ComponentGradientBuffers("cpu", pin_memory=True)
    gradients = [torch.ones_like(parameter) for _, parameter in named_params]

    buffers.add("forget", named_params, gradients)

    assert buffers._data["forget"]["a"].is_pinned()
    torch.testing.assert_close(
        buffers.tensor("forget", "a", named_params[0][1].device).cpu(),
        torch.ones(2),
    )
