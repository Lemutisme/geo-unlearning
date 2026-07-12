import math

import pytest
import torch

from trainer.unlearn.uam import (
    PerturbationStats,
    RetainReplayBuffer,
    TemporaryParameterPerturbation,
)


class TwoParameterModule(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=dtype))
        self.second = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=dtype))


def clone_parameters(named_params):
    return {name: parameter.detach().clone() for name, parameter in named_params}


@pytest.mark.parametrize("device_mode", ["cpu", "parameter"])
def test_replay_buffer_preserves_nested_data_without_aliasing(device_mode):
    source_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    source = {
        "tensor": source_tensor,
        "nested": [torch.tensor([3]), (torch.tensor([4]), "leaf")],
        "metadata": {"count": 2, "optional": None},
    }
    buffer = RetainReplayBuffer(device_mode)

    buffer.append(source)
    source_tensor.data.fill_(9)
    source["nested"][0].fill_(8)
    source["nested"][1][0].fill_(7)
    source["metadata"]["count"] = 99

    (replayed,) = list(buffer.batches(torch.device("cpu")))
    assert isinstance(replayed, dict)
    assert isinstance(replayed["nested"], list)
    assert isinstance(replayed["nested"][1], tuple)
    assert replayed["tensor"].tolist() == [1.0, 2.0]
    assert replayed["nested"][0].tolist() == [3]
    assert replayed["nested"][1][0].tolist() == [4]
    assert replayed["nested"][1][1] == "leaf"
    assert replayed["metadata"] == {"count": 2, "optional": None}
    assert replayed["tensor"].requires_grad is False
    assert replayed["tensor"].device.type == "cpu"


def test_replay_buffer_repeated_iteration_isolated_from_consumers():
    buffer = RetainReplayBuffer("cpu")
    buffer.append({"values": torch.tensor([1, 2, 3])})

    first = list(buffer.batches("cpu"))
    first[0]["values"].fill_(99)
    second = list(buffer.batches("cpu"))
    second[0]["values"][0] = -1
    third = list(buffer.batches("cpu"))

    assert second[0]["values"].tolist() == [-1, 2, 3]
    assert third[0]["values"].tolist() == [1, 2, 3]
    assert first[0] is not second[0]
    assert first[0]["values"] is not second[0]["values"]


def test_replay_buffer_iteration_does_not_change_consumer_grad_mode():
    buffer = RetainReplayBuffer("cpu")
    buffer.append({"values": torch.tensor([1.0], requires_grad=True)})

    batches = buffer.batches("cpu")
    try:
        replayed = next(batches)
        assert torch.is_grad_enabled()
        assert replayed["values"].requires_grad is False
    finally:
        batches.close()


def test_replay_buffer_len_empty_and_clear():
    buffer = RetainReplayBuffer("parameter")
    assert buffer.empty
    assert len(buffer) == 0

    buffer.append((torch.tensor([1]), "batch"))
    buffer.append((torch.tensor([2]), "batch"))
    assert not buffer.empty
    assert len(buffer) == 2

    buffer.clear()
    assert buffer.empty
    assert len(buffer) == 0
    assert list(buffer.batches("cpu")) == []


@pytest.mark.parametrize("device_mode", ["cuda", "gpu", "", None])
def test_replay_buffer_rejects_unsupported_device_mode(device_mode):
    with pytest.raises(ValueError, match="device mode.*cpu.*parameter"):
        RetainReplayBuffer(device_mode)


def test_fp32_perturbation_is_visible_and_restores_exactly():
    module = TwoParameterModule()
    named_params = list(module.named_parameters())
    originals = clone_parameters(named_params)
    deltas = {
        "first": torch.tensor([0.25, -0.5], requires_grad=True),
        "second": torch.tensor([1.0, 0.125]),
    }

    with TemporaryParameterPerturbation(named_params, deltas) as stats:
        assert isinstance(stats, PerturbationStats)
        assert torch.equal(module.first, originals["first"] + deltas["first"])
        assert torch.equal(module.second, originals["second"] + deltas["second"])
        assert not module.first.requires_grad or module.first.is_leaf
        assert stats.requested_norm > 0.0
        assert stats.effective_norm > 0.0
        assert stats.ratio == pytest.approx(1.0)

    for name, parameter in module.named_parameters():
        assert torch.equal(parameter, originals[name])


def test_perturbation_restores_exactly_when_body_raises():
    module = TwoParameterModule()
    named_params = list(module.named_parameters())
    originals = clone_parameters(named_params)
    body_error = RuntimeError("body failed")

    with pytest.raises(RuntimeError, match="body failed") as caught:
        with TemporaryParameterPerturbation(
            named_params,
            {"first": torch.tensor([0.5, 0.5])},
        ):
            raise body_error

    assert caught.value is body_error
    for name, parameter in module.named_parameters():
        assert torch.equal(parameter, originals[name])


def test_bfloat16_representable_perturbation_reports_effective_ratio():
    module = TwoParameterModule(dtype=torch.bfloat16)
    named_params = list(module.named_parameters())
    originals = clone_parameters(named_params)

    with TemporaryParameterPerturbation(
        named_params,
        {"first": torch.tensor([0.125, -0.25])},
    ) as stats:
        assert not torch.equal(module.first, originals["first"])
        assert math.isfinite(stats.ratio)
        assert stats.ratio > 0.0

    for name, parameter in module.named_parameters():
        assert parameter.dtype is torch.bfloat16
        assert torch.equal(parameter, originals[name])


def test_bfloat16_fully_rounded_perturbation_raises_and_restores():
    module = TwoParameterModule(dtype=torch.bfloat16)
    named_params = list(module.named_parameters())
    originals = clone_parameters(named_params)

    with pytest.raises(RuntimeError, match="rounded entirely to zero"):
        with TemporaryParameterPerturbation(
            named_params,
            {"first": torch.full((2,), 1e-6)},
        ):
            pytest.fail("a fully rounded perturbation must not enter the body")

    for name, parameter in module.named_parameters():
        assert torch.equal(parameter, originals[name])


def test_empty_perturbation_is_allowed():
    context = TemporaryParameterPerturbation([], {})

    with context as stats:
        assert stats.requested_norm == 0.0
        assert stats.effective_norm == 0.0
        assert stats.ratio == 0.0


def test_all_zero_perturbation_leaves_parameters_unchanged():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    original = parameter.detach().clone()

    with TemporaryParameterPerturbation(
        [("weight", parameter)],
        {"weight": torch.zeros(2)},
    ) as stats:
        assert torch.equal(parameter, original)
        assert stats == PerturbationStats()

    assert torch.equal(parameter, original)


def validation_cases():
    return [
        pytest.param(
            [("first", torch.nn.Parameter(torch.tensor([1.0])))],
            {"unknown": torch.tensor([1.0])},
            "unknown",
            id="unknown-delta-name",
        ),
        pytest.param(
            [
                ("duplicate", torch.nn.Parameter(torch.tensor([1.0]))),
                ("duplicate", torch.nn.Parameter(torch.tensor([2.0]))),
            ],
            {"duplicate": torch.tensor([1.0])},
            "duplicate",
            id="duplicate-parameter-name",
        ),
        pytest.param(
            [("first", torch.nn.Parameter(torch.tensor([1.0, 2.0])))],
            {"first": torch.tensor([1.0])},
            "shape",
            id="wrong-shape",
        ),
        pytest.param(
            [("first", torch.nn.Parameter(torch.tensor([1.0])))],
            {"first": torch.tensor([float("nan")])},
            "finite",
            id="nan-delta",
        ),
        pytest.param(
            [("first", torch.nn.Parameter(torch.tensor([1.0])))],
            {"first": torch.tensor([float("inf")])},
            "finite",
            id="infinite-delta",
        ),
        pytest.param(
            [
                ("first", torch.nn.Parameter(torch.tensor([1.0]))),
                ("second", torch.nn.Parameter(torch.tensor([float("nan")]))),
            ],
            {"first": torch.tensor([1.0])},
            "finite",
            id="nan-parameter",
        ),
        pytest.param(
            [
                ("first", torch.nn.Parameter(torch.tensor([1.0]))),
                ("second", torch.nn.Parameter(torch.tensor([float("inf")]))),
            ],
            {"first": torch.tensor([1.0])},
            "finite",
            id="infinite-parameter",
        ),
    ]


@pytest.mark.parametrize(("named_params", "deltas", "message"), validation_cases())
def test_invalid_perturbation_inputs_fail_without_partial_mutation(
    named_params,
    deltas,
    message,
):
    finite_originals = [
        parameter.detach().clone() if torch.isfinite(parameter).all() else None
        for _, parameter in named_params
    ]

    with pytest.raises((ValueError, RuntimeError), match=message):
        with TemporaryParameterPerturbation(named_params, deltas):
            pytest.fail("invalid inputs must fail before entering the body")

    for (_, parameter), original in zip(
        named_params,
        finite_originals,
        strict=True,
    ):
        if original is not None:
            assert torch.equal(parameter, original)


def test_enter_failure_after_an_earlier_mutation_restores_everything():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([torch.finfo(torch.float32).max]))
    named_params = [("first", first), ("second", second)]
    originals = clone_parameters(named_params)

    with pytest.raises(RuntimeError, match="non-finite"):
        with TemporaryParameterPerturbation(
            named_params,
            {
                "first": torch.tensor([1.0]),
                "second": torch.tensor([torch.finfo(torch.float32).max]),
            },
        ):
            pytest.fail("non-finite perturbation must not enter the body")

    assert torch.equal(first, originals["first"])
    assert torch.equal(second, originals["second"])


def test_perturbation_context_rejects_reentrant_enter_and_restores():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    context = TemporaryParameterPerturbation(
        [("weight", parameter)],
        {"weight": torch.tensor([1.0])},
    )

    with context:
        with pytest.raises(RuntimeError, match="already active"):
            context.__enter__()
        assert parameter.item() == 2.0

    assert parameter.item() == 1.0


def test_perturbation_context_cannot_be_reused_after_exit():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    context = TemporaryParameterPerturbation(
        [("weight", parameter)],
        {"weight": torch.tensor([1.0])},
    )

    with context:
        pass

    with pytest.raises(RuntimeError, match="cannot be reused"):
        with context:
            pass
    assert parameter.item() == 1.0


def test_perturbation_stats_ratio_guards_zero_requested_norm():
    assert PerturbationStats(requested_norm=2.0, effective_norm=0.5).ratio == 0.25
    assert PerturbationStats(requested_norm=0.0, effective_norm=0.0).ratio == 0.0
    assert PerturbationStats(requested_norm=0.0, effective_norm=1.0).ratio == 0.0
