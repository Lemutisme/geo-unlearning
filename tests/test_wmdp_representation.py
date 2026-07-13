from types import SimpleNamespace

import pytest
import torch

from trainer.unlearn.wmdp_representation import (
    find_exact_module,
    forward_with_activation,
    masked_representation_mse,
    seeded_gaussian_noise,
    supervised_token_mask,
)


class TupleBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, hidden):
        return (self.proj(hidden), None)


class ToyRepresentationModel(torch.nn.Module):
    def __init__(self, *, fail_after_layers=False):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TupleBlock(4) for _ in range(8)])
        self.fail_after_layers = fail_after_layers

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        if self.fail_after_layers:
            raise RuntimeError("forced forward failure")
        return SimpleNamespace(logits=hidden)


def make_inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "labels": torch.tensor([[-100, 2, 3], [-100, 5, -100]]),
    }


def test_find_exact_module_and_capture_tuple_activation():
    model = ToyRepresentationModel()
    module = find_exact_module(model, "model.layers.7")

    activation, outputs = forward_with_activation(model, make_inputs(), module)

    assert activation is not None
    assert activation.shape == (2, 3, 4)
    assert outputs.logits is activation
    assert module._forward_hooks == {}


def test_find_exact_module_rejects_missing_name():
    with pytest.raises(ValueError, match="found 0"):
        find_exact_module(ToyRepresentationModel(), "model.layers.8")


def test_forward_hook_is_removed_when_model_forward_fails():
    model = ToyRepresentationModel(fail_after_layers=True)
    module = find_exact_module(model, "model.layers.7")

    with pytest.raises(RuntimeError, match="forced forward failure"):
        forward_with_activation(model, make_inputs(), module)

    assert module._forward_hooks == {}


def test_no_grad_capture_returns_detached_reference_activation():
    model = ToyRepresentationModel()
    module = find_exact_module(model, "model.layers.7")

    activation, _ = forward_with_activation(
        model,
        make_inputs(),
        module,
        no_grad=True,
    )

    assert activation.requires_grad is False


def test_masked_representation_mse_ignores_unsupervised_and_padding_tokens():
    inputs = make_inputs()
    mask = supervised_token_mask(inputs)
    actual = torch.zeros(2, 3, 2)
    target = torch.zeros_like(actual)
    target[0, 1] = 2.0
    target[0, 2] = 4.0
    target[1, 1] = 6.0
    target[0, 0] = 10_000.0
    target[1, 2] = 10_000.0

    loss = masked_representation_mse(actual, target, mask)

    assert mask.tolist() == [[False, True, True], [False, True, False]]
    assert loss.item() == pytest.approx(((4.0 + 16.0) / 2.0 + 36.0) / 2.0)


def test_supervised_token_mask_rejects_empty_mask():
    inputs = make_inputs()
    inputs["labels"].fill_(-100)

    with pytest.raises(RuntimeError, match="empty token mask"):
        supervised_token_mask(inputs)


def test_masked_representation_mse_rejects_shape_and_nonfinite_values():
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="incompatible"):
        masked_representation_mse(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 3, 3),
            mask,
        )
    with pytest.raises(RuntimeError, match="non-finite"):
        masked_representation_mse(
            torch.tensor([[[float("nan")], [0.0]]]),
            torch.zeros(1, 2, 1),
            mask,
        )


def test_seeded_gaussian_noise_is_reproducible_finite_and_nonzero():
    first_generator = torch.Generator(device="cpu").manual_seed(42)
    second_generator = torch.Generator(device="cpu").manual_seed(42)

    first = seeded_gaussian_noise(
        (2, 3, 4),
        std=0.01,
        generator=first_generator,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    second = seeded_gaussian_noise(
        (2, 3, 4),
        std=0.01,
        generator=second_generator,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert first.dtype is torch.bfloat16
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.count_nonzero(first).item() > 0


def test_seeded_gaussian_noise_is_invariant_to_batch_chunking():
    full_generator = torch.Generator(device="cpu").manual_seed(43)
    chunked_generator = torch.Generator(device="cpu").manual_seed(43)

    full = seeded_gaussian_noise(
        (4, 3, 2),
        std=0.01,
        generator=full_generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    chunked = torch.cat(
        [
            seeded_gaussian_noise(
                (1, 3, 2),
                std=0.01,
                generator=chunked_generator,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(4)
        ]
    )

    assert torch.equal(full, chunked)


@pytest.mark.parametrize("std", [0.0, -0.01, float("nan"), float("inf")])
def test_seeded_gaussian_noise_requires_positive_finite_std(std):
    with pytest.raises(ValueError, match="positive and finite"):
        seeded_gaussian_noise(
            (1,),
            std=std,
            generator=torch.Generator(device="cpu"),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
