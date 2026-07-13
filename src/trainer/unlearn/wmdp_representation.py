import math

import torch


def find_exact_module(model, name):
    matches = [
        module
        for module_name, module in model.named_modules()
        if module_name == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one module named {name!r}, found {len(matches)}."
        )
    return matches[0]


def _activation_tensor(output):
    activation = output[0] if isinstance(output, tuple) else output
    if not isinstance(activation, torch.Tensor):
        raise TypeError("Captured WMDP activation must be a tensor.")
    if activation.ndim != 3:
        raise ValueError(
            "Captured WMDP activation must have shape [batch, sequence, hidden]."
        )
    if not torch.isfinite(activation).all().item():
        raise RuntimeError("Captured WMDP activation is non-finite.")
    return activation


def forward_with_activation(model, inputs, module, *, no_grad=False):
    captured = []

    def capture(_module, _args, output):
        captured.append(_activation_tensor(output))

    handle = module.register_forward_hook(capture)
    try:
        with torch.set_grad_enabled(not no_grad):
            outputs = model(**inputs)
    finally:
        handle.remove()

    if len(captured) != 1:
        raise RuntimeError(
            f"Expected one captured WMDP activation, found {len(captured)}."
        )
    return captured[0], outputs


def supervised_token_mask(inputs):
    try:
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
    except KeyError as error:
        raise KeyError(
            "WMDP representation inputs require attention_mask and labels."
        ) from error
    if attention_mask.shape != labels.shape:
        raise ValueError("WMDP attention mask and labels must have matching shapes.")
    mask = attention_mask.bool() & labels.ne(-100)
    if not mask.any().item():
        raise RuntimeError("WMDP representation objective has an empty token mask.")
    return mask


def representation_model_inputs(inputs):
    required = ("input_ids", "attention_mask", "labels")
    missing = [name for name in required if name not in inputs]
    if missing:
        raise KeyError(f"WMDP representation batch is missing: {missing}.")
    return {name: inputs[name] for name in required}


def forward_representation_pair(
    model,
    reference_model,
    inputs,
    model_module,
    reference_module,
):
    model_inputs = representation_model_inputs(inputs)
    activation, outputs = forward_with_activation(
        model,
        model_inputs,
        model_module,
    )
    reference, _ = forward_with_activation(
        reference_model,
        model_inputs,
        reference_module,
        no_grad=True,
    )
    if reference.device != activation.device:
        reference = reference.to(activation.device)
    mask = supervised_token_mask(model_inputs)
    return activation, reference, outputs, mask


def masked_representation_mse(actual, target, mask):
    if (
        actual.shape != target.shape
        or actual.ndim != 3
        or actual.shape[:2] != mask.shape
    ):
        raise ValueError(
            "WMDP activation, target, and mask shapes are incompatible."
        )
    per_token = (actual.float() - target.float()).square().mean(dim=-1)
    mask = mask.bool()
    token_counts = mask.sum(dim=-1)
    if (token_counts == 0).any().item():
        raise RuntimeError(
            "WMDP representation objective has an empty per-example token mask."
        )
    per_example = (per_token * mask).sum(dim=-1) / token_counts.to(per_token)
    loss = per_example.mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError("WMDP representation loss is non-finite.")
    return loss


def seeded_gaussian_noise(shape, *, std, generator, device, dtype):
    if isinstance(std, bool) or not isinstance(std, (int, float)):
        raise ValueError("WMDP Gaussian noise std must be positive and finite.")
    std = float(std)
    if std <= 0.0 or not math.isfinite(std):
        raise ValueError("WMDP Gaussian noise std must be positive and finite.")
    if not isinstance(generator, torch.Generator):
        raise TypeError("WMDP Gaussian noise requires a torch.Generator.")

    shape = tuple(shape)
    if not shape or shape[0] <= 0:
        raise ValueError("WMDP Gaussian noise requires a positive batch dimension.")

    noise = torch.stack(
        [
            torch.randn(
                shape[1:],
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
            for _ in range(shape[0])
        ],
        dim=0,
    )
    noise.mul_(std)
    noise = noise.to(device=device, dtype=dtype)
    if not torch.isfinite(noise).all().item():
        raise RuntimeError("WMDP Gaussian noise is non-finite.")
    return noise
