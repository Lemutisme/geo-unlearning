from collections import Counter


EXPECTED_WMDP_PARAMETER_NAMES = (
    "model.layers.5.mlp.down_proj.weight",
    "model.layers.6.mlp.down_proj.weight",
    "model.layers.7.mlp.down_proj.weight",
)


def select_wmdp_parameters(model):
    named_parameters = dict(model.named_parameters())
    missing = [
        name for name in EXPECTED_WMDP_PARAMETER_NAMES if name not in named_parameters
    ]
    if missing:
        raise ValueError(f"WMDP selected parameters are missing: {missing}.")

    unexpected_trainable = [
        name
        for name, parameter in named_parameters.items()
        if parameter.requires_grad and name not in EXPECTED_WMDP_PARAMETER_NAMES
    ]
    if unexpected_trainable:
        raise ValueError(
            "WMDP model has unexpected trainable parameters: "
            f"{unexpected_trainable}."
        )

    selected = [
        (name, named_parameters[name]) for name in EXPECTED_WMDP_PARAMETER_NAMES
    ]
    frozen_selected = [name for name, parameter in selected if not parameter.requires_grad]
    if frozen_selected:
        raise ValueError(f"WMDP selected parameters are frozen: {frozen_selected}.")
    return selected


def configure_wmdp_trainable_parameters(model):
    named_parameters = dict(model.named_parameters())
    missing = [
        name for name in EXPECTED_WMDP_PARAMETER_NAMES if name not in named_parameters
    ]
    if missing:
        raise ValueError(f"WMDP selected parameters are missing: {missing}.")
    for name, parameter in named_parameters.items():
        parameter.requires_grad_(name in EXPECTED_WMDP_PARAMETER_NAMES)
    return select_wmdp_parameters(model)


def validate_wmdp_optimizer(optimizer, selected):
    selected = list(selected)
    selected_by_id = {id(parameter): name for name, parameter in selected}
    if len(selected_by_id) != len(selected):
        raise ValueError("WMDP selected parameter list contains duplicate tensors.")

    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    counts = Counter(id(parameter) for parameter in optimizer_parameters)
    duplicate = [
        selected_by_id.get(parameter_id, f"parameter@{parameter_id}")
        for parameter_id, count in counts.items()
        if count != 1
    ]
    if duplicate:
        raise ValueError(f"WMDP optimizer contains duplicate parameters: {duplicate}.")

    optimizer_ids = set(counts)
    selected_ids = set(selected_by_id)
    missing_ids = selected_ids - optimizer_ids
    extra_ids = optimizer_ids - selected_ids
    if missing_ids or extra_ids:
        missing = [selected_by_id[parameter_id] for parameter_id in missing_ids]
        extra = [f"parameter@{parameter_id}" for parameter_id in extra_ids]
        raise ValueError(
            f"WMDP optimizer parameter mismatch; missing={missing}, extra={extra}."
        )
