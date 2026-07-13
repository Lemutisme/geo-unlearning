import copy
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path
from transformers import PretrainedConfig, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

from trainer import TRAINER_REGISTRY
from trainer.unlearn.wmdp_selection import EXPECTED_WMDP_PARAMETER_NAMES
from trainer.unlearn.wmdp_uam import WMDPUAMUnlearn
from tests.helpers import nested_collator


class ToyWMDPLayer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, hidden):
        return (hidden + torch.tanh(self.mlp.down_proj(hidden)),)


class ToyWMDPCausalLM(torch.nn.Module):
    main_input_name = "input_ids"

    def __init__(self, layer_count=8, width=6, vocab_size=32):
        super().__init__()
        self.config = PretrainedConfig(vocab_size=vocab_size)
        self.config.use_cache = False
        self.embed = torch.nn.Embedding(vocab_size, width)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [ToyWMDPLayer(width) for _ in range(layer_count)]
        )
        self.lm_head = torch.nn.Linear(width, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return CausalLMOutput(logits=self.lm_head(hidden))


def make_batch(seed=1, batch_size=2, sequence_length=5):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        1,
        31,
        (batch_size, sequence_length),
        generator=generator,
    )
    labels = input_ids.clone()
    labels[:, :2] = -100
    attention_mask = torch.ones_like(input_ids)
    attention_mask[-1, -1] = 0
    labels[-1, -1] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def unbatch_wmdp_dataset(batch):
    batch_size = batch["forget"]["input_ids"].shape[0]
    return [
        {
            component: {
                name: tensor[index].clone()
                for name, tensor in component_batch.items()
            }
            for component, component_batch in batch.items()
        }
        for index in range(batch_size)
    ]


def make_wmdp_uam_trainer(tmp_path, *, model=None, **overrides):
    model = ToyWMDPCausalLM() if model is None else model
    train_dataset = overrides.pop("train_dataset", None)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=overrides.pop("per_device_train_batch_size", 1),
        gradient_accumulation_steps=overrides.pop(
            "gradient_accumulation_steps",
            1,
        ),
        max_steps=overrides.pop("max_steps", 1),
        learning_rate=5e-5,
        optim="adamw_torch",
        adam_beta1=0.0,
        weight_decay=0.0,
        max_grad_norm=0.0,
        remove_unused_columns=False,
        disable_tqdm=True,
        save_strategy="no",
    )
    geometric_config = SimpleNamespace(
        loss="simnpo",
        gu_enabled=True,
        gradient_surgery="gu",
        projection_eps=1e-12,
        trainable_params_regex=[
            r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"
        ],
        component_buffer_device="parameter",
        diagnostics_path=None,
        actual_delta_mode="off",
        actual_delta_steps=[1, 80],
        actual_delta_sample_elements=1_000,
    )
    uam_config = SimpleNamespace(
        mode=overrides.pop("mode", "uam"),
        forget_signal="representation",
        reflection_geometry=overrides.pop("reflection_geometry", "euclidean"),
        perturbation_normalization="fixed_loss",
        rho=overrides.pop("rho", 5e-5),
        reflection_gamma=overrides.pop("reflection_gamma", 2.0),
        residual_lambda=1.0,
        sign_tau=0.0,
        replay_device="cpu",
        module_name=overrides.pop("module_name", "model.layers.7"),
        noise_std=overrides.pop("noise_std", 0.01),
        seed=overrides.pop("seed", 42),
        require_paged_adamw=overrides.pop("require_paged_adamw", False),
    )
    simnpo_config = SimpleNamespace(
        delta=0.0,
        beta=4.5,
        alpha=1.0,
        gamma=1.0,
        retain_loss_type="NLL",
    )
    assert not overrides, overrides
    trainer = WMDPUAMUnlearn(
        model=model,
        args=args,
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        geometric_config=geometric_config,
        uam_config=uam_config,
        simnpo_config=simnpo_config,
        npo_config=None,
        dpo_config=None,
        undial_config=None,
        wga_config=None,
        satimp_config=None,
        train_dataset=train_dataset,
        data_collator=nested_collator if train_dataset is not None else None,
    )
    return trainer, model


def test_wmdp_uam_freezes_reference_and_selects_exact_parameters(tmp_path):
    trainer, model = make_wmdp_uam_trainer(tmp_path)

    assert trainer.ref_model is not None
    assert trainer.ref_model.training is False
    assert all(not parameter.requires_grad for parameter in trainer.ref_model.parameters())
    assert tuple(name for name, _ in trainer._selected_named_parameters()) == (
        EXPECTED_WMDP_PARAMETER_NAMES
    )
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == set(EXPECTED_WMDP_PARAMETER_NAMES)


def test_wmdp_uam_is_registered_and_yaml_resolves_w2_contract():
    assert TRAINER_REGISTRY["WMDPUAMUnlearn"] is WMDPUAMUnlearn
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="unlearn",
            overrides=["trainer=WMDPUAMUnlearn"],
        )

    trainer = config.trainer
    assert trainer.handler == "WMDPUAMUnlearn"
    assert trainer.args.per_device_train_batch_size == 1
    assert trainer.args.gradient_accumulation_steps == 4
    assert trainer.args.max_steps == 80
    assert trainer.args.learning_rate == pytest.approx(5e-5)
    assert trainer.args.bf16 is True
    assert trainer.args.fp16 is False
    assert trainer.args.optim == "paged_adamw_32bit"
    assert trainer.args.adam_beta1 == 0.0
    assert trainer.args.weight_decay == 0.0
    assert trainer.args.save_strategy == "no"
    assert trainer.method_args.geometric_config.loss == "wmdp_uam"
    assert trainer.method_args.geometric_config.actual_delta_mode == "full"
    assert trainer.method_args.geometric_config.actual_delta_steps == [1, 80]
    assert trainer.method_args.uam_config.forget_signal == "representation"
    assert trainer.method_args.uam_config.reflection_geometry == "euclidean"
    assert trainer.method_args.uam_config.rho == pytest.approx(5e-5)
    assert trainer.method_args.uam_config.reflection_gamma == 2.0
    assert trainer.method_args.uam_config.noise_std == 0.01
    assert trainer.method_args.uam_config.require_paged_adamw is True


def test_retain_representation_loss_is_zero_at_initialization(tmp_path):
    trainer, model = make_wmdp_uam_trainer(tmp_path)

    loss = trainer.compute_retain_loss(model, make_batch(seed=2))

    assert loss.item() == pytest.approx(0.0, abs=0.0)
    assert loss.requires_grad is True


def test_seeded_forget_noise_produces_nonzero_reproducible_selected_gradient(
    tmp_path,
):
    torch.manual_seed(3)
    base_model = ToyWMDPCausalLM()
    first, first_model = make_wmdp_uam_trainer(
        tmp_path / "first",
        model=copy.deepcopy(base_model),
        seed=77,
    )
    second, second_model = make_wmdp_uam_trainer(
        tmp_path / "second",
        model=copy.deepcopy(base_model),
        seed=77,
    )
    batch = make_batch(seed=4)

    first_loss, _ = first.compute_uam_forget_signal(first_model, batch)
    second_loss, _ = second.compute_uam_forget_signal(second_model, batch)
    first_params = [parameter for _, parameter in first._selected_named_parameters()]
    second_params = [parameter for _, parameter in second._selected_named_parameters()]
    first_grads = torch.autograd.grad(first_loss, first_params)
    second_grads = torch.autograd.grad(second_loss, second_params)

    assert first_loss.item() > 0.0
    assert any(torch.count_nonzero(gradient).item() for gradient in first_grads)
    for first_gradient, second_gradient in zip(
        first_grads,
        second_grads,
        strict=True,
    ):
        torch.testing.assert_close(first_gradient, second_gradient, rtol=0, atol=0)


def test_retain_objective_does_not_advance_forget_noise_generator(tmp_path):
    trainer, model = make_wmdp_uam_trainer(tmp_path)
    before = trainer.wmdp_noise_generator.get_state().clone()

    trainer.compute_retain_loss(model, make_batch(seed=5))

    assert torch.equal(trainer.wmdp_noise_generator.get_state(), before)


def test_optimizer_contains_only_the_three_selected_parameters(tmp_path):
    trainer, _ = make_wmdp_uam_trainer(tmp_path)

    trainer.create_optimizer()

    optimizer_parameters = [
        parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    ]
    selected = trainer._selected_named_parameters()
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter) for _, parameter in selected
    }


@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("reflection_geometry", "optimizer", "Euclidean reflection"),
        ("rho", 1e-4, "rho=5e-5"),
        ("reflection_gamma", 1.0, "reflection_gamma=2"),
        ("noise_std", 0.02, "noise_std=0.01"),
        ("module_name", "model.layers.6", "model.layers.7"),
    ],
)
def test_wmdp_runtime_rejects_nonpaper_objective_settings(
    tmp_path,
    override,
    value,
    message,
):
    trainer, _ = make_wmdp_uam_trainer(tmp_path, **{override: value})
    trainer.create_optimizer()

    with pytest.raises((ValueError, NotImplementedError), match=message):
        trainer._validate_uam_runtime()


@pytest.mark.parametrize("mode", ["uam", "uam_gu"])
def test_wmdp_uam_gas_matches_effective_batch_update(tmp_path, mode):
    torch.manual_seed(101)
    base = ToyWMDPCausalLM()
    full_model = copy.deepcopy(base)
    accumulated_model = copy.deepcopy(base)
    dataset = unbatch_wmdp_dataset(
        {
            "forget": make_batch(seed=102, batch_size=4),
            "retain": make_batch(seed=103, batch_size=4),
        }
    )
    full, _ = make_wmdp_uam_trainer(
        tmp_path / "full",
        model=full_model,
        mode=mode,
        train_dataset=dataset,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
    )
    accumulated, _ = make_wmdp_uam_trainer(
        tmp_path / "accumulated",
        model=accumulated_model,
        mode=mode,
        train_dataset=dataset,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
    )

    full.train()
    accumulated.train()

    assert full.uam_calls == full.replay_calls == 1
    assert accumulated.uam_calls == accumulated.replay_calls == 1
    assert full.last_uam_diagnostics["replay_microsteps"] == 1
    assert accumulated.last_uam_diagnostics["replay_microsteps"] == 4
    for name in EXPECTED_WMDP_PARAMETER_NAMES:
        torch.testing.assert_close(
            full_model.get_parameter(name),
            accumulated_model.get_parameter(name),
            rtol=1e-5,
            atol=1e-6,
        )


def test_wmdp_uam_replay_failure_restores_parameters_and_clears_window(
    tmp_path,
    monkeypatch,
):
    trainer, model = make_wmdp_uam_trainer(tmp_path)
    trainer.create_optimizer()
    trainer._validate_uam_runtime()
    batch = trainer._prepare_inputs(
        {
            "forget": make_batch(seed=104),
            "retain": make_batch(seed=105),
        }
    )
    trainer._collect_uam_microstep(model, batch)
    named_params = trainer._selected_named_parameters()
    originals = {name: parameter.detach().clone() for name, parameter in named_params}

    def fail_retain_replay(_model, _inputs):
        raise RuntimeError("forced WMDP retain replay failure")

    monkeypatch.setattr(trainer, "compute_retain_loss", fail_retain_replay)

    with pytest.raises(RuntimeError, match="forced WMDP retain replay failure"):
        trainer._finalize_uam_gradients(named_params)

    for name, parameter in named_params:
        assert torch.equal(parameter, originals[name])
        assert parameter.grad is None
    assert trainer.component_buffers.empty
    assert trainer.replay_buffer.empty
    assert trainer._uam_microsteps == 0
    assert trainer.uam_calls == trainer.replay_calls == 0
