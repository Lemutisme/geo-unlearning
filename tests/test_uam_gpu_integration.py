import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    TrainerCallback,
    TrainingArguments,
)

from trainer.unlearn.optimizer_geometry import unwrap_optimizer
from trainer.unlearn.uam import UAMUnlearn


pytestmark = pytest.mark.gpu


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; BF16 UAM integration requires a GPU")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16 UAM integration")


def require_flash_attention_2():
    pytest.importorskip(
        "flash_attn",
        reason="flash_attn is unavailable; FlashAttention-2 integration cannot run",
    )
    major, _minor = torch.cuda.get_device_capability()
    if major < 8:
        pytest.skip(
            "flash_attn BF16 is incompatible with CUDA compute capability below 8.0"
        )


def make_tofu_like_dataset(num_examples=4, sequence_length=16, seed=701):
    generator = torch.Generator().manual_seed(seed)
    dataset = []
    for _ in range(num_examples):
        example = {}
        for component in ("forget", "retain"):
            input_ids = torch.randint(
                3,
                64,
                (sequence_length,),
                generator=generator,
            )
            labels = input_ids.clone()
            labels[:5] = -100
            example[component] = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            }
        dataset.append(example)
    return dataset


def nested_collator(features):
    return {
        component: {
            key: torch.stack([feature[component][key] for feature in features])
            for key in features[0][component]
        }
        for component in features[0]
    }


def make_bf16_llama(attention_implementation, state_dict=None):
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = attention_implementation
    model = LlamaForCausalLM(config)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.train()
    return model


@pytest.fixture
def uam_gpu_trainer(tmp_path):
    require_cuda()
    trainer_count = 0

    def factory(
        *,
        mode,
        attention_implementation="eager",
        optim="adamw_torch",
        max_steps=2,
        initial_state=None,
        seed=709,
    ):
        nonlocal trainer_count
        trainer_count += 1
        torch.manual_seed(seed)
        model = make_bf16_llama(attention_implementation, initial_state)
        output_dir = tmp_path / (
            f"run-{trainer_count}-{mode}-{attention_implementation}-{optim}"
        )
        args = TrainingArguments(
            output_dir=str(output_dir),
            report_to=[],
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=max_steps,
            learning_rate=1e-2,
            lr_scheduler_type="constant",
            optim=optim,
            adam_beta1=0.0,
            weight_decay=0.0,
            max_grad_norm=0.0,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            dataloader_num_workers=0,
            disable_tqdm=True,
            save_strategy="no",
            seed=seed,
            data_seed=seed,
        )
        geometric_config = SimpleNamespace(
            loss="simnpo",
            gu_enabled=True,
            gradient_surgery="gu",
            projection_eps=1e-12,
            trainable_params_regex=[".*"],
            component_buffer_device="parameter",
            diagnostics_path=None,
            actual_delta_mode="off",
            actual_delta_steps=[1, 2],
            actual_delta_sample_elements=1_000,
        )
        uam_config = SimpleNamespace(
            mode=mode,
            forget_signal="nll",
            perturbation_normalization="auto",
            rho=5e-2,
            reflection_gamma=2.0,
            residual_lambda=1.0,
            sign_tau=0.0,
            replay_device="cpu",
        )
        simnpo_config = SimpleNamespace(
            delta=0.0,
            beta=4.5,
            alpha=1.0,
            gamma=0.125,
            retain_loss_type="NLL",
        )
        trainer = UAMUnlearn(
            model=model,
            args=args,
            train_dataset=make_tofu_like_dataset(seed=seed + 1),
            data_collator=nested_collator,
            gamma=0.125,
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
        )
        return trainer, model

    return factory


class CapturePreOptimizerState(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.gradients = []
        self.gradient_dtypes = []
        self.diagnostics = []

    def on_pre_optimizer_step(self, _args, _state, control, **_kwargs):
        selected = self.trainer._selected_named_parameters()
        self.gradients.append(
            {
                name: parameter.grad.detach().float().cpu().clone()
                for name, parameter in selected
                if parameter.grad is not None
            }
        )
        self.gradient_dtypes.append(
            {
                name: parameter.grad.dtype
                for name, parameter in selected
                if parameter.grad is not None
            }
        )
        self.diagnostics.append(copy.deepcopy(self.trainer.last_uam_diagnostics))
        return control


class CapturePagedState(TrainerCallback):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state2_dtypes = []

    def on_optimizer_step(self, _args, _state, control, **_kwargs):
        initialized = [
            state["state2"].dtype
            for state in self.optimizer.state.values()
            if "state2" in state
        ]
        self.state2_dtypes.append(initialized)
        return control


def install_restoration_audit(trainer):
    replay_records = []
    real_replay = trainer._replay_perturbed_retain_gradients

    def audited_replay(named_params, deltas):
        named_params = list(named_params)
        snapshots = {
            name: parameter.detach().clone() for name, parameter in named_params
        }
        stats, replay_microsteps = real_replay(named_params, deltas)
        restored = all(
            torch.equal(parameter.detach(), snapshots[name])
            for name, parameter in named_params
        )
        replay_records.append(
            {
                "restored": restored,
                "requested_norm": stats.requested_norm,
                "effective_norm": stats.effective_norm,
                "replay_microsteps": replay_microsteps,
            }
        )
        return stats, replay_microsteps

    trainer._replay_perturbed_retain_gradients = audited_replay
    return replay_records


def snapshot_parameters(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def assert_all_parameters_finite_and_changed(model, initial_parameters):
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
        assert not torch.equal(parameter.detach().cpu(), initial_parameters[name]), name


def assert_diagnostics_finite(diagnostics):
    string_keys = {
        "record_type",
        "mode",
        "forget_signal",
        "perturbation_normalization",
        "optimizer_geometry",
        "component_buffer_device",
    }
    assert diagnostics
    assert all(
        math.isfinite(float(value))
        for key, value in diagnostics.items()
        if key not in string_keys
    )


def assert_diagnostics_coherent(diagnostics, mode, update_step):
    assert_diagnostics_finite(diagnostics)
    assert diagnostics["mode"] == mode
    assert diagnostics["update_step"] == update_step
    assert diagnostics["uam_calls"] == diagnostics["replay_calls"] == update_step
    assert diagnostics["replay_microsteps"] == 1
    assert diagnostics["requested_perturbation_norm"] > 0.0
    assert diagnostics["effective_perturbation_norm"] > 0.0
    assert diagnostics["effective_perturbation_ratio"] > 0.0
    assert diagnostics["effective_perturbation_ratio"] == pytest.approx(
        diagnostics["effective_perturbation_norm"]
        / diagnostics["requested_perturbation_norm"]
    )
    if mode == "uam":
        assert diagnostics["residual_tangent_norm"] == 0.0
        assert diagnostics["residual_normal_norm"] == 0.0
        assert diagnostics["relative_residual_orthogonality"] == 0.0
        assert diagnostics["residual_sign_gate_passed"] is False
        assert diagnostics["residual_gate_kept"] is False
        assert diagnostics["residual_orthogonality_safe"] is True
    else:
        if diagnostics["residual_gate_kept"]:
            assert diagnostics["residual_sign_gate_passed"] is True
            assert diagnostics["residual_orthogonality_safe"] is True


def assert_no_checkpoint_payload(output_dir):
    output_dir = Path(output_dir)
    forbidden_names = {
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    }
    forbidden = []
    if output_dir.exists():
        for path in output_dir.rglob("*"):
            payload_prefix = path.name.lower().startswith(
                ("optimizer", "scheduler", "rng_state", "trainer_state")
            )
            if (
                path.name.startswith("checkpoint-")
                or path.suffix in {".safetensors", ".bin", ".ckpt"}
                or path.name in forbidden_names
                or payload_prefix
            ):
                forbidden.append(path)
    assert forbidden == []


def assert_attention_implementation(model, expected):
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    assert model.config.attention_dropout == 0.0
    assert all(
        module.p == 0.0
        for module in model.modules()
        if isinstance(module, torch.nn.Dropout)
    )
    assert model.config._attn_implementation == expected
    attention_classes = {
        layer.self_attn.__class__.__name__ for layer in model.model.layers
    }
    if expected == "flash_attention_2":
        assert attention_classes == {"LlamaFlashAttention2"}
    else:
        assert attention_classes == {"LlamaAttention"}


@pytest.mark.parametrize("mode", ["uam", "uam_gu"])
def test_torch_adamw_bf16_runs_two_public_uam_updates(
    uam_gpu_trainer,
    mode,
):
    trainer, model = uam_gpu_trainer(mode=mode, max_steps=2)
    assert isinstance(trainer, UAMUnlearn)
    assert trainer.args.bf16 is True
    assert trainer.args.gradient_checkpointing is True
    assert trainer.args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert trainer.args.world_size == trainer.args.n_gpu == 1
    assert model.config.use_cache is False
    assert_attention_implementation(model, "eager")

    initial_parameters = snapshot_parameters(model)
    replay_records = install_restoration_audit(trainer)
    capture = CapturePreOptimizerState(trainer)
    trainer.add_callback(capture)

    trainer.train()

    assert trainer.uam_calls == trainer.replay_calls == trainer.state.global_step == 2
    assert len(replay_records) == len(capture.diagnostics) == 2
    assert all(
        set(gradient_dtypes.values()) == {torch.bfloat16}
        for gradient_dtypes in capture.gradient_dtypes
    )
    assert all(record["restored"] for record in replay_records)
    assert all(record["requested_norm"] > 0.0 for record in replay_records)
    assert all(record["effective_norm"] > 0.0 for record in replay_records)
    assert all(record["replay_microsteps"] == 1 for record in replay_records)
    for update_step, diagnostics in enumerate(capture.diagnostics, start=1):
        assert_diagnostics_coherent(diagnostics, mode, update_step)
    assert capture.diagnostics[0]["identity_fallback_parameters"] > 0
    assert capture.diagnostics[1]["identity_fallback_parameters"] == 0
    assert trainer.component_buffers.empty
    assert trainer.replay_buffer.empty
    assert trainer._uam_microsteps == 0
    assert_all_parameters_finite_and_changed(model, initial_parameters)
    assert_no_checkpoint_payload(trainer.args.output_dir)


def test_bf16_flash_attention_two_matches_eager_public_uam_update(
    uam_gpu_trainer,
):
    require_flash_attention_2()
    eager_trainer, eager_model = uam_gpu_trainer(
        mode="uam",
        attention_implementation="eager",
        max_steps=1,
    )
    initial_state = snapshot_parameters(eager_model)
    flash_trainer, flash_model = uam_gpu_trainer(
        mode="uam",
        attention_implementation="flash_attention_2",
        max_steps=1,
        initial_state=initial_state,
    )
    assert_attention_implementation(eager_model, "eager")
    assert_attention_implementation(flash_model, "flash_attention_2")

    eager_replays = install_restoration_audit(eager_trainer)
    flash_replays = install_restoration_audit(flash_trainer)
    eager_capture = CapturePreOptimizerState(eager_trainer)
    flash_capture = CapturePreOptimizerState(flash_trainer)
    eager_trainer.add_callback(eager_capture)
    flash_trainer.add_callback(flash_capture)

    eager_trainer.train()
    flash_trainer.train()

    for trainer, replays, capture in (
        (eager_trainer, eager_replays, eager_capture),
        (flash_trainer, flash_replays, flash_capture),
    ):
        assert trainer.uam_calls == trainer.replay_calls == 1
        assert len(replays) == len(capture.gradients) == 1
        assert set(capture.gradient_dtypes[0].values()) == {torch.bfloat16}
        assert replays[0]["restored"] is True
        assert replays[0]["effective_norm"] > 0.0
        assert_diagnostics_coherent(capture.diagnostics[0], "uam", 1)
        assert_no_checkpoint_payload(trainer.args.output_dir)

    eager_diagnostics = eager_capture.diagnostics[0]
    flash_diagnostics = flash_capture.diagnostics[0]
    for key in (
        "uam_coefficient",
        "effective_perturbation_norm",
        "effective_perturbation_ratio",
        "final_coordinate_norm",
    ):
        assert flash_diagnostics[key] == pytest.approx(
            eager_diagnostics[key],
            rel=8e-2,
            abs=2e-3,
        )

    assert eager_capture.gradients[0].keys() == flash_capture.gradients[0].keys()
    for name in eager_capture.gradients[0]:
        eager_gradient = eager_capture.gradients[0][name]
        flash_gradient = flash_capture.gradients[0][name]
        assert torch.isfinite(eager_gradient).all(), name
        assert torch.isfinite(flash_gradient).all(), name
        torch.testing.assert_close(
            flash_gradient,
            eager_gradient,
            rtol=1.5e-1,
            atol=3e-3,
        )

    for name, eager_parameter in eager_model.named_parameters():
        flash_parameter = flash_model.get_parameter(name)
        assert torch.isfinite(eager_parameter).all(), name
        assert torch.isfinite(flash_parameter).all(), name
        initial_parameter = initial_state[name].float().to(eager_parameter.device)
        eager_update = eager_parameter.float() - initial_parameter
        flash_update = flash_parameter.float() - initial_parameter
        update_delta = flash_update - eager_update
        close_fraction = (
            torch.isclose(
                flash_parameter.float(),
                eager_parameter.float(),
                rtol=2e-2,
                atol=5e-4,
            )
            .float()
            .mean()
        )
        relative_update_error = torch.linalg.vector_norm(
            update_delta
        ) / torch.linalg.vector_norm(eager_update).clamp_min(1e-12)
        update_cosine = torch.nn.functional.cosine_similarity(
            eager_update.reshape(1, -1),
            flash_update.reshape(1, -1),
        )
        assert close_fraction.item() >= 0.99, name
        assert relative_update_error.item() < 0.13, name
        assert update_cosine.item() > 0.99, name


def test_paged_adamw_32bit_bf16_runs_two_public_uam_gu_updates(
    uam_gpu_trainer,
):
    pytest.importorskip(
        "bitsandbytes",
        reason="bitsandbytes is unavailable; PagedAdamW32 integration cannot run",
    )
    trainer, model = uam_gpu_trainer(
        mode="uam_gu",
        optim="paged_adamw_32bit",
        max_steps=2,
    )
    assert trainer.args.bf16 is True
    assert_attention_implementation(model, "eager")
    initial_parameters = snapshot_parameters(model)
    trainer.create_optimizer()
    optimizer = unwrap_optimizer(trainer.optimizer)
    assert optimizer.__class__.__module__.startswith("bitsandbytes.optim.adamw")
    assert optimizer.__class__.__name__ == "AdamW"
    assert optimizer.is_paged is True
    assert optimizer.args.optim_bits == 32
    assert optimizer.param_groups[0]["betas"][0] == 0.0
    assert optimizer.param_groups[0]["weight_decay"] == 0.0

    prefetch_calls = 0
    real_prefetch = optimizer.prefetch_state

    def record_prefetch(parameter):
        nonlocal prefetch_calls
        prefetch_calls += 1
        return real_prefetch(parameter)

    optimizer.prefetch_state = record_prefetch
    replay_records = install_restoration_audit(trainer)
    pre_optimizer_capture = CapturePreOptimizerState(trainer)
    paged_state_capture = CapturePagedState(optimizer)
    trainer.add_callback(pre_optimizer_capture)
    trainer.add_callback(paged_state_capture)

    trainer.train()

    assert trainer.uam_calls == trainer.replay_calls == trainer.state.global_step == 2
    assert len(replay_records) == 2
    assert all(record["restored"] for record in replay_records)
    assert all(
        set(gradient_dtypes.values()) == {torch.bfloat16}
        for gradient_dtypes in pre_optimizer_capture.gradient_dtypes
    )
    assert len(paged_state_capture.state2_dtypes) == 2
    assert paged_state_capture.state2_dtypes[0]
    assert all(dtype == torch.float32 for dtype in paged_state_capture.state2_dtypes[0])
    assert prefetch_calls > 0
    for update_step, diagnostics in enumerate(
        pre_optimizer_capture.diagnostics,
        start=1,
    ):
        assert_diagnostics_coherent(diagnostics, "uam_gu", update_step)
        assert diagnostics["optimizer_geometry"] == "paged_adamw_32bit"
    assert pre_optimizer_capture.diagnostics[0]["identity_fallback_parameters"] > 0
    assert pre_optimizer_capture.diagnostics[1]["identity_fallback_parameters"] == 0
    assert trainer.component_buffers.empty
    assert trainer.replay_buffer.empty
    assert trainer._uam_microsteps == 0
    assert_all_parameters_finite_and_changed(model, initial_parameters)
    assert_no_checkpoint_payload(trainer.args.output_dir)
