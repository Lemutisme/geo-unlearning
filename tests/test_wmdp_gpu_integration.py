import gc
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

from tests.helpers import nested_collator
from trainer.unlearn.optimizer_geometry import unwrap_optimizer
from trainer.unlearn.wmdp_rmu import WMDPRMUUnlearn
from trainer.unlearn.wmdp_selection import EXPECTED_WMDP_PARAMETER_NAMES
from trainer.unlearn.wmdp_uam import WMDPUAMUnlearn


pytestmark = pytest.mark.gpu


def require_wmdp_gpu_stack():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unavailable")
    pytest.importorskip("flash_attn")
    pytest.importorskip("bitsandbytes")


@pytest.fixture(autouse=True)
def cleanup_cuda_after_wmdp_test():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def make_llama():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "flash_attention_2"
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    model.enable_input_require_grads()
    return model


def make_dataset(seed=301, example_count=4, sequence_length=16):
    generator = torch.Generator().manual_seed(seed)
    dataset = []
    for _ in range(example_count):
        example = {}
        for component in ("forget", "retain"):
            input_ids = torch.randint(
                3,
                128,
                (sequence_length,),
                generator=generator,
            )
            labels = input_ids.clone()
            labels[:4] = -100
            example[component] = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            }
        dataset.append(example)
    return dataset


def base_geometric_config(*, loss, gu_enabled):
    return SimpleNamespace(
        loss=loss,
        gu_enabled=gu_enabled,
        gradient_surgery="gu",
        projection_eps=1e-12,
        trainable_params_regex=[
            r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"
        ],
        component_buffer_device="parameter",
        diagnostics_path=None,
        actual_delta_mode="off",
        actual_delta_steps=[1, 2],
        actual_delta_sample_elements=1_000,
    )


def make_trainer(tmp_path, arm):
    model = make_llama()
    args = TrainingArguments(
        output_dir=str(tmp_path / arm),
        report_to=[],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=2,
        learning_rate=5e-5,
        optim="paged_adamw_32bit",
        adam_beta1=0.0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=0,
        disable_tqdm=True,
        save_strategy="no",
        seed=42,
        data_seed=42,
    )
    common = dict(
        model=model,
        args=args,
        train_dataset=make_dataset(),
        data_collator=nested_collator,
        simnpo_config=None,
        npo_config=None,
        dpo_config=None,
        undial_config=None,
        wga_config=None,
        satimp_config=None,
    )
    if arm in {"rmu", "rmu_gu"}:
        trainer = WMDPRMUUnlearn(
            **common,
            gamma=1.0,
            alpha=100.0,
            retain_loss_type="EMBED_DIFF",
            geometric_config=base_geometric_config(
                loss="rmu",
                gu_enabled=arm == "rmu_gu",
            ),
            rmu_config=SimpleNamespace(
                module_name="model.layers.7",
                steering_coeff=20.0,
                seed=42,
                require_paged_adamw=True,
            ),
        )
    else:
        trainer = WMDPUAMUnlearn(
            **common,
            gamma=1.0,
            alpha=1.0,
            retain_loss_type="NLL",
            geometric_config=base_geometric_config(
                loss="wmdp_uam",
                gu_enabled=True,
            ),
            uam_config=SimpleNamespace(
                mode=arm,
                forget_signal="representation",
                reflection_geometry="euclidean",
                perturbation_normalization="fixed_loss",
                rho=5e-5,
                reflection_gamma=2.0,
                residual_lambda=1.0,
                sign_tau=0.0,
                replay_device="cpu",
                module_name="model.layers.7",
                noise_std=0.01,
                seed=42,
                require_paged_adamw=True,
            ),
        )
    return trainer, model


class CaptureSelectedGradients(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.records = []

    def on_pre_optimizer_step(self, _args, _state, control, **_kwargs):
        self.records.append(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in self.trainer._selected_named_parameters()
            }
        )
        return control


def install_restoration_audit(trainer):
    if not isinstance(trainer, WMDPUAMUnlearn):
        return []
    real_replay = trainer._replay_perturbed_retain_gradients
    records = []

    def audited_replay(named_params, deltas):
        before = {
            name: parameter.detach().clone() for name, parameter in named_params
        }
        result = real_replay(named_params, deltas)
        records.append(
            all(
                torch.equal(parameter.detach(), before[name])
                for name, parameter in named_params
            )
        )
        return result

    trainer._replay_perturbed_retain_gradients = audited_replay
    return records


def assert_no_checkpoint_payload(output_dir):
    forbidden = []
    for path in Path(output_dir).rglob("*"):
        if (
            path.name.startswith("checkpoint-")
            or path.suffix in {".bin", ".pt", ".pth", ".ckpt", ".safetensors"}
            or path.name.startswith(("optimizer", "scheduler", "rng_state"))
        ):
            forbidden.append(path)
    assert forbidden == []


@pytest.mark.parametrize("arm", ["rmu", "rmu_gu", "uam", "uam_gu"])
def test_wmdp_arm_runs_two_real_bf16_flash_paged_updates(tmp_path, arm):
    require_wmdp_gpu_stack()
    trainer, model = make_trainer(tmp_path, arm)
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    capture = CaptureSelectedGradients(trainer)
    trainer.add_callback(capture)
    restoration_records = install_restoration_audit(trainer)
    trainer.create_optimizer()
    optimizer = unwrap_optimizer(trainer.optimizer)

    assert optimizer.__class__.__module__.startswith("bitsandbytes.optim.adamw")
    assert optimizer.is_paged is True
    assert optimizer.args.optim_bits == 32
    assert model.config._attn_implementation == "flash_attention_2"
    assert model.config.use_cache is False

    trainer.train()

    assert trainer.state.global_step == 2
    assert len(capture.records) == 2
    assert all(
        gradient.dtype is torch.bfloat16
        and torch.isfinite(gradient).all().item()
        and torch.count_nonzero(gradient).item() > 0
        for record in capture.records
        for gradient in record.values()
    )
    if arm == "rmu":
        assert trainer.gu_projection_calls == 0
    elif arm == "rmu_gu":
        assert trainer.gu_projection_calls == trainer.state.global_step
    else:
        assert trainer.uam_calls == trainer.replay_calls == trainer.state.global_step
        assert restoration_records == [True, True]

    selected = set(EXPECTED_WMDP_PARAMETER_NAMES)
    for name, parameter in model.named_parameters():
        changed = not torch.equal(parameter.detach().cpu(), initial[name])
        assert changed is (name in selected), name
    assert_no_checkpoint_payload(trainer.args.output_dir)
