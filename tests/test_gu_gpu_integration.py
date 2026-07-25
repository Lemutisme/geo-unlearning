import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from trainer.unlearn.optimizer_geometry import make_optimizer_geometry_adapter
from trainer.unlearn.base import UnlearnTrainer


pytestmark = pytest.mark.gpu


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def test_real_paged_adamw32_state_uses_finite_frozen_coordinates():
    require_cuda()
    bitsandbytes = pytest.importorskip("bitsandbytes")
    parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 4096, device="cuda"))
    optimizer = bitsandbytes.optim.AdamW(
        [parameter],
        lr=1e-3,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        optim_bits=32,
        is_paged=True,
    )
    parameter.grad = torch.linspace(1.0, 2.0, parameter.numel(), device="cuda")

    optimizer.step()

    state = optimizer.state[parameter]
    adapter = make_optimizer_geometry_adapter(optimizer)
    adapter.validate([("weight", parameter)])
    sqrt_denominator = adapter.sqrt_denominator(
        parameter,
        optimizer.param_groups[0],
    )
    assert state["step"] == 1
    assert state["state2"].dtype == torch.float32
    assert adapter.name == "paged_adamw_32bit"
    assert sqrt_denominator.dtype == torch.float32
    assert torch.isfinite(sqrt_denominator).all()
    assert (sqrt_denominator > 0).all()


def test_realized_delta_projection_accepts_cpu_second_moment(tmp_path):
    require_cuda()
    parameter = torch.nn.Parameter(torch.zeros(2, device="cuda"))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
    before = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(torch.tensor([1.0, -0.25], device="cuda"))
    optimizer.state[parameter]["exp_avg_sq"] = torch.ones(2, device="cpu")
    pending = (torch.tensor([1.0, 0.0], device="cuda"),)
    diagnostics_path = tmp_path / "gu_diagnostics.jsonl"
    diagnostics_path.write_text("")
    trainer = SimpleNamespace(
        _gu_parameter_snapshot=(before,),
        _gu_pending_history_covector=pending,
        _gu_constraints_used=(pending,),
        _gu_constraint_history=(),
        _gu_selected=(("weight", parameter),),
        _gu_diagnostics_path=diagnostics_path,
        gu_config={
            "projection_eps": 1.0e-6,
            "retain_filter": "first_order",
            "retain_history_rank": 1,
        },
        gu_projection_calls=0,
        gu_last_diagnostics=None,
    )

    UnlearnTrainer._gu_optimizer_step_post_hook(trainer, optimizer, (), {})

    applied = parameter.detach() - before
    assert applied[0].item() <= trainer.gu_config["projection_eps"]
    assert trainer.gu_projection_calls == 1
    assert trainer.gu_last_diagnostics["max_violation_after"] <= 1.0e-6


def make_llama(attention_implementation, state_dict=None):
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


def component_losses_and_gradients(model, forget_batch, retain_batch):
    forget_loss = model(**forget_batch).loss
    retain_loss = model(**retain_batch).loss
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    forget_grads = torch.autograd.grad(
        forget_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    retain_grads = torch.autograd.grad(
        retain_loss,
        parameters,
        allow_unused=True,
    )
    return forget_loss, retain_loss, forget_grads, retain_grads


def test_bf16_flash_attention_two_matches_eager_component_gradients():
    require_cuda()
    pytest.importorskip("flash_attn")
    torch.manual_seed(7)
    eager = make_llama("eager")
    flash = make_llama("flash_attention_2", copy.deepcopy(eager.state_dict()))
    generator = torch.Generator(device="cuda").manual_seed(19)

    def make_batch():
        input_ids = torch.randint(
            0,
            eager.config.vocab_size,
            (2, 16),
            generator=generator,
            device="cuda",
        )
        labels = input_ids.clone()
        labels[:, :3] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }

    forget_batch = make_batch()
    retain_batch = make_batch()

    eager_values = component_losses_and_gradients(eager, forget_batch, retain_batch)
    flash_values = component_losses_and_gradients(flash, forget_batch, retain_batch)

    for eager_loss, flash_loss in zip(eager_values[:2], flash_values[:2]):
        assert torch.isfinite(eager_loss)
        assert torch.isfinite(flash_loss)
        torch.testing.assert_close(eager_loss, flash_loss, rtol=2e-2, atol=2e-3)
    for eager_gradients, flash_gradients in zip(eager_values[2:], flash_values[2:]):
        for eager_gradient, flash_gradient in zip(eager_gradients, flash_gradients):
            if eager_gradient is None or flash_gradient is None:
                assert eager_gradient is flash_gradient
                continue
            assert torch.isfinite(eager_gradient).all()
            assert torch.isfinite(flash_gradient).all()
            torch.testing.assert_close(
                eager_gradient,
                flash_gradient,
                rtol=2e-2,
                atol=2e-3,
            )
