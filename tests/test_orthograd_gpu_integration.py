import copy
import json

import pytest
import torch

from tests.helpers import make_unlearn_batch
from tests.test_geometric_adam import unbatch
from tests.test_gu_gpu_integration import make_llama, require_cuda
from tests.test_orthograd_trainer import make_orthograd_trainer
from trainer.unlearn.optimizer_geometry import unwrap_optimizer


pytestmark = pytest.mark.gpu


def make_dataset(batch_size=4, seed=0):
    return unbatch(
        make_unlearn_batch(
            batch_size=batch_size,
            sequence_length=16,
            vocab_size=64,
            seed=seed,
        )
    )


def test_bf16_flash_matches_eager_public_orthograd_update(tmp_path):
    require_cuda()
    pytest.importorskip("flash_attn")
    torch.manual_seed(101)
    eager_model = make_llama("eager")
    flash_model = make_llama(
        "flash_attention_2",
        copy.deepcopy(eager_model.state_dict()),
    )
    dataset = make_dataset(batch_size=2, seed=103)
    eager, _ = make_orthograd_trainer(
        tmp_path / "eager",
        model=eager_model,
        train_dataset=dataset,
        per_device_train_batch_size=2,
        max_steps=1,
        use_cpu=False,
        bf16=True,
        resource_profile_path=str(tmp_path / "eager.json"),
    )
    flash, _ = make_orthograd_trainer(
        tmp_path / "flash",
        model=flash_model,
        train_dataset=dataset,
        per_device_train_batch_size=2,
        max_steps=1,
        use_cpu=False,
        bf16=True,
        resource_profile_path=str(tmp_path / "flash.json"),
    )

    eager.train()
    flash.train()

    assert eager.orthograd_calls == flash.orthograd_calls == 1
    assert eager.last_orthograd_diagnostics["maximum_relative_residual"] < 1e-4
    assert flash.last_orthograd_diagnostics["maximum_relative_residual"] < 1e-4
    for eager_parameter, flash_parameter in zip(
        eager.model.parameters(),
        flash.model.parameters(),
    ):
        torch.testing.assert_close(
            eager_parameter,
            flash_parameter,
            rtol=2e-2,
            atol=2e-3,
        )
    for path in (tmp_path / "eager.json", tmp_path / "flash.json"):
        profile = json.loads(path.read_text())
        assert profile["cuda_max_allocated_bytes"] > 0
        assert profile["cuda_max_reserved_bytes"] >= profile["cuda_max_allocated_bytes"]


def test_paged_adamw32_bf16_runs_two_public_orthograd_updates(tmp_path):
    require_cuda()
    pytest.importorskip("bitsandbytes")
    torch.manual_seed(107)
    model = make_llama("eager")
    dataset = make_dataset(batch_size=4, seed=109)
    trainer, _ = make_orthograd_trainer(
        tmp_path,
        model=model,
        train_dataset=dataset,
        per_device_train_batch_size=2,
        max_steps=2,
        use_cpu=False,
        bf16=True,
        optim="paged_adamw_32bit",
        resource_profile_path=str(tmp_path / "resource.json"),
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]

    trainer.train()

    optimizer = unwrap_optimizer(trainer.optimizer)
    assert trainer.orthograd_calls == trainer.state.global_step == 2
    assert trainer.retain_basis.empty
    assert trainer.component_buffers.empty
    assert any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, model.parameters())
    )
    initialized_states = [
        state for state in optimizer.state.values() if state.get("state2") is not None
    ]
    assert initialized_states
    assert all(state["state2"].dtype == torch.float32 for state in initialized_states)
    profile = json.loads((tmp_path / "resource.json").read_text())
    assert profile["update_count"] == 2
    assert profile["cuda_max_allocated_bytes"] > 0
