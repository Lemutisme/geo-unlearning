import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "scripts/pcgrad_smoke_arm.sh"
MATRIX = ROOT / "scripts/pcgrad_smoke_matrix.sh"


def script_text(path):
    assert path.exists(), f"Missing launcher: {path}"
    return path.read_text()


def test_arm_launcher_activates_environment_and_disables_all_saves():
    text = script_text(ARM)

    assert 'source "${conda_base}/etc/profile.d/conda.sh"' in text
    assert "conda activate unlearning" in text
    assert "save_model_after_train=false" in text
    assert "trainer.args.save_strategy=no" in text
    assert "trainer.args.save_only_model=false" in text


def test_production_arm_contract_is_paged_bf16_flash_and_nonreentrant():
    text = script_text(ARM)

    assert "paged_adamw_32bit" in text
    assert "trainer.args.adam_beta1=0.0" in text
    assert "trainer.args.weight_decay=0.0" in text
    assert "torch_dtype=bfloat16" in text
    assert '"model.model_args.torch_dtype=${torch_dtype}"' in text
    assert "attention_implementation=flash_attention_2" in text
    assert '"model.model_args.attn_implementation=${attention_implementation}"' in text
    assert "bf16=true" in text
    assert '"trainer.args.bf16=${bf16}"' in text
    assert "trainer.args.fp16=false" in text
    assert "trainer.args.gradient_checkpointing_kwargs.use_reentrant=false" in text


def test_arm_forces_model_and_dataset_cache_off_the_workspace_fuse_mount():
    text = script_text(ARM)

    assert "export HF_HOME=${PCGRAD_HF_HOME:-/root/.cache/huggingface}" in text
    assert "local_root=${PCGRAD_LOCAL_ROOT:-/tmp/pcgrad_smoke}" in text
    assert '"paths.output_dir=${local_arm_dir}"' in text
    assert "persist_artifacts" in text
    assert 'cp "${summary_path}" "${persistent_summary}"' in text


def test_muse_uses_cpu_buffers_and_all_checkpoint_payloads_are_audited():
    text = script_text(ARM)

    assert "muse_news|muse_books" in text
    assert "component_buffer_device=cpu" in text
    for payload in (
        "*.safetensors",
        "pytorch_model*.bin",
        "training_args.bin",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "*.ckpt",
    ):
        assert payload in text


def test_muse_default_eval_uses_supported_retain_knowledge_metric():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(
            config_name="unlearn.yaml",
            overrides=[
                "experiment=unlearn/muse/default",
                "task_name=pcgrad_muse_config_test",
            ],
        )

    metrics = config.eval.muse.metrics
    assert "retain_knowmem_ROUGE" in metrics
    assert "retain_extraction_strength" not in metrics


def test_arm_launcher_has_all_dataset_method_and_system_modes():
    text = script_text(ARM)

    for dataset in ("tofu01", "tofu05", "tofu10", "muse_news", "muse_books"):
        assert dataset in text
    for method in ("control", "gu", "pcgrad"):
        assert method in text
    for system_mode in ("production", "torch_flash", "torch_eager_fp32"):
        assert system_mode in text


def test_matrix_schedules_fifteen_algorithm_arms_and_two_system_arms():
    text = script_text(MATRIX)

    assert "RUN_MANIFEST.tsv" in text
    assert "tofu01:pcgrad:torch_flash" in text
    assert "tofu01:pcgrad:torch_eager_fp32" in text
    assert 'for dataset in "${datasets[@]}"' in text
    assert 'for method in "${methods[@]}"' in text
    assert "active_pid[0]" in text
    assert "active_pid[1]" in text
    assert "failure=1" in text


def test_arm_overrides_accelerate_config_with_the_assigned_physical_gpu():
    text = script_text(ARM)

    assert "--gpu_ids" in text
    assert '"${gpu}"' in text


def test_launchers_have_valid_bash_syntax():
    for path in (ARM, MATRIX):
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_arm_bootstraps_conda_without_an_inherited_shell_function(tmp_path):
    environment = os.environ.copy()
    environment["PATH"] = "/usr/bin:/bin"
    environment["CONDA_EXE"] = "/root/miniconda3/bin/conda"
    result = subprocess.run(
        [str(ARM), "invalid", "control", "0", "test", "production"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Unsupported dataset: invalid" in result.stderr
    assert "conda: command not found" not in result.stderr


def test_checkpoint_audit_expression_executes_and_detects_payloads(tmp_path):
    clean = tmp_path / "clean"
    dirty = tmp_path / "dirty"
    clean.mkdir()
    dirty.mkdir()

    clean_result = subprocess.run(
        [str(ARM), "--audit-only", str(clean)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert clean_result.returncode == 0, clean_result.stderr

    (dirty / "optimizer.pt").write_bytes(b"checkpoint")
    dirty_result = subprocess.run(
        [str(ARM), "--audit-only", str(dirty)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert dirty_result.returncode == 1
    assert "Unexpected checkpoint payload" in dirty_result.stderr
