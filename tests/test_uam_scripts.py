import importlib.util
import json
import math
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "scripts/uam_smoke_arm.sh"
MATRIX = ROOT / "scripts/uam_smoke_matrix.sh"
ANALYZER = ROOT / "scripts/analyze_uam_smoke.py"
METHODS = ("uam_nll", "uam_simnpo", "uam_gu_nll", "uam_gu_simnpo")
MANIFEST_HEADER = (
    "pid",
    "gpu",
    "method",
    "start_utc",
    "end_utc",
    "exit_code",
    "command",
)
GEOMETRY_REQUIRED = {
    "update_step",
    "mode",
    "forget_signal",
    "perturbation_normalization",
    "rho",
    "requested_perturbation_norm",
    "effective_perturbation_norm",
    "effective_perturbation_ratio",
    "forget_norm",
    "retain_norm",
    "perturbed_retain_norm",
    "uam_coefficient",
    "residual_tangent_norm",
    "residual_normal_norm",
    "relative_residual_orthogonality",
    "residual_forget_gate_dot",
    "residual_gate_kept",
    "predicted_forget_directional_derivative",
    "predicted_retain_directional_derivative",
    "identity_fallback_parameters",
    "replay_microsteps",
    "uam_calls",
    "replay_calls",
}


def script_text(path):
    assert path.is_file(), f"Missing script: {path}"
    return path.read_text()


def test_arm_launcher_contract_and_exact_method_mapping():
    text = script_text(ARM)

    assert "Usage: $0 <method> <gpu> <timestamp>" in text
    assert "if [[ $# -eq 2 && $1 == --audit-only ]]" in text
    mappings = {
        "uam_nll": "mode=uam; signal=nll; normalization=fixed_loss ;;\n",
        "uam_simnpo": "mode=uam; signal=simnpo; normalization=fixed_loss ;;\n",
        "uam_gu_nll": "mode=uam_gu; signal=nll; normalization=metric_trust ;;\n",
        "uam_gu_simnpo": (
            "mode=uam_gu; signal=simnpo; normalization=metric_trust ;;\n"
        ),
    }
    for method, assignment in mappings.items():
        assert f"    {method})\n        {assignment}" in text


def test_arm_activates_conda_and_uses_local_staging_and_hf_cache():
    text = script_text(ARM)

    assert 'source "${conda_base}/etc/profile.d/conda.sh"' in text
    assert "conda activate unlearning" in text
    assert 'export CUDA_VISIBLE_DEVICES="${gpu}"' in text
    assert "export HF_HOME=/root/.cache/huggingface" in text
    assert "local_root=${UAM_LOCAL_ROOT:-/tmp/uam_smoke}" in text
    assert 'local_arm_dir="${local_root}/${timestamp}/${method}"' in text
    assert 'matrix_root="saves/exp/UAM_SMOKE/${timestamp}"' in text
    assert 'arm_dir="${matrix_root}/${method}"' in text
    assert '"paths.output_dir=${local_arm_dir}"' in text


def test_arm_uses_approved_tofu_production_stack_and_smoke_rho_override():
    text = script_text(ARM)

    for token in (
        "accelerate launch",
        "--config_file configs/accelerate/gu_single_gpu.yaml",
        '--gpu_ids "${gpu}"',
        "src/train.py",
        "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/default",
        "trainer=UAMUnlearn",
        "model=Llama-3.2-1B-Instruct",
        "base_model=open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "model.tokenizer_args.pretrained_model_name_or_path=${base_model}",
        "model.model_args.torch_dtype=bfloat16",
        "model.model_args.attn_implementation=flash_attention_2",
        "retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json",
        "forget_split=forget01",
        "retain_split=retain99",
        "holdout_split=holdout01",
        "trainer.args.per_device_train_batch_size=4",
        "trainer.args.gradient_accumulation_steps=8",
        "trainer.args.max_steps=10",
        "trainer.args.learning_rate=${smoke_learning_rate}",
        "trainer.args.optim=paged_adamw_32bit",
        "trainer.args.adam_beta1=0.0",
        "trainer.args.weight_decay=0.0",
        "trainer.args.fp16=false",
        "trainer.args.bf16=true",
        "trainer.args.bf16_full_eval=true",
        "trainer.args.gradient_checkpointing=true",
        "trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
        "trainer.args.logging_steps=1",
        "trainer.args.report_to=none",
        "trainer.args.do_train=true",
        "trainer.args.do_eval=true",
        "trainer.args.eval_strategy=no",
        "trainer.method_args.geometric_config.loss=simnpo",
        "trainer.method_args.uam_config.mode=${mode}",
        "trainer.method_args.uam_config.forget_signal=${signal}",
        "trainer.method_args.uam_config.perturbation_normalization=${normalization}",
        "trainer.method_args.uam_config.rho=${smoke_rho}",
        "smoke_rho=${UAM_SMOKE_RHO:-0.05}",
        "smoke_learning_rate=${UAM_SMOKE_LEARNING_RATE:-1e-5}",
        "trainer.args.warmup_epochs=0",
        "trainer.method_args.geometric_config.actual_delta_mode=full",
        "trainer.method_args.geometric_config.actual_delta_steps=[1,10]",
    ):
        assert token in text


def test_arm_disables_model_and_checkpoint_persistence():
    text = script_text(ARM)

    for token in (
        "save_model_after_train=false",
        "trainer.args.save_strategy=no",
        "trainer.args.save_only_model=false",
    ):
        assert token in text
    assert "cp -r" not in text
    assert "cp -R" not in text
    assert "checkpoint-10/evals" not in text


def test_arm_persists_only_approved_artifacts_and_audits_both_roots():
    text = script_text(ARM)

    assert 'find "${local_arm_dir}" -type f -name TOFU_SUMMARY.json' in text
    assert "Expected exactly one TOFU_SUMMARY.json" in text
    assert 'persistent_summary="${arm_dir}/evals/TOFU_SUMMARY.json"' in text
    for artifact in (
        "run.log",
        "UAMUnlearn.log",
        "uam_diagnostics.jsonl",
        "uam_diagnostics.summary.json",
        ".hydra/config.yaml",
        "evals/TOFU_SUMMARY.json",
    ):
        assert artifact in text
    assert 'audit_checkpoint_payloads "${local_arm_dir}"' in text
    assert 'audit_checkpoint_payloads "${arm_dir}"' in text


def test_arm_audit_covers_checkpoint_directories_and_payload_patterns():
    text = script_text(ARM)

    for token in (
        "checkpoint-*",
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "trainer_state.json",
        "optimizer*",
        "scheduler*",
        "rng_state*",
    ):
        assert token in text


def test_launchers_have_valid_bash_syntax():
    for path in (ARM, MATRIX):
        result = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("relative_path", "is_directory"),
    [
        ("nested/checkpoint-10", True),
        ("model.safetensors", False),
        ("nested/adapter_model.bin", False),
        ("nested/anything.bin", False),
        ("optimizer.pt", False),
        ("scheduler.pt", False),
        ("rng_state.pth", False),
        ("weights.pt", False),
        ("weights.pth", False),
        ("snapshot.ckpt", False),
        ("trainer_state.json", False),
        ("model_state.json", False),
        ("training_args.json", False),
    ],
)
def test_audit_only_rejects_every_checkpoint_representative(
    tmp_path, relative_path, is_directory
):
    clean = tmp_path / "clean"
    clean.mkdir()
    clean_result = subprocess.run(
        ["bash", str(ARM), "--audit-only", str(clean)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert clean_result.returncode == 0, clean_result.stderr

    dirty = tmp_path / "dirty"
    payload = dirty / relative_path
    if is_directory:
        payload.mkdir(parents=True)
    else:
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"checkpoint")

    dirty_result = subprocess.run(
        ["bash", str(ARM), "--audit-only", str(dirty)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert dirty_result.returncode == 1
    assert "Unexpected checkpoint payload" in dirty_result.stderr


@pytest.mark.parametrize(
    "link_name",
    ["optimizer.pt", "checkpoint-10", "ordinary-linked-directory"],
)
def test_audit_only_rejects_every_symlink_without_following_it(tmp_path, link_name):
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    if link_name == "ordinary-linked-directory":
        target = tmp_path / "outside-directory"
        target.mkdir()
        (target / "model.bin").write_bytes(b"outside")
        (audit_root / link_name).symlink_to(target, target_is_directory=True)
    elif link_name == "checkpoint-10":
        target = tmp_path / "outside-checkpoint"
        target.mkdir()
        (audit_root / link_name).symlink_to(target, target_is_directory=True)
    else:
        target = tmp_path / "outside-file"
        target.write_bytes(b"outside")
        (audit_root / link_name).symlink_to(target)

    result = subprocess.run(
        ["bash", str(ARM), "--audit-only", str(audit_root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Unexpected symlink" in result.stderr


def test_matrix_contract_has_exact_methods_header_and_commands():
    text = script_text(MATRIX)

    assert 'matrix_root="saves/exp/UAM_SMOKE/${timestamp}"' in text
    assert "arms=(uam_nll uam_simnpo uam_gu_nll uam_gu_simnpo)" in text
    assert (
        "printf 'pid\\tgpu\\tmethod\\tstart_utc\\tend_utc\\texit_code\\tcommand\\n'"
        in text
    )
    assert (
        'local command="bash scripts/uam_smoke_arm.sh ${method} ${slot} ${timestamp}"'
        in text
    )
    assert (
        'bash scripts/uam_smoke_arm.sh "${method}" "${slot}" "${timestamp}" &' in text
    )
    assert "active_pid[0]" in text
    assert "active_pid[1]" in text
    assert "failure=1" in text
    assert "wait -n" not in text


def _write_fake_conda(tmp_path):
    conda_root = tmp_path / "fake-conda"
    profile = conda_root / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text("conda() { return 0; }\n")
    executable = tmp_path / "conda"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == info && $2 == --base ]]; then\n"
        f"    printf '%s\\n' '{conda_root}'\n"
        "    exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    executable.chmod(0o755)
    return executable


def _write_conda_spy(tmp_path):
    marker = tmp_path / "conda-spy-called"
    conda_root = tmp_path / "conda-spy-root"
    profile = conda_root / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text("conda() { return 0; }\n")
    executable = tmp_path / "conda-spy"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'called\\n' >> '{marker}'\n"
        "if [[ $1 == info && $2 == --base ]]; then\n"
        f"    printf '%s\\n' '{conda_root}'\n"
        "    exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    executable.chmod(0o755)
    return executable, marker


def _write_fake_accelerate(tmp_path):
    executable = tmp_path / "bin/accelerate"
    executable.parent.mkdir()
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "root=${FAKE_LOCAL_ARM:?}\n"
        "if [[ -n ${FAKE_ACCELERATE_ARGV:-} ]]; then\n"
        '    printf \'%s\\n\' "$@" > "${FAKE_ACCELERATE_ARGV}"\n'
        "fi\n"
        'mkdir -p "${root}/.hydra" "${root}/checkpoint-10/evals"\n'
        "printf 'config: fake\\n' > \"${root}/.hydra/config.yaml\"\n"
        "printf '{\"metric\": 1.0}\\n' > "
        '"${root}/checkpoint-10/evals/TOFU_SUMMARY.json"\n'
        "printf '{\"detail\": true}\\n' > "
        '"${root}/checkpoint-10/evals/details.json"\n'
        "if [[ ${FAKE_CHECKPOINT_PAYLOAD:-0} == 1 ]]; then\n"
        "    printf 'payload\\n' > \"${root}/checkpoint-10/optimizer.pt\"\n"
        "fi\n"
        "if [[ ${FAKE_CHECKPOINT_SYMLINK:-0} == 1 ]]; then\n"
        '    ln -s "${root}/.hydra/config.yaml" '
        '"${root}/checkpoint-10/ordinary-link"\n'
        "fi\n"
        'printf \'{"record_type": "uam_geometry"}\\n\' > '
        '"${root}/uam_diagnostics.jsonl"\n'
        "printf '{\"surgery_count\": 1}\\n' > "
        '"${root}/uam_diagnostics.summary.json"\n'
        "printf 'trainer log\\n' > \"${root}/UAMUnlearn.log\"\n"
        "if [[ -n ${FAKE_PERSISTENT_JUNK:-} ]]; then\n"
        '    mkdir -p "$(dirname "${FAKE_PERSISTENT_JUNK}")"\n'
        "    printf 'junk\\n' > \"${FAKE_PERSISTENT_JUNK}\"\n"
        "fi\n"
    )
    executable.chmod(0o755)
    return executable


def _process_running(pid):
    status = Path(f"/proc/{pid}/stat")
    if not status.exists():
        return False
    try:
        return status.read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _cleanup_pids(paths):
    pids = []
    for path in paths:
        if path.is_file():
            pids.append(int(path.read_text()))
    for pid in pids:
        if _process_running(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    _wait_until(lambda: not any(_process_running(pid) for pid in pids), timeout=1.0)
    for pid in pids:
        if _process_running(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    ("learning_rate_override", "expected_learning_rate"),
    [(None, "1e-5"), ("2e-4", "2e-4")],
)
def test_arm_normalizes_eval_container_and_persists_only_allowlist(
    tmp_path,
    learning_rate_override,
    expected_learning_rate,
):
    local_arm = tmp_path / "local/stamp/uam_nll"
    matrix_root = tmp_path / "saves/exp/UAM_SMOKE/stamp"
    persistent_arm = matrix_root / "uam_nll"
    matrix_root.mkdir(parents=True)
    (matrix_root / "RUN_MANIFEST.tsv").write_text("manifest sentinel\n")
    sibling = matrix_root / "uam_simnpo/sentinel.txt"
    sibling.parent.mkdir()
    sibling.write_text("sibling sentinel\n")
    fake_accelerate = _write_fake_accelerate(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)
    argv_log = tmp_path / "accelerate.argv"
    environment["FAKE_ACCELERATE_ARGV"] = str(argv_log)
    if learning_rate_override is not None:
        environment["UAM_SMOKE_LEARNING_RATE"] = learning_rate_override

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert not list(local_arm.rglob("checkpoint-*"))
    assert not list(persistent_arm.rglob("checkpoint-*"))
    allowed = {
        ".hydra/config.yaml",
        "UAMUnlearn.log",
        "evals/TOFU_SUMMARY.json",
        "run.log",
        "uam_diagnostics.jsonl",
        "uam_diagnostics.summary.json",
    }
    assert {
        str(path.relative_to(persistent_arm))
        for path in persistent_arm.rglob("*")
        if path.is_file()
    } == allowed
    assert {
        str(path.relative_to(local_arm))
        for path in local_arm.rglob("*")
        if path.is_file()
    } == allowed
    assert (matrix_root / "RUN_MANIFEST.tsv").read_text() == "manifest sentinel\n"
    assert sibling.read_text() == "sibling sentinel\n"
    argv = argv_log.read_text().splitlines()
    assert argv[:5] == [
        "launch",
        "--config_file",
        "configs/accelerate/gu_single_gpu.yaml",
        "--gpu_ids",
        "0",
    ]
    for exact_override in (
        "trainer=UAMUnlearn",
        "forget_split=forget01",
        "retain_split=retain99",
        "holdout_split=holdout01",
        "trainer.args.max_steps=10",
        f"trainer.args.learning_rate={expected_learning_rate}",
        "trainer.args.warmup_epochs=0",
        "trainer.args.optim=paged_adamw_32bit",
        "trainer.method_args.geometric_config.actual_delta_mode=full",
        "trainer.method_args.geometric_config.actual_delta_steps=[1,10]",
        "trainer.method_args.uam_config.mode=uam",
        "trainer.method_args.uam_config.forget_signal=nll",
        "trainer.method_args.uam_config.perturbation_normalization=fixed_loss",
        "trainer.method_args.uam_config.rho=0.05",
    ):
        assert exact_override in argv


@pytest.mark.parametrize("gpu", ["-1", "gpu0", "1.5", "2"])
def test_arm_rejects_invalid_gpu_before_accelerate(tmp_path, gpu):
    matrix_root = tmp_path / "saves/exp/UAM_SMOKE/stamp"
    matrix_root.mkdir(parents=True)
    local_arm = tmp_path / "local/stamp/uam_nll"
    fake_accelerate = _write_fake_accelerate(tmp_path)
    argv_log = tmp_path / "accelerate.argv"
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)
    environment["FAKE_ACCELERATE_ARGV"] = str(argv_log)

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", gpu, "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "Invalid GPU" in result.stderr
    assert not argv_log.exists()


@pytest.mark.parametrize("rho", ["0", "-0.01", "nan", "inf", "true"])
def test_arm_rejects_nonpositive_nonfinite_or_boolean_rho_before_accelerate(
    tmp_path,
    rho,
):
    matrix_root = tmp_path / "saves/exp/UAM_SMOKE/stamp"
    matrix_root.mkdir(parents=True)
    local_arm = tmp_path / "local/stamp/uam_nll"
    fake_accelerate = _write_fake_accelerate(tmp_path)
    argv_log = tmp_path / "accelerate.argv"
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)
    environment["FAKE_ACCELERATE_ARGV"] = str(argv_log)
    environment["UAM_SMOKE_RHO"] = rho

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "Invalid UAM_SMOKE_RHO" in result.stderr
    assert not argv_log.exists()


@pytest.mark.parametrize("learning_rate", ["0", "-0.0001", "nan", "inf", "true"])
def test_arm_rejects_nonpositive_nonfinite_or_boolean_learning_rate_before_accelerate(
    tmp_path,
    learning_rate,
):
    matrix_root = tmp_path / "saves/exp/UAM_SMOKE/stamp"
    matrix_root.mkdir(parents=True)
    local_arm = tmp_path / "local/stamp/uam_nll"
    fake_accelerate = _write_fake_accelerate(tmp_path)
    argv_log = tmp_path / "accelerate.argv"
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)
    environment["FAKE_ACCELERATE_ARGV"] = str(argv_log)
    environment["UAM_SMOKE_LEARNING_RATE"] = learning_rate

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "Invalid UAM_SMOKE_LEARNING_RATE" in result.stderr
    assert not argv_log.exists()


def test_matrix_rejects_reused_timestamp_without_modifying_old_results(tmp_path):
    matrix_root = tmp_path / "saves/exp/UAM_SMOKE/reused"
    old_arm = matrix_root / "uam_nll/sentinel.txt"
    old_arm.parent.mkdir(parents=True)
    old_arm.write_text("old arm\n")
    manifest = matrix_root / "RUN_MANIFEST.tsv"
    manifest.write_text("old manifest\n")
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))

    result = subprocess.run(
        ["bash", str(MATRIX), "reused"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert manifest.read_text() == "old manifest\n"
    assert old_arm.read_text() == "old arm\n"


def test_arm_rejects_existing_persistent_arm_and_preserves_siblings(tmp_path):
    matrix_root = tmp_path / "saves/exp/UAM_SMOKE/stamp"
    persistent_arm = matrix_root / "uam_nll"
    sentinel = persistent_arm / "sentinel.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("old arm\n")
    sibling = matrix_root / "uam_simnpo/sibling.txt"
    sibling.parent.mkdir()
    sibling.write_text("sibling\n")
    local_arm = tmp_path / "local/stamp/uam_nll"
    fake_accelerate = _write_fake_accelerate(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert sentinel.read_text() == "old arm\n"
    assert sibling.read_text() == "sibling\n"


def test_arm_rejects_existing_local_staging_without_publishing(tmp_path):
    matrix_root = tmp_path / "saves/exp/UAM_SMOKE/stamp"
    matrix_root.mkdir(parents=True)
    local_arm = tmp_path / "local/stamp/uam_nll"
    local_arm.mkdir(parents=True)
    sentinel = local_arm / "sentinel.txt"
    sentinel.write_text("old local\n")
    fake_accelerate = _write_fake_accelerate(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "Local staging already exists" in result.stderr
    assert sentinel.read_text() == "old local\n"
    assert not (matrix_root / "uam_nll").exists()


def test_arm_copy_failure_leaves_no_final_or_temporary_persistent_arm(tmp_path):
    matrix_root = tmp_path / "saves/exp/UAM_SMOKE/stamp"
    matrix_root.mkdir(parents=True)
    local_arm = tmp_path / "local/stamp/uam_nll"
    fake_accelerate = _write_fake_accelerate(tmp_path)
    fake_cp = fake_accelerate.parent / "cp"
    fake_cp.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == */.hydra/config.yaml ]]; then exit 73; fi\n"
        'exec /bin/cp "$@"\n'
    )
    fake_cp.chmod(0o755)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert not (matrix_root / "uam_nll").exists()
    assert not list(matrix_root.glob(".uam_nll.tmp.*"))


def test_matrix_rejects_symlinked_persistent_ancestor(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "saves").symlink_to(outside, target_is_directory=True)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))

    result = subprocess.run(
        ["bash", str(MATRIX), "symlinked"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "symlink" in result.stderr.lower()
    assert not (outside / "exp/UAM_SMOKE/symlinked").exists()


def test_arm_rejects_symlinked_persistent_ancestor_before_training(tmp_path):
    outside = tmp_path / "outside"
    matrix_root = outside / "exp/UAM_SMOKE/stamp"
    matrix_root.mkdir(parents=True)
    (tmp_path / "saves").symlink_to(outside, target_is_directory=True)
    local_arm = tmp_path / "local/stamp/uam_nll"
    fake_accelerate = _write_fake_accelerate(tmp_path)
    argv_log = tmp_path / "accelerate.argv"
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)
    environment["FAKE_ACCELERATE_ARGV"] = str(argv_log)

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "symlink" in result.stderr.lower()
    assert not argv_log.exists()
    assert not (matrix_root / "uam_nll").exists()


def test_arm_rejects_relative_local_root_before_conda_or_writes(tmp_path):
    (tmp_path / "saves/exp/UAM_SMOKE/stamp").mkdir(parents=True)
    conda_spy, marker = _write_conda_spy(tmp_path)
    fake_accelerate = _write_fake_accelerate(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(conda_spy)
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = "relative-local"
    environment["FAKE_LOCAL_ARM"] = str(tmp_path / "relative-local/stamp/uam_nll")

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "UAM_LOCAL_ROOT" in result.stderr
    assert "absolute" in result.stderr.lower()
    assert not marker.exists()
    assert not (tmp_path / "relative-local").exists()


@pytest.mark.parametrize("nested_suffix", ["", "nested"])
def test_arm_rejects_symlink_component_in_local_root_before_conda_or_writes(
    tmp_path,
    nested_suffix,
):
    (tmp_path / "saves/exp/UAM_SMOKE/stamp").mkdir(parents=True)
    outside = tmp_path / "outside-local"
    outside.mkdir()
    linked = tmp_path / "linked-local"
    linked.symlink_to(outside, target_is_directory=True)
    local_root = linked / nested_suffix if nested_suffix else linked
    local_arm = outside / nested_suffix / "stamp/uam_nll"
    conda_spy, marker = _write_conda_spy(tmp_path)
    fake_accelerate = _write_fake_accelerate(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(conda_spy)
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(local_root)
    environment["FAKE_LOCAL_ARM"] = str(local_arm)

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "symlink" in result.stderr.lower()
    assert not marker.exists()
    assert not (outside / nested_suffix / "stamp").exists()


def test_arm_rejects_unsafe_timestamp_before_environment_bootstrap(tmp_path):
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(tmp_path / "must-not-run-conda")

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "../escape"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "Unsafe timestamp" in result.stderr


def test_arm_rejects_checkpoint_payload_before_deleting_eval_container(tmp_path):
    (tmp_path / "saves/exp/UAM_SMOKE/stamp").mkdir(parents=True)
    local_arm = tmp_path / "local/stamp/uam_nll"
    fake_accelerate = _write_fake_accelerate(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)
    environment["FAKE_CHECKPOINT_PAYLOAD"] = "1"

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    payload = local_arm / "checkpoint-10/optimizer.pt"
    assert result.returncode == 1
    assert "Unexpected checkpoint payload" in result.stderr
    assert payload.is_file()
    assert not (local_arm / "evals/TOFU_SUMMARY.json").exists()


def test_arm_rejects_symlink_before_deleting_eval_container(tmp_path):
    (tmp_path / "saves/exp/UAM_SMOKE/stamp").mkdir(parents=True)
    local_arm = tmp_path / "local/stamp/uam_nll"
    fake_accelerate = _write_fake_accelerate(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)
    environment["FAKE_CHECKPOINT_SYMLINK"] = "1"

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    link = local_arm / "checkpoint-10/ordinary-link"
    assert result.returncode == 1
    assert "Unexpected symlink" in result.stderr
    assert link.is_symlink()
    assert not (local_arm / "evals/TOFU_SUMMARY.json").exists()


def test_arm_deletes_newline_checkpoint_as_one_path_without_touching_sentinel(
    tmp_path,
):
    (tmp_path / "saves/exp/UAM_SMOKE/stamp").mkdir(parents=True)
    local_arm = tmp_path / "local/stamp/uam_nll"
    checkpoint_name = "checkpoint-10\n--preserve-root"
    checkpoint_dir = local_arm / checkpoint_name
    sentinel = tmp_path / "--preserve-root/sentinel.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("keep me\n")
    executable = tmp_path / "bin/accelerate"
    executable.parent.mkdir()
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "root=${FAKE_LOCAL_ARM:?}\n"
        "checkpoint_name=${FAKE_CHECKPOINT_NAME:?}\n"
        'mkdir -p "${root}/.hydra" "${root}/evals" '
        '"${root}/${checkpoint_name}"\n'
        "printf 'config: fake\\n' > \"${root}/.hydra/config.yaml\"\n"
        "printf '{\"metric\": 1.0}\\n' > "
        '"${root}/evals/TOFU_SUMMARY.json"\n'
        "printf '{\"detail\": true}\\n' > "
        '"${root}/${checkpoint_name}/details.json"\n'
        'printf \'{"record_type": "uam_geometry"}\\n\' > '
        '"${root}/uam_diagnostics.jsonl"\n'
        "printf '{\"surgery_count\": 1}\\n' > "
        '"${root}/uam_diagnostics.summary.json"\n'
    )
    executable.chmod(0o755)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{executable.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)
    environment["FAKE_CHECKPOINT_NAME"] = checkpoint_name

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert not checkpoint_dir.exists()
    assert sentinel.read_text() == "keep me\n"
    assert (tmp_path / "saves/exp/UAM_SMOKE/stamp/uam_nll").is_dir()


def test_arm_final_allowlist_rejects_junk_created_after_initial_cleanup(tmp_path):
    (tmp_path / "saves/exp/UAM_SMOKE/stamp").mkdir(parents=True)
    local_arm = tmp_path / "local/stamp/uam_nll"
    persistent_arm = tmp_path / "saves/exp/UAM_SMOKE/stamp/uam_nll"
    junk = persistent_arm / "junk.json"
    fake_accelerate = _write_fake_accelerate(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{fake_accelerate.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["FAKE_LOCAL_ARM"] = str(local_arm)
    environment["FAKE_PERSISTENT_JUNK"] = str(junk)

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "stamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 1
    assert "Persistent arm already exists" in result.stderr
    assert junk.is_file()


def test_matrix_fail_fast_waits_active_and_schedules_no_new_arm(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(MATRIX, scripts / MATRIX.name)
    fake_arm = scripts / ARM.name
    fake_arm.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$1\" >> launched.txt\n"
        "if [[ $1 == uam_nll ]]; then exit 7; fi\n"
        "sleep 0.1\n"
        "exit 0\n"
    )
    fake_arm.chmod(0o755)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))

    result = subprocess.run(
        ["bash", f"scripts/{MATRIX.name}", "teststamp"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    launched = (tmp_path / "launched.txt").read_text().splitlines()
    assert set(launched) == {"uam_nll", "uam_simnpo"}
    manifest = (
        (tmp_path / "saves/exp/UAM_SMOKE/teststamp/RUN_MANIFEST.tsv")
        .read_text()
        .splitlines()
    )
    assert manifest[0] == ("pid\tgpu\tmethod\tstart_utc\tend_utc\texit_code\tcommand")
    assert len(manifest) == 3
    rows = [line.split("\t") for line in manifest[1:]]
    assert {row[2] for row in rows} == {"uam_nll", "uam_simnpo"}
    assert {int(row[5]) for row in rows} == {0, 7}
    assert all(
        row[6] == f"bash scripts/uam_smoke_arm.sh {row[2]} {row[1]} teststamp"
        for row in rows
    )


def test_matrix_rejects_unsafe_timestamp_before_conda_or_path_writes(tmp_path):
    conda_spy, marker = _write_conda_spy(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(conda_spy)
    escape_name = f"uam-matrix-traversal-{tmp_path.name}"
    unsafe_timestamp = f"../../{escape_name}"
    escaped_manifest = (
        tmp_path / "saves/exp/UAM_SMOKE" / unsafe_timestamp / "RUN_MANIFEST.tsv"
    ).resolve()
    escaped_root = escaped_manifest.parent
    assert escaped_root.parent == (tmp_path / "saves").resolve()
    if escaped_root.exists():
        shutil.rmtree(escaped_root)

    try:
        result = subprocess.run(
            ["bash", str(MATRIX), unsafe_timestamp],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 2
        assert "Invalid timestamp" in result.stderr
        assert not marker.exists()
        assert not escaped_manifest.exists()
    finally:
        if escaped_root.exists():
            shutil.rmtree(escaped_root)


def test_matrix_observes_fast_slot_one_failure_before_scheduling_third_arm(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(MATRIX, scripts / MATRIX.name)
    fake_arm = scripts / ARM.name
    fake_arm.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$1\" >> launched.txt\n"
        "case $1 in\n"
        "    uam_nll) sleep 0.4; exit 0 ;;\n"
        "    uam_simnpo) sleep 0.03; exit 9 ;;\n"
        "    *) exit 0 ;;\n"
        "esac\n"
    )
    fake_arm.chmod(0o755)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))

    result = subprocess.run(
        ["bash", f"scripts/{MATRIX.name}", "race"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert set((tmp_path / "launched.txt").read_text().splitlines()) == {
        "uam_nll",
        "uam_simnpo",
    }
    manifest = (
        (tmp_path / "saves/exp/UAM_SMOKE/race/RUN_MANIFEST.tsv")
        .read_text()
        .splitlines()
    )
    assert len(manifest) == 3
    rows = [line.split("\t") for line in manifest[1:]]
    assert {row[2] for row in rows} == {"uam_nll", "uam_simnpo"}
    assert {int(row[5]) for row in rows} == {0, 9}


def test_matrix_success_records_each_of_four_methods_once(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(MATRIX, scripts / MATRIX.name)
    fake_arm = scripts / ARM.name
    fake_arm.write_text(
        "#!/usr/bin/env bash\n" "printf '%s\\n' \"$1\" >> launched.txt\n" "sleep 0.02\n"
    )
    fake_arm.chmod(0o755)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))

    result = subprocess.run(
        ["bash", f"scripts/{MATRIX.name}", "success"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert set((tmp_path / "launched.txt").read_text().splitlines()) == set(METHODS)
    manifest = (
        (tmp_path / "saves/exp/UAM_SMOKE/success/RUN_MANIFEST.tsv")
        .read_text()
        .splitlines()
    )
    assert len(manifest) == 5
    rows = [line.split("\t") for line in manifest[1:]]
    assert {row[2] for row in rows} == set(METHODS)
    assert all(int(row[5]) == 0 for row in rows)


def test_matrix_term_reaps_active_arms_and_their_descendants(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(MATRIX, scripts / MATRIX.name)
    fake_arm = scripts / ARM.name
    fake_arm.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$$" > "$1.arm.pid"\n'
        "printf '%s\\n' \"$1\" >> launched.txt\n"
        "sleep 60 &\n"
        "child=$!\n"
        'printf \'%s\\n\' "${child}" > "$1.child.pid"\n'
        "cleanup() {\n"
        '    kill -TERM "${child}" 2>/dev/null || true\n'
        '    wait "${child}" 2>/dev/null || true\n'
        "    exit 143\n"
        "}\n"
        "trap cleanup TERM INT HUP\n"
        'wait "${child}"\n'
    )
    fake_arm.chmod(0o755)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    pid_paths = [
        tmp_path / f"{method}.{kind}.pid"
        for method in ("uam_nll", "uam_simnpo")
        for kind in ("arm", "child")
    ]
    process = subprocess.Popen(
        ["bash", f"scripts/{MATRIX.name}", "term-matrix"],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_until(lambda: all(path.is_file() for path in pid_paths))
        process.terminate()
        assert process.wait(timeout=10) != 0
        pids = [int(path.read_text()) for path in pid_paths]
        assert _wait_until(lambda: not any(_process_running(pid) for pid in pids))
        assert set((tmp_path / "launched.txt").read_text().splitlines()) == {
            "uam_nll",
            "uam_simnpo",
        }
        manifest = (
            (tmp_path / "saves/exp/UAM_SMOKE/term-matrix/RUN_MANIFEST.tsv")
            .read_text()
            .splitlines()
        )
        assert len(manifest) == 3
        assert {line.split("\t")[2] for line in manifest[1:]} == {
            "uam_nll",
            "uam_simnpo",
        }
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        _cleanup_pids(pid_paths)


@pytest.mark.parametrize("grace", ["-1", "1.5", "invalid"])
def test_arm_rejects_invalid_term_grace_before_conda(tmp_path, grace):
    conda_spy, marker = _write_conda_spy(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(conda_spy)
    environment["UAM_ARM_TERM_GRACE_SECONDS"] = grace

    result = subprocess.run(
        ["bash", str(ARM), "uam_nll", "0", "bad-grace"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "UAM_ARM_TERM_GRACE_SECONDS" in result.stderr
    assert not marker.exists()
    assert not (tmp_path / "saves").exists()


@pytest.mark.parametrize(
    ("arm_grace", "matrix_grace"),
    [
        ("-1", "3"),
        ("1.5", "3"),
        ("invalid", "3"),
        ("1", "-1"),
        ("1", "3.5"),
        ("1", "invalid"),
        ("1", "1"),
        ("3", "1"),
    ],
)
def test_matrix_rejects_invalid_or_nonhierarchical_term_graces_before_conda(
    tmp_path,
    arm_grace,
    matrix_grace,
):
    conda_spy, marker = _write_conda_spy(tmp_path)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(conda_spy)
    environment["UAM_ARM_TERM_GRACE_SECONDS"] = arm_grace
    environment["UAM_MATRIX_TERM_GRACE_SECONDS"] = matrix_grace

    result = subprocess.run(
        ["bash", str(MATRIX), "bad-grace"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "grace" in result.stderr.lower()
    assert not marker.exists()
    assert not (tmp_path / "saves").exists()


def test_matrix_term_allows_arm_to_kill_and_reap_term_resistant_training_group(
    tmp_path,
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(MATRIX, scripts / MATRIX.name)
    shutil.copy2(ARM, scripts / ARM.name)
    executable = tmp_path / "bin/accelerate"
    executable.parent.mkdir()
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "output_dir=\n"
        'for argument in "$@"; do\n'
        "    case ${argument} in\n"
        "        paths.output_dir=*) output_dir=${argument#*=} ;;\n"
        "    esac\n"
        "done\n"
        "method=${output_dir##*/}\n"
        'printf \'%s\\n\' "$PPID" > "${FAKE_PID_ROOT}/${method}.arm.pid"\n'
        'printf \'%s\\n\' "$$" > "${FAKE_PID_ROOT}/${method}.trainer.pid"\n'
        "trap '' TERM\n"
        "sleep 60 &\n"
        "child=$!\n"
        "printf '%s\\n' \"${child}\" > "
        '"${FAKE_PID_ROOT}/${method}.child.pid"\n'
        'wait "${child}"\n'
    )
    executable.chmod(0o755)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{executable.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    environment["UAM_ARM_TERM_GRACE_SECONDS"] = "1"
    environment["UAM_MATRIX_TERM_GRACE_SECONDS"] = "3"
    environment["FAKE_PID_ROOT"] = str(tmp_path)
    pid_paths = [
        tmp_path / f"{method}.{kind}.pid"
        for method in ("uam_nll", "uam_simnpo")
        for kind in ("arm", "trainer", "child")
    ]
    process = subprocess.Popen(
        ["bash", f"scripts/{MATRIX.name}", "grace-hierarchy"],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_until(lambda: all(path.is_file() for path in pid_paths))
        process.terminate()
        assert process.wait(timeout=2.5) != 0
        pids = [int(path.read_text()) for path in pid_paths]
        assert _wait_until(lambda: not any(_process_running(pid) for pid in pids))
        manifest = (
            (tmp_path / "saves/exp/UAM_SMOKE/grace-hierarchy/RUN_MANIFEST.tsv")
            .read_text()
            .splitlines()
        )
        rows = [line.split("\t") for line in manifest[1:]]
        assert len(rows) == 2
        assert len({row[0] for row in rows}) == 2
        assert {row[2] for row in rows} == {"uam_nll", "uam_simnpo"}
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        _cleanup_pids(pid_paths)


def test_arm_term_kills_training_session_and_cleans_local_staging(tmp_path):
    matrix_root = tmp_path / "saves/exp/UAM_SMOKE/term-arm"
    matrix_root.mkdir(parents=True)
    local_arm = tmp_path / "local/term-arm/uam_nll"
    executable = tmp_path / "bin/accelerate"
    executable.parent.mkdir()
    trainer_pid_path = tmp_path / "trainer.pid"
    child_pid_path = tmp_path / "trainer-child.pid"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"$$\" > '{trainer_pid_path}'\n"
        "sleep 60 &\n"
        "child=$!\n"
        f"printf '%s\\n' \"${{child}}\" > '{child_pid_path}'\n"
        "cleanup() {\n"
        '    kill -TERM "${child}" 2>/dev/null || true\n'
        '    wait "${child}" 2>/dev/null || true\n'
        "    exit 143\n"
        "}\n"
        "trap cleanup TERM INT HUP\n"
        'wait "${child}"\n'
    )
    executable.chmod(0o755)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))
    environment["PATH"] = f"{executable.parent}:{environment['PATH']}"
    environment["UAM_LOCAL_ROOT"] = str(tmp_path / "local")
    process = subprocess.Popen(
        ["bash", str(ARM), "uam_nll", "0", "term-arm"],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid_paths = [trainer_pid_path, child_pid_path]
    try:
        assert _wait_until(lambda: all(path.is_file() for path in pid_paths))
        process.terminate()
        assert process.wait(timeout=10) != 0
        pids = [int(path.read_text()) for path in pid_paths]
        assert _wait_until(lambda: not any(_process_running(pid) for pid in pids))
        assert not local_arm.exists()
        assert not (matrix_root / "uam_nll").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        _cleanup_pids(pid_paths)


def test_matrix_never_starts_wave_two_when_both_wave_one_children_are_done(
    tmp_path,
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(MATRIX, scripts / MATRIX.name)
    fake_arm = scripts / ARM.name
    fake_arm.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s:%s\\n\' "$3" "$1" >> launched.txt\n'
        "case $1 in\n"
        "    uam_nll) exit 0 ;;\n"
        "    uam_simnpo) exit 9 ;;\n"
        "    *) exit 0 ;;\n"
        "esac\n"
    )
    fake_arm.chmod(0o755)
    environment = os.environ.copy()
    environment["CONDA_EXE"] = str(_write_fake_conda(tmp_path))

    for attempt in range(30):
        timestamp = f"wave-race-{attempt}"
        result = subprocess.run(
            ["bash", f"scripts/{MATRIX.name}", timestamp],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0

    launches = (tmp_path / "launched.txt").read_text().splitlines()
    for attempt in range(30):
        timestamp = f"wave-race-{attempt}"
        assert {
            line.split(":", maxsplit=1)[1]
            for line in launches
            if line.startswith(f"{timestamp}:")
        } == {"uam_nll", "uam_simnpo"}


def load_analyzer():
    assert ANALYZER.is_file(), f"Missing analyzer: {ANALYZER}"
    spec = importlib.util.spec_from_file_location("analyze_uam_smoke", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyzer_mean_is_stable_across_python_float_sum_algorithms():
    analyzer = load_analyzer()
    values = [
        0.7913729666566313,
        1.0000085533190375,
        1.0003326608700953,
        1.0002989363633583,
        1.0002284297544124,
        1.0003058529494582,
        1.0013736281680377,
        1.0009572335259038,
        1.001692950182522,
        1.0016288681299705,
    ]

    assert analyzer._mean(values) == math.fsum(values) / len(values)


def method_parameters(method):
    return {
        "uam_nll": ("uam", "nll", "fixed_loss"),
        "uam_simnpo": ("uam", "simnpo", "fixed_loss"),
        "uam_gu_nll": ("uam_gu", "nll", "metric_trust"),
        "uam_gu_simnpo": ("uam_gu", "simnpo", "metric_trust"),
    }[method]


def geometry_record(method, update_step):
    mode, signal, normalization = method_parameters(method)
    gate_kept = method.startswith("uam_gu") and update_step % 2 == 0
    record = {
        "record_type": "uam_geometry",
        "update_step": update_step,
        "mode": mode,
        "forget_signal": signal,
        "perturbation_normalization": normalization,
        "rho": 0.05,
        "requested_perturbation_norm": 0.2,
        "effective_perturbation_norm": 0.16 + 0.002 * update_step,
        "effective_perturbation_ratio": 0.8 + 0.01 * update_step,
        "forget_norm": 2.0,
        "retain_norm": 3.0,
        "perturbed_retain_norm": 3.2,
        "uam_coefficient": 0.4,
        "residual_tangent_norm": 0.3,
        "residual_normal_norm": 0.1,
        "relative_residual_orthogonality": 1e-8 * update_step,
        "residual_forget_gate_dot": -0.2,
        "residual_gate_kept": gate_kept,
        "predicted_forget_directional_derivative": -0.3,
        "predicted_retain_directional_derivative": -0.4,
        "identity_fallback_parameters": 0,
        "replay_microsteps": 8,
        "uam_calls": update_step,
        "replay_calls": update_step,
    }
    assert GEOMETRY_REQUIRED <= record.keys()
    return record


def actual_delta_record(update_step):
    return {
        "record_type": "actual_delta",
        "update_step": update_step,
        "coverage": "full",
        "sampled_elements": 100,
        "forget_directional_derivative": -0.02 * update_step,
        "retain_directional_derivative": -0.03 * update_step,
        "forget_gradient_norm": 2.0,
        "retain_gradient_norm": 3.0,
        "parameter_delta_norm": 0.01 * update_step,
    }


def write_diagnostics(path, method):
    records = []
    for update_step in range(1, 11):
        records.append(geometry_record(method, update_step))
        if update_step in {1, 10}:
            records.append(actual_delta_record(update_step))
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def write_arm(root, method, metric_offset):
    arm_dir = root / method
    eval_dir = arm_dir / "evals"
    eval_dir.mkdir(parents=True)
    (eval_dir / "TOFU_SUMMARY.json").write_text(
        json.dumps(
            {
                "forget_metric": 10.0 + metric_offset,
                "retain_metric": 20.0 - metric_offset,
            }
        )
        + "\n"
    )
    write_diagnostics(arm_dir / "uam_diagnostics.jsonl", method)


def write_manifest(root, rows):
    (root / "RUN_MANIFEST.tsv").write_text(
        "\t".join(MANIFEST_HEADER)
        + "\n"
        + "".join("\t".join(row) + "\n" for row in rows)
    )


def make_matrix(tmp_path):
    root = tmp_path / "stamp"
    root.mkdir()
    rows = []
    for index, method in enumerate(METHODS):
        write_arm(root, method, float(index))
        gpu = str(index % 2)
        rows.append(
            [
                str(100 + index),
                gpu,
                method,
                f"start-{index}",
                f"end-{index}",
                "0",
                f"bash scripts/uam_smoke_arm.sh {method} {gpu} stamp",
            ]
        )
    write_manifest(root, rows)
    return root, rows


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_analyzer_public_api_and_happy_path(tmp_path):
    analyzer = load_analyzer()
    root, rows = make_matrix(tmp_path)

    assert analyzer.METHODS == METHODS
    for name in (
        "read_manifest",
        "find_summary",
        "read_diagnostics",
        "checkpoint_payloads",
        "analyze_matrix",
        "render_markdown",
        "write_outputs",
    ):
        assert callable(getattr(analyzer, name))

    result = analyzer.analyze_matrix(root)

    assert result["raw"]["uam_nll"]["forget_metric"] == 10.0
    assert result["raw"]["uam_gu_simnpo"]["retain_metric"] == 17.0
    assert result["deltas"]["uam_nll_minus_uam_nll"] == {
        "forget_metric": 0.0,
        "retain_metric": 0.0,
    }
    assert result["deltas"]["uam_gu_nll_minus_uam_nll"]["forget_metric"] == 2.0
    diagnostics = result["diagnostics"]["uam_gu_nll"]
    assert diagnostics["mean_effective_perturbation_ratio"] == pytest.approx(0.855)
    assert diagnostics["mean_residual_gate_rate"] == pytest.approx(0.5)
    assert diagnostics["max_relative_residual_orthogonality_kept"] == pytest.approx(
        1e-7
    )
    assert diagnostics["finalizer_count"] == 10
    assert diagnostics["replay_count"] == 10
    assert [probe["update_step"] for probe in result["actual_deltas"]["uam_nll"]] == [
        1,
        10,
    ]
    assert result["provenance"]["manifest_path"] == str(root / "RUN_MANIFEST.tsv")
    assert result["provenance"]["manifest_rows"] == 4
    assert result["provenance"]["successful_rows"] == 4
    assert result["provenance"]["checkpoint_payload_count"] == 0
    assert result["provenance"]["manifest"] == {
        row[2]: dict(zip(MANIFEST_HEADER, row, strict=True)) for row in rows
    }


def test_pure_uam_has_no_kept_residual_orthogonality_maximum(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)

    result = analyzer.analyze_matrix(root)

    assert (
        result["diagnostics"]["uam_nll"]["max_relative_residual_orthogonality_kept"]
        is None
    )
    assert result["diagnostics"]["uam_nll"]["mean_residual_gate_rate"] == 0.0


def test_analyzer_writes_sorted_json_and_single_newline_markdown(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    result = analyzer.analyze_matrix(root)
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "analysis.json"

    analyzer.write_outputs(result, markdown_path, json_path)

    markdown = markdown_path.read_text()
    assert "one seed" in markdown.lower()
    assert "ten updates" in markdown.lower()
    assert "mechanism check" in markdown.lower()
    assert "no superiority claim" in markdown.lower()
    assert "Method diagnostics" in markdown
    assert "Metric and actual-delta probes" in markdown
    assert markdown.endswith("\n") and not markdown.endswith("\n\n")
    encoded = json_path.read_text()
    assert encoded == json.dumps(result, indent=2, sort_keys=True) + "\n"
    assert encoded.endswith("\n") and not encoded.endswith("\n\n")


@pytest.mark.parametrize("case", ["duplicate", "missing", "failed", "extra"])
def test_manifest_rejects_duplicate_missing_failed_and_extra_methods(tmp_path, case):
    analyzer = load_analyzer()
    root, rows = make_matrix(tmp_path)
    if case == "duplicate":
        rows.append(rows[0].copy())
    elif case == "missing":
        rows.pop()
    elif case == "failed":
        rows[2][5] = "9"
    else:
        extra = rows[-1].copy()
        extra[2] = "unapproved"
        rows.append(extra)
    write_manifest(root, rows)

    with pytest.raises(ValueError):
        analyzer.read_manifest(root)


def test_manifest_rejects_header_drift(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    manifest = root / "RUN_MANIFEST.tsv"
    manifest.write_text(manifest.read_text().replace("\tcommand\n", "\textra\n", 1))

    with pytest.raises(ValueError, match="schema"):
        analyzer.read_manifest(root)


def test_manifest_rejects_rows_with_extra_columns(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    manifest = root / "RUN_MANIFEST.tsv"
    lines = manifest.read_text().splitlines()
    lines[1] += "\textra-cell"
    manifest.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="schema"):
        analyzer.read_manifest(root)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("pid", "0"),
        ("pid", "not-a-pid"),
        ("gpu", "2"),
        ("gpu", "-1"),
        ("start_utc", ""),
        ("end_utc", ""),
        ("command", "bash scripts/uam_smoke_arm.sh uam_nll 1 stamp"),
    ],
)
def test_manifest_rejects_invalid_process_gpu_time_and_command(
    tmp_path,
    column,
    value,
):
    analyzer = load_analyzer()
    root, rows = make_matrix(tmp_path)
    index = MANIFEST_HEADER.index(column)
    rows[0][index] = value
    write_manifest(root, rows)

    with pytest.raises(ValueError, match="manifest|Manifest|PID|GPU|command|timestamp"):
        analyzer.read_manifest(root)


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_summary_requires_exactly_one_file(tmp_path, case):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    arm_dir = root / "uam_nll"
    summary = arm_dir / "evals/TOFU_SUMMARY.json"
    if case == "missing":
        summary.unlink()
    else:
        duplicate = arm_dir / "nested/TOFU_SUMMARY.json"
        duplicate.parent.mkdir()
        duplicate.write_text(summary.read_text())

    with pytest.raises(ValueError, match="exactly one"):
        analyzer.find_summary(arm_dir)


@pytest.mark.parametrize(
    "summary",
    [
        {},
        {"metric": "not numeric"},
        {"metric": True},
        {"metric": math.inf},
    ],
)
def test_summary_rejects_empty_nonnumeric_and_nonfinite_metrics(tmp_path, summary):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_nll/evals/TOFU_SUMMARY.json"
    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError):
        analyzer.find_summary(root / "uam_nll")


def test_analyzer_rejects_metric_key_drift(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_gu_nll/evals/TOFU_SUMMARY.json"
    path.write_text(json.dumps({"different_metric": 1.0}))

    with pytest.raises(ValueError, match="Metric keys differ"):
        analyzer.analyze_matrix(root)


def test_analyzer_rejects_nonfinite_metric_subtraction(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    baseline_path = root / "uam_nll/evals/TOFU_SUMMARY.json"
    comparison_path = root / "uam_simnpo/evals/TOFU_SUMMARY.json"
    baseline = json.loads(baseline_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    baseline["forget_metric"] = -1e308
    comparison["forget_metric"] = 1e308
    baseline_path.write_text(json.dumps(baseline))
    comparison_path.write_text(json.dumps(comparison))

    with pytest.raises(ValueError, match="finite"):
        analyzer.analyze_matrix(root)


def test_write_outputs_serialization_failure_preserves_both_sentinels(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    result = analyzer.analyze_matrix(root)
    result["not_json_serializable"] = object()
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "analysis.json"
    markdown_path.write_bytes(b"old markdown\n")
    json_path.write_bytes(b"old json\n")

    with pytest.raises(TypeError):
        analyzer.write_outputs(result, markdown_path, json_path)

    assert markdown_path.read_bytes() == b"old markdown\n"
    assert json_path.read_bytes() == b"old json\n"
    assert {path.name for path in tmp_path.iterdir()} == {
        "stamp",
        "report.md",
        "analysis.json",
    }


def test_write_outputs_temp_fsync_failure_preserves_both_sentinels(
    tmp_path,
    monkeypatch,
):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    result = analyzer.analyze_matrix(root)
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "analysis.json"
    markdown_path.write_bytes(b"old markdown\n")
    json_path.write_bytes(b"old json\n")
    real_fsync = os.fsync
    calls = 0

    def fail_second_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced second temp fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_second_fsync)

    with pytest.raises(OSError, match="forced second temp fsync failure"):
        analyzer.write_outputs(result, markdown_path, json_path)

    assert markdown_path.read_bytes() == b"old markdown\n"
    assert json_path.read_bytes() == b"old json\n"
    assert {path.name for path in tmp_path.iterdir()} == {
        "stamp",
        "report.md",
        "analysis.json",
    }


@pytest.mark.parametrize(
    ("markdown_before", "json_before"),
    [
        (b"old markdown\n", b"old json\n"),
        (None, b"old json\n"),
        (b"old markdown\n", None),
        (None, None),
    ],
)
def test_write_outputs_second_replace_failure_restores_exact_prior_state(
    tmp_path,
    monkeypatch,
    markdown_before,
    json_before,
):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    result = analyzer.analyze_matrix(root)
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "analysis.json"
    if markdown_before is not None:
        markdown_path.write_bytes(markdown_before)
    if json_before is not None:
        json_path.write_bytes(json_before)
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced second publication replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="forced second publication replace failure"):
        analyzer.write_outputs(result, markdown_path, json_path)

    assert calls >= 2
    if markdown_before is None:
        assert not markdown_path.exists()
    else:
        assert markdown_path.read_bytes() == markdown_before
    if json_before is None:
        assert not json_path.exists()
    else:
        assert json_path.read_bytes() == json_before
    expected_names = {"stamp"}
    if markdown_before is not None:
        expected_names.add("report.md")
    if json_before is not None:
        expected_names.add("analysis.json")
    assert {path.name for path in tmp_path.iterdir()} == expected_names


def test_diagnostics_rejects_malformed_json(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_nll/uam_diagnostics.jsonl"
    path.write_text("{not-json}\n")

    with pytest.raises(ValueError, match="Malformed diagnostics"):
        analyzer.read_diagnostics(path)


@pytest.mark.parametrize("case", ["missing_field", "nonfinite", "wrong_count"])
def test_diagnostics_rejects_bad_geometry_schema(tmp_path, case):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_nll/uam_diagnostics.jsonl"
    records = read_jsonl(path)
    geometry = [record for record in records if record["record_type"] == "uam_geometry"]
    if case == "missing_field":
        geometry[0].pop("rho")
    elif case == "nonfinite":
        geometry[0]["rho"] = math.nan
    else:
        records.remove(geometry[-1])
    write_jsonl(path, records)

    with pytest.raises(ValueError):
        analyzer.read_diagnostics(path)


def test_diagnostics_rejects_counter_drift(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_simnpo/uam_diagnostics.jsonl"
    records = read_jsonl(path)
    geometry = next(record for record in records if record["update_step"] == 4)
    geometry["replay_calls"] = 3
    write_jsonl(path, records)

    with pytest.raises(ValueError, match="counter"):
        analyzer.read_diagnostics(path)


def test_diagnostics_rejects_rho_drift_across_updates(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_nll/uam_diagnostics.jsonl"
    records = read_jsonl(path)
    geometry = [record for record in records if record["record_type"] == "uam_geometry"]
    geometry[-1]["rho"] = 0.06
    write_jsonl(path, records)

    with pytest.raises(ValueError, match="rho"):
        analyzer.read_diagnostics(path)


@pytest.mark.parametrize(
    "field",
    [
        "requested_perturbation_norm",
        "effective_perturbation_norm",
        "effective_perturbation_ratio",
    ],
)
def test_diagnostics_rejects_nonpositive_perturbation_fields(tmp_path, field):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_gu_nll/uam_diagnostics.jsonl"
    records = read_jsonl(path)
    geometry = next(
        record for record in records if record["record_type"] == "uam_geometry"
    )
    geometry[field] = 0.0
    write_jsonl(path, records)

    with pytest.raises(ValueError, match="positive"):
        analyzer.read_diagnostics(path)


@pytest.mark.parametrize(
    "field",
    [
        "requested_perturbation_norm",
        "effective_perturbation_norm",
        "forget_norm",
        "retain_norm",
        "perturbed_retain_norm",
        "residual_tangent_norm",
        "residual_normal_norm",
    ],
)
def test_diagnostics_rejects_negative_geometry_norms(tmp_path, field):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_gu_nll/uam_diagnostics.jsonl"
    records = read_jsonl(path)
    geometry = next(
        record for record in records if record["record_type"] == "uam_geometry"
    )
    geometry[field] = -0.1
    write_jsonl(path, records)

    with pytest.raises(ValueError, match="nonnegative|positive"):
        analyzer.read_diagnostics(path)


def test_diagnostics_requires_eight_replay_microsteps_per_update(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_simnpo/uam_diagnostics.jsonl"
    records = read_jsonl(path)
    geometry = next(
        record for record in records if record["record_type"] == "uam_geometry"
    )
    geometry["replay_microsteps"] = 7
    write_jsonl(path, records)

    with pytest.raises(ValueError, match="replay_microsteps"):
        analyzer.read_diagnostics(path)


@pytest.mark.parametrize("case", ["missing_step", "extra_probe", "wrong_coverage"])
def test_diagnostics_requires_exact_full_actual_delta_steps(tmp_path, case):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_gu_simnpo/uam_diagnostics.jsonl"
    records = read_jsonl(path)
    actual = [record for record in records if record["record_type"] == "actual_delta"]
    if case == "missing_step":
        records.remove(actual[-1])
    elif case == "extra_probe":
        records.append(actual_delta_record(5))
    else:
        actual[0]["coverage"] = "sampled"
    write_jsonl(path, records)

    with pytest.raises(ValueError, match="actual-delta|coverage"):
        analyzer.read_diagnostics(path)


def test_diagnostics_rejects_missing_or_nonfinite_actual_fields(tmp_path):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_gu_nll/uam_diagnostics.jsonl"
    records = read_jsonl(path)
    actual = next(
        record for record in records if record["record_type"] == "actual_delta"
    )
    actual["forget_directional_derivative"] = math.inf
    write_jsonl(path, records)

    with pytest.raises(ValueError):
        analyzer.read_diagnostics(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("forget_gradient_norm", -0.1, "nonnegative"),
        ("retain_gradient_norm", -0.1, "nonnegative"),
        ("parameter_delta_norm", -0.1, "positive"),
        ("parameter_delta_norm", 0.0, "positive"),
    ],
)
def test_diagnostics_rejects_invalid_actual_delta_norms(
    tmp_path,
    field,
    value,
    message,
):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    path = root / "uam_gu_nll/uam_diagnostics.jsonl"
    records = read_jsonl(path)
    actual = next(
        record for record in records if record["record_type"] == "actual_delta"
    )
    actual[field] = value
    write_jsonl(path, records)

    with pytest.raises(ValueError, match=message):
        analyzer.read_diagnostics(path)


@pytest.mark.parametrize(
    ("relative_path", "is_directory"),
    [
        ("checkpoint-3", True),
        ("nested/model.safetensors", False),
        ("nested/model.bin", False),
        ("nested/optimizer.pt", False),
        ("nested/rng_state.pth", False),
        ("nested/state.ckpt", False),
        ("nested/model_state.json", False),
        ("nested/training_args.json", False),
    ],
)
def test_analyzer_rejects_checkpoint_payloads(tmp_path, relative_path, is_directory):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    payload = root / relative_path
    if is_directory:
        payload.mkdir()
    else:
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"payload")

    assert analyzer.checkpoint_payloads(root)
    with pytest.raises(ValueError, match="checkpoint payload"):
        analyzer.analyze_matrix(root)


@pytest.mark.parametrize(
    "link_name",
    ["optimizer.pt", "checkpoint-10", "ordinary-linked-directory"],
)
def test_analyzer_rejects_any_symlink_without_following_target(tmp_path, link_name):
    analyzer = load_analyzer()
    root, _ = make_matrix(tmp_path)
    if link_name == "ordinary-linked-directory":
        target = tmp_path / "outside-directory"
        target.mkdir()
        (target / "model.bin").write_bytes(b"outside")
        (root / link_name).symlink_to(target, target_is_directory=True)
    elif link_name == "checkpoint-10":
        target = tmp_path / "outside-checkpoint"
        target.mkdir()
        (root / link_name).symlink_to(target, target_is_directory=True)
    else:
        target = tmp_path / "outside-file"
        target.write_bytes(b"outside")
        (root / link_name).symlink_to(target)

    payloads = analyzer.checkpoint_payloads(root)

    assert str(root / link_name) in payloads
    with pytest.raises(ValueError, match="checkpoint payload"):
        analyzer.analyze_matrix(root)
