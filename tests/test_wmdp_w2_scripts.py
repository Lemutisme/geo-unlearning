import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "scripts/wmdp_uam_w2_arm.sh"
MATRIX = ROOT / "scripts/wmdp_uam_w2_matrix.sh"
STRONG = ROOT / "scripts/wmdp_uam_strong_pair.sh"


def script_text(path):
    assert path.is_file()
    return path.read_text(encoding="utf-8")


def test_arm_script_has_fixed_five_arm_gpu0_contract():
    text = script_text(ARM)
    for arm in ("base", "rmu", "rmu_gu", "uam", "uam_gu"):
        assert f"{arm})" in text
    for token in (
        'if [[ "${gpu}" != "0" ]]',
        "conda activate unlearning",
        'export CUDA_VISIBLE_DEVICES="0"',
        "experiment=unlearn/wmdp/default",
        "trainer=WMDPRMUUnlearn",
        "trainer=WMDPUAMUnlearn",
        "save_model_after_train=false",
        "trainer.args.optim=paged_adamw_32bit",
        "trainer.args.gradient_accumulation_steps=4",
        "learning_rate=${WMDP_W2_LEARNING_RATE:-5e-5}",
        "max_steps=${WMDP_W2_MAX_STEPS:-80}",
        "lr_scheduler_type=${WMDP_W2_LR_SCHEDULER_TYPE:-linear}",
        '"trainer.args.learning_rate=${learning_rate}"',
        '"trainer.args.max_steps=${max_steps}"',
        '"trainer.args.lr_scheduler_type=${lr_scheduler_type}"',
        '"trainer.method_args.geometric_config.actual_delta_steps=[1,${max_steps}]"',
        "data/datasets@data.retain=WMDP_wikitext_retain",
        "~data.retain.WMDP_retain",
        "audit_checkpoint_payloads",
        "provenance.json",
        "resource.json",
    ):
        assert token in text


def test_wikitext_override_removes_inline_wmdp_retain_fragment():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="unlearn",
            overrides=[
                "experiment=unlearn/wmdp/default",
                "data/datasets@data.retain=WMDP_wikitext_retain",
                "~data.retain.WMDP_retain",
                "task_name=cfg_probe",
            ],
        )

    assert list(cfg.data.retain) == ["WMDP_wikitext_retain"]
    assert cfg.data.retain.WMDP_wikitext_retain.handler == "PretrainingDataset"


def test_matrix_is_strictly_base_gated_and_sequential():
    text = script_text(MATRIX)
    positions = [text.index(f'run_arm "{arm}"') for arm in ("base", "rmu", "rmu_gu", "uam", "uam_gu")]
    assert positions == sorted(positions)
    gate = text.index("--check-base-gate")
    assert positions[0] < gate < positions[1]
    assert "CUDA_VISIBLE_DEVICES=0" in text


def test_scripts_have_valid_bash_syntax():
    for script in (ARM, MATRIX, STRONG):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_strong_pair_has_fixed_two_arm_calibration_contract():
    text = script_text(STRONG)
    for token in (
        "export CUDA_VISIBLE_DEVICES=0",
        "export WMDP_W2_LEARNING_RATE=1.25e-4",
        "export WMDP_W2_MAX_STEPS=160",
        "export WMDP_W2_LR_SCHEDULER_TYPE=constant",
        "WMDP_UAM_W2_STRONG",
        'run_arm "uam"',
        'run_arm "uam_gu"',
        "scripts/analyze_wmdp_uam_strong_pair.py",
        "w2-20260713-2/W2_SUMMARY.json",
    ):
        assert token in text
    assert 'run_arm "base"' not in text
    assert 'run_arm "rmu"' not in text
    assert 'run_arm "rmu_gu"' not in text
    assert text.index('run_arm "uam"') < text.index('run_arm "uam_gu"')


def test_strong_pair_rejects_unsafe_or_existing_run_id(tmp_path):
    environment = os.environ.copy()
    environment["WMDP_UAM_STRONG_ROOT"] = str(tmp_path)

    unsafe = subprocess.run(
        ["bash", str(STRONG), "../unsafe"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert unsafe.returncode == 2
    assert "Unsafe run id" in unsafe.stderr

    existing = tmp_path / "existing"
    existing.mkdir()
    duplicate = subprocess.run(
        ["bash", str(STRONG), "existing"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode == 1
    assert "already exists" in duplicate.stderr


def test_arm_runner_does_not_require_external_gnu_time():
    text = script_text(ARM)
    assert "/usr/bin/time" not in text
    assert 'ps -o rss= -g "${process_pid}"' in text


def test_arm_runner_prints_command_log_before_cleaning_failed_staging():
    text = script_text(ARM)
    failure = text.index('if [[ ${exit_code} -ne 0 ]]')
    cleanup = text.index('exit "${exit_code}"', failure)
    assert 'tail -200 "${local_arm}/run.log" >&2' in text[failure:cleanup]


def run_invalid_strength(tmp_path, field, value):
    environment = os.environ.copy()
    environment["WMDP_DATA_ROOT"] = str(tmp_path / "missing-data")
    environment[field] = value
    return subprocess.run(
        ["bash", str(ARM), "uam", "0", f"invalid-{field.lower()}"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "true"])
def test_arm_rejects_invalid_learning_rate_before_launch(tmp_path, value):
    result = run_invalid_strength(tmp_path, "WMDP_W2_LEARNING_RATE", value)

    assert result.returncode == 2
    assert "learning rate" in result.stderr.lower()
    assert "Missing WMDP-Cyber corpus" not in result.stderr


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true"])
def test_arm_rejects_invalid_max_steps_before_launch(tmp_path, value):
    result = run_invalid_strength(tmp_path, "WMDP_W2_MAX_STEPS", value)

    assert result.returncode == 2
    assert "max steps" in result.stderr.lower()
    assert "Missing WMDP-Cyber corpus" not in result.stderr


def test_arm_rejects_invalid_scheduler_before_launch(tmp_path):
    result = run_invalid_strength(
        tmp_path,
        "WMDP_W2_LR_SCHEDULER_TYPE",
        "cosine",
    )

    assert result.returncode == 2
    assert "scheduler" in result.stderr.lower()
    assert "Missing WMDP-Cyber corpus" not in result.stderr


@pytest.mark.parametrize(
    "relative_path",
    [
        "checkpoint-80",
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ],
)
def test_arm_audit_only_rejects_checkpoint_payloads(tmp_path, relative_path):
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
    if relative_path.startswith("checkpoint-"):
        payload.mkdir(parents=True)
    else:
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"payload")
    result = subprocess.run(
        ["bash", str(ARM), "--audit-only", str(dirty)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Unexpected checkpoint payload" in result.stderr
