import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "scripts/wmdp_uam_w2_arm.sh"
MATRIX = ROOT / "scripts/wmdp_uam_w2_matrix.sh"


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
        "trainer.args.max_steps=80",
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
    for script in (ARM, MATRIX):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_arm_runner_does_not_require_external_gnu_time():
    text = script_text(ARM)
    assert "/usr/bin/time" not in text
    assert 'ps -o rss= -g "${process_pid}"' in text


def test_arm_runner_prints_command_log_before_cleaning_failed_staging():
    text = script_text(ARM)
    failure = text.index('if [[ ${exit_code} -ne 0 ]]')
    cleanup = text.index('exit "${exit_code}"', failure)
    assert 'tail -200 "${local_arm}/run.log" >&2' in text[failure:cleanup]


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
