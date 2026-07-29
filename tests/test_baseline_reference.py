import importlib.util
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts/run_baseline_reference.py"
GU_REGISTRY = ROOT / "scripts/run_gu_full_matrix.py"


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_baseline():
    return load_script(BASELINE, "baseline_reference")


def load_gu_registry():
    return load_script(GU_REGISTRY, "baseline_gu_registry")


def compose_command(command):
    config_index = command.index("--config-name=unlearn.yaml")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            config_name="unlearn.yaml",
            overrides=command[config_index + 1 :],
        )


def test_seed_zero_baseline_manifest_matches_gu_registry():
    baseline = load_baseline()
    gu = load_gu_registry()

    manifest = baseline.build_manifest()

    assert len(manifest["jobs"]) == 60
    assert manifest["not_applicable"] == gu.build_manifest(0)["not_applicable"]
    assert [
        (job["method"], job["benchmark"], job["seed"], job["arm"])
        for job in manifest["jobs"]
    ] == [
        (job["method"], job["benchmark"], 0, "baseline")
        for job in gu.build_manifest(0)["jobs"]
    ]


def test_baseline_commands_are_live_no_save_and_have_no_gu(tmp_path):
    baseline = load_baseline()

    for job in baseline.build_manifest()["jobs"]:
        command = baseline.build_command(job, tmp_path / job["job_id"])
        rendered = " ".join(command)
        config = compose_command(command)

        assert command[:3] == [
            sys.executable,
            str(ROOT / "src/train.py"),
            "--config-name=unlearn.yaml",
        ]
        assert "trainer.method_args.gu" not in rendered
        assert "gu" not in config.trainer.get("method_args", {})
        assert config.trainer.handler == job["method"]
        assert config.trainer.args.seed == config.trainer.args.data_seed == 0
        assert "save_model_after_train=false" in command
        assert "trainer.args.save_strategy=no" in command
        assert "trainer.args.save_only_model=false" in command
        assert "trainer.args.do_eval=true" in command
        assert "trainer.args.eval_strategy=no" in command
