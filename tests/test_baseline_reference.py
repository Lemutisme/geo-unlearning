import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def install_baseline_child(
    monkeypatch,
    baseline,
    job,
    output_dir,
    *,
    returncode=0,
    write_endpoint=True,
    write_updates=True,
    forbidden_path=None,
    observe_gpu=True,
):
    captured = {}
    prefix = {"tofu": "TOFU", "muse": "MUSE", "lm_eval": "LMEval"}[
        job["evaluator_kind"]
    ]

    class StubPopen:
        pid = 4242

        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            self.polls = 0
            if write_updates:
                kwargs["stdout"].write(
                    "{'loss': 1.25, 'epoch': 0.5}\n"
                    "{'loss': 0.75, 'epoch': 1.0}\n"
                )
                kwargs["stdout"].flush()
            if write_endpoint:
                endpoint = output_dir / "checkpoint-2" / "evals"
                endpoint.mkdir(parents=True, exist_ok=True)
                (endpoint / f"{prefix}_SUMMARY.json").write_text(
                    json.dumps({"metric": 0.5})
                )
                (endpoint / f"{prefix}_EVAL.json").write_text(
                    json.dumps({"metric": {"agg_value": 0.5}})
                )
            (output_dir / f'{job["method"]}.log').write_text("hydra\n")
            if forbidden_path is not None:
                path = output_dir / forbidden_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("forbidden\n")

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else returncode

    def sample_gpu(*_args, **_kwargs):
        stdout = "4242, 384 MiB\n" if observe_gpu else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(baseline.subprocess, "Popen", StubPopen)
    monkeypatch.setattr(baseline.subprocess, "run", sample_gpu)
    monkeypatch.setattr(baseline.time, "sleep", lambda _seconds: None)
    return captured


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


def test_completed_baseline_requires_endpoint_update_memory_and_no_artifacts(
    tmp_path,
    monkeypatch,
):
    baseline = load_baseline()
    job = baseline.build_manifest()["jobs"][0]
    output_dir = tmp_path / job["job_id"]
    captured = install_baseline_child(
        monkeypatch,
        baseline,
        job,
        output_dir,
    )

    result = baseline.run_job(job, output_dir, physical_gpu=1)

    assert result["status"] == "completed"
    assert result["optimizer_update_count"] == 2
    assert result["peak_nvml_mib"] == 384
    assert result["endpoint_summary_path"].endswith("_SUMMARY.json")
    assert result["endpoint_raw_path"].endswith("_EVAL.json")
    assert result["forbidden_artifacts"] == []
    assert result["issues"] == []
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert not (output_dir / "gu_diagnostics.jsonl").exists()
    assert json.loads((output_dir / "BASELINE_JOB_RESULT.json").read_text()) == result


@pytest.mark.parametrize(
    ("corruption", "expected"),
    [
        ("missing_endpoint", "endpoint"),
        ("zero_updates", "optimizer update"),
        ("no_gpu_sample", "GPU"),
        ("checkpoint_config", "forbidden"),
        ("nonzero_exit", "subprocess"),
    ],
)
def test_baseline_rejects_invalid_evidence(
    tmp_path,
    monkeypatch,
    corruption,
    expected,
):
    baseline = load_baseline()
    job = baseline.build_manifest()["jobs"][0]
    output_dir = tmp_path / job["job_id"]
    install_baseline_child(
        monkeypatch,
        baseline,
        job,
        output_dir,
        returncode=1 if corruption == "nonzero_exit" else 0,
        write_endpoint=corruption != "missing_endpoint",
        write_updates=corruption != "zero_updates",
        forbidden_path=(
            "checkpoint-2/config.json" if corruption == "checkpoint_config" else None
        ),
        observe_gpu=corruption != "no_gpu_sample",
    )

    result = baseline.run_job(job, output_dir, physical_gpu=0)

    assert result["status"] != "completed"
    assert expected.lower() in json.dumps(result).lower()


def write_baseline_manifest(tmp_path, baseline):
    manifest = baseline.build_manifest()
    manifest["output_root"] = str(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def worker_result(baseline, job, output_dir, gpu, status):
    command = baseline._command_record(job, output_dir, gpu)
    return {
        "job_id": job["job_id"],
        "status": status,
        "failure_kind": (
            "host_io"
            if status == "failed_infrastructure"
            else "subprocess_exit"
            if status == "invalid_scientific"
            else None
        ),
        "command_identity": baseline._command_identity(command),
    }


def test_two_worker_partitions_are_disjoint_and_complete():
    baseline = load_baseline()
    jobs = baseline.build_manifest()["jobs"]

    left = baseline.partition(jobs, worker=0, workers=2)
    right = baseline.partition(jobs, worker=1, workers=2)

    assert len(left) == len(right) == 30
    assert {job["job_id"] for job in left}.isdisjoint(
        job["job_id"] for job in right
    )
    assert {job["job_id"] for job in left + right} == {
        job["job_id"] for job in jobs
    }


def test_worker_retries_only_one_identical_infrastructure_failure(
    tmp_path,
    monkeypatch,
):
    baseline = load_baseline()
    manifest_path, manifest = write_baseline_manifest(tmp_path, baseline)
    job = manifest["jobs"][0]
    monkeypatch.setattr(baseline, "partition", lambda *_args, **_kwargs: [job])
    monkeypatch.setattr(baseline, "admit_gpu", lambda _gpu: 0)
    statuses = iter(["failed_infrastructure", "completed"])

    def run_job(candidate, output_dir, physical_gpu, launched=None):
        if launched is not None:
            launched(4242)
        return worker_result(
            baseline,
            candidate,
            output_dir,
            physical_gpu,
            next(statuses),
        )

    monkeypatch.setattr(baseline, "run_job", run_job)

    state = baseline.run_worker(manifest_path, worker=0, physical_gpu=0)

    actual = state["jobs"][0]
    assert actual["status"] == "completed"
    assert actual["attempt_count"] == 2
    assert actual["attempt_history"][0]["command_identity"] == (
        actual["attempt_history"][1]["command_identity"]
    )


def test_worker_never_retries_scientific_failure(tmp_path, monkeypatch):
    baseline = load_baseline()
    manifest_path, manifest = write_baseline_manifest(tmp_path, baseline)
    job = manifest["jobs"][0]
    monkeypatch.setattr(baseline, "partition", lambda *_args, **_kwargs: [job])
    monkeypatch.setattr(baseline, "admit_gpu", lambda _gpu: 0)
    launches = 0

    def run_job(candidate, output_dir, physical_gpu, launched=None):
        nonlocal launches
        launches += 1
        return worker_result(
            baseline,
            candidate,
            output_dir,
            physical_gpu,
            "invalid_scientific",
        )

    monkeypatch.setattr(baseline, "run_job", run_job)

    state = baseline.run_worker(manifest_path, worker=0, physical_gpu=0)

    assert launches == 1
    assert state["jobs"][0]["status"] == "invalid_scientific"
    assert state["jobs"][0]["attempt_count"] == 1


def test_worker_cli_routes_gpu_and_partition(tmp_path, monkeypatch):
    baseline = load_baseline()
    manifest_path, _ = write_baseline_manifest(tmp_path, baseline)
    observed = {}

    def run_worker(path, worker, physical_gpu):
        observed.update(path=Path(path), worker=worker, physical_gpu=physical_gpu)
        return {"jobs": []}

    monkeypatch.setattr(baseline, "run_worker", run_worker)

    code = baseline.main(
        [
            "worker",
            "--manifest",
            str(manifest_path),
            "--worker",
            "1",
            "--gpu",
            "1",
        ]
    )

    assert code == 0
    assert observed == {"path": manifest_path, "worker": 1, "physical_gpu": 1}


def test_command_evidence_persists_only_registered_environment_overrides(
    tmp_path,
    monkeypatch,
):
    baseline = load_baseline()
    gu = load_gu_registry()
    job = baseline.build_manifest()["jobs"][0]
    monkeypatch.setenv("BASELINE_TEST_SECRET", "must-not-be-persisted")

    record = baseline._command_record(job, tmp_path / job["job_id"], 0)
    child = baseline.child_environment(job, 0)
    expected = gu.environment_overrides(job)
    expected["CUDA_VISIBLE_DEVICES"] = "0"

    assert record["environment_overrides"] == expected
    assert "BASELINE_TEST_SECRET" not in record["environment_overrides"]
    assert child["BASELINE_TEST_SECRET"] == "must-not-be-persisted"


def test_eval_all_is_checkpoint_free_dual_worker_launcher():
    text = (ROOT / "scripts/eval_all.sh").read_text()

    assert "run_baseline_reference.py manifest" in text
    assert "--worker 0 --gpu 0" in text
    assert "--worker 1 --gpu 1" in text
    assert text.count("setsid python scripts/run_baseline_reference.py worker") == 2
    assert "src/eval.py" not in text
    assert "accelerate launch" not in text
    assert "saves/unlearn" not in text


def test_improvement_sign_is_positive_when_gu_is_better():
    baseline = load_baseline()

    assert baseline.improvement(
        "model_utility", baseline_value=0.60, gu_value=0.65
    ) == pytest.approx(0.05)
    assert baseline.improvement(
        "extraction_strength", baseline_value=0.70, gu_value=0.40
    ) == pytest.approx(0.30)


def test_render_comparison_contains_all_six_seed_zero_tables():
    baseline = load_baseline()
    report = {
        "scope": "seed0 reference; no variance estimate",
        "benchmarks": {
            benchmark: [
                {
                    "method": "CEU",
                    "metric": "model_utility",
                    "baseline": 0.60,
                    "gu": 0.65,
                    "improvement": 0.05,
                }
            ]
            for benchmark in (
                "tofu_forget01",
                "tofu_forget05",
                "tofu_forget10",
                "muse_news",
                "muse_books",
                "wmdp_cyber",
            )
        },
    }

    markdown = baseline.render_comparison(report)

    assert markdown.count("| Method | Metric | Baseline | GU | Δ improvement |") == 6
    assert "seed0 reference; no variance estimate" in markdown


def test_compare_cli_routes_both_roots(tmp_path, monkeypatch):
    baseline = load_baseline()
    baseline_root = tmp_path / "baseline"
    gu_root = tmp_path / "gu"
    observed = {}
    report = {"scope": "seed0 reference", "benchmarks": {}}

    def compare(left, right):
        observed.update(left=Path(left), right=Path(right))
        return report

    monkeypatch.setattr(baseline, "compare", compare)
    monkeypatch.setattr(
        baseline,
        "write_comparison",
        lambda payload, root: observed.update(payload=payload, output=Path(root)),
    )

    code = baseline.main(
        [
            "compare",
            "--baseline-root",
            str(baseline_root),
            "--gu-root",
            str(gu_root),
        ]
    )

    assert code == 0
    assert observed == {
        "left": baseline_root,
        "right": gu_root,
        "payload": report,
        "output": baseline_root,
    }


def test_smoke_command_is_one_step_no_eval_no_save_and_no_gu(tmp_path):
    baseline = load_baseline()
    for benchmark in ("tofu_forget01", "muse_news", "wmdp_cyber"):
        job = next(
            job
            for job in baseline.build_manifest()["jobs"]
            if job["benchmark"] == benchmark
        )
        command = baseline.build_smoke_command(job, tmp_path / job["job_id"])
        config = compose_command(command)

        assert config.trainer.args.max_steps == 1
        assert config.trainer.args.do_eval is False
        assert config.save_model_after_train is False
        assert "gu" not in config.trainer.get("method_args", {})
        assert not any("trainer.method_args.gu" in arg for arg in command)


def test_smoke_run_requires_update_but_not_endpoint(tmp_path, monkeypatch):
    baseline = load_baseline()
    job = baseline.build_manifest()["jobs"][0]
    output_dir = tmp_path / job["job_id"]
    install_baseline_child(
        monkeypatch,
        baseline,
        job,
        output_dir,
        write_endpoint=False,
    )

    result = baseline.run_job(job, output_dir, physical_gpu=0, smoke=True)

    assert result["status"] == "completed"
    assert result["optimizer_update_count"] == 2
    assert result["endpoint_summary_path"] is None
    assert result["endpoint_raw_path"] is None


def test_smoke_cli_routes_gpu_and_benchmark_families(tmp_path, monkeypatch):
    baseline = load_baseline()
    manifest_path, _ = write_baseline_manifest(tmp_path, baseline)
    observed = {}

    def run_smoke(path, physical_gpu, families):
        observed.update(
            path=Path(path), physical_gpu=physical_gpu, families=tuple(families)
        )
        return [{"status": "completed"}]

    monkeypatch.setattr(baseline, "run_smoke", run_smoke, raising=False)

    code = baseline.main(
        [
            "smoke",
            "--manifest",
            str(manifest_path),
            "--gpu",
            "1",
            "--benchmarks",
            "muse,wmdp",
        ]
    )

    assert code == 0
    assert observed == {
        "path": manifest_path,
        "physical_gpu": 1,
        "families": ("muse", "wmdp"),
    }


def test_script_cli_manifest_runs_outside_repository(tmp_path):
    output_root = tmp_path / "baseline"

    result = subprocess.run(
        [
            sys.executable,
            str(BASELINE),
            "manifest",
            "--output-root",
            str(output_root),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "manifest.json").is_file()


def test_script_cli_survives_unrelated_scripts_module_shadow(tmp_path):
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "scripts.py").write_text("value = 'unrelated'\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(shadow)

    result = subprocess.run(
        [sys.executable, str(BASELINE), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
