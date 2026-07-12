import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
PROCESS_PROFILER = ROOT / "scripts/profile_process_tree.py"
ARM = ROOT / "scripts/orthograd_mvp_arm.sh"
MATRIX = ROOT / "scripts/orthograd_mvp_matrix.sh"
PREPARE_WMDP = ROOT / "scripts/prepare_wmdp_data.sh"
ANALYZER = ROOT / "scripts/analyze_orthograd_mvp.py"


def load_process_profiler():
    spec = importlib.util.spec_from_file_location(
        "profile_process_tree",
        PROCESS_PROFILER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "analyze_orthograd_mvp",
        ANALYZER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_profiler_records_real_child_rss_and_injected_gpu_peak(tmp_path):
    profiler = load_process_profiler()
    gpu_values = iter([100, 120, 160, 140, 130])

    def gpu_query(_gpu):
        return next(gpu_values, 130) * 1024 * 1024

    output = tmp_path / "profile.json"
    command = [
        sys.executable,
        "-c",
        "import time; payload=bytearray(16*1024*1024); time.sleep(0.25)",
    ]

    exit_code, record = profiler.profile_command(
        command,
        output,
        gpu_index=0,
        poll_ms=25,
        gpu_query=gpu_query,
    )

    assert exit_code == 0
    assert record["exit_code"] == 0
    assert record["wall_seconds"] >= 0.2
    assert record["peak_process_tree_rss_bytes"] >= 16 * 1024 * 1024
    assert record["gpu_baseline_used_bytes"] == 100 * 1024 * 1024
    assert record["gpu_peak_used_bytes"] == 160 * 1024 * 1024
    assert record["gpu_peak_delta_bytes"] == 60 * 1024 * 1024
    assert record["sample_count"] >= 2
    assert json.loads(output.read_text()) == record
    assert output.read_text().endswith("\n")


def test_process_profiler_persists_and_propagates_child_failure(tmp_path):
    profiler = load_process_profiler()
    output = tmp_path / "failed.json"

    exit_code, record = profiler.profile_command(
        [sys.executable, "-c", "raise SystemExit(7)"],
        output,
        gpu_index=0,
        poll_ms=10,
        gpu_query=lambda _gpu: 0,
    )

    assert exit_code == 7
    assert record["exit_code"] == 7
    assert json.loads(output.read_text())["exit_code"] == 7


def test_gpu_query_parses_nvidia_smi_mib(monkeypatch):
    profiler = load_process_profiler()

    class Result:
        returncode = 0
        stdout = "1234\n"
        stderr = ""

    monkeypatch.setattr(profiler.subprocess, "run", lambda *args, **kwargs: Result())

    assert profiler.query_gpu_used_bytes(1) == 1234 * 1024 * 1024


def test_gpu_query_rejects_malformed_output(monkeypatch):
    profiler = load_process_profiler()

    class Result:
        returncode = 0
        stdout = "not-a-number\n"
        stderr = ""

    monkeypatch.setattr(profiler.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeError, match="memory.used"):
        profiler.query_gpu_used_bytes(0)


def test_arm_has_matched_no_checkpoint_production_contract():
    text = ARM.read_text()

    for token in (
        "trainer.args.gradient_accumulation_steps=1",
        "trainer.args.max_steps=10",
        "trainer.args.learning_rate=1e-5",
        "trainer.args.optim=paged_adamw_32bit",
        "trainer.args.warmup_epochs=0",
        "trainer.args.bf16=true",
        "model.model_args.attn_implementation=flash_attention_2",
        "trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
        "save_model_after_train=false",
        "trainer.args.save_strategy=no",
    ):
        assert token in text
    assert "profile_process_tree.py" in text
    assert "training_resource_profile.json" in text
    assert "process_resource_profile.json" in text


def test_arm_preserves_shared_hf_cache_and_uses_the_public_model_tokenizer():
    text = ARM.read_text()

    assert (
        "export HF_HOME=${ORTHOGRAD_HF_HOME:-${HF_HOME:-/root/.cache/huggingface}}"
        in text
    )
    assert '"model.tokenizer_args.pretrained_model_name_or_path=${base_model}"' in text


def test_arm_maps_four_methods_and_family_memory_contracts():
    text = ARM.read_text()

    for method in ("control", "orthograd", "gu_euclidean", "gu_adam"):
        assert method in text
    assert "trainer=OrthogradUnlearn" in text
    assert "trainer=GeometricUnlearn" in text
    assert "geometric_config.gu_enabled=false" in text
    assert "geometric_config.optimizer_geometry=euclidean" in text
    assert "geometric_config.optimizer_geometry=adam" in text
    assert "orthograd_config.basis_device=cpu" in text
    assert "geometric_config.component_buffer_device=cpu" in text
    assert "per_device_batch_size=4" in text
    assert "per_device_batch_size=2" in text


def test_muse_mvp_eval_contains_only_the_four_selected_native_metrics():
    text = ARM.read_text()
    assert "eval=muse_mvp" in text

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(
            config_name="unlearn.yaml",
            overrides=[
                "experiment=unlearn/muse/default",
                "eval=muse_mvp",
                "task_name=orthograd_muse_metric_test",
            ],
        )

    assert set(config.eval.muse.metrics) == {
        "forget_knowmem_ROUGE",
        "forget_verbmem_ROUGE",
        "privleak",
        "retain_knowmem_ROUGE",
    }


def test_matrix_contains_seven_benchmarks_and_four_sequential_methods():
    text = MATRIX.read_text()

    for benchmark in (
        "tofu01",
        "tofu05",
        "tofu10",
        "muse_news",
        "muse_books",
        "wmdp_cyber",
        "wmdp_bio",
    ):
        assert benchmark in text
    assert "methods=(control orthograd gu_euclidean gu_adam)" in text
    assert 'for benchmark in "${benchmarks[@]}"' in text
    assert 'for method in "${methods[@]}"' in text
    assert "RUN_MANIFEST.tsv" in text


def test_wmdp_preparation_is_pinned_and_requires_authorized_bio_forget():
    text = PREPARE_WMDP.read_text()

    assert "cais/wmdp-corpora" in text
    assert "daf89fa9b618b63a624228061a9cebacca88009c" in text
    assert "WMDP_BIO_FORGET_SOURCE" in text
    assert "WMDP_BIO_FORGET_URL" in text
    assert "sha256" in text.lower()
    assert "provenance.json" in text


def test_launchers_have_valid_bash_syntax():
    for path in (ARM, MATRIX, PREPARE_WMDP):
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_checkpoint_audit_rejects_payloads_and_symlinks(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    result = subprocess.run(
        [str(ARM), "--audit-only", str(clean)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "optimizer.pt").write_bytes(b"not allowed")
    result = subprocess.run(
        [str(ARM), "--audit-only", str(payload)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "checkpoint payload" in result.stderr.lower()

    link_root = tmp_path / "link"
    link_root.mkdir()
    (link_root / "external").symlink_to(clean, target_is_directory=True)
    result = subprocess.run(
        [str(ARM), "--audit-only", str(link_root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "symbolic link" in result.stderr.lower()


DATASETS = (
    "tofu01",
    "tofu05",
    "tofu10",
    "muse_news",
    "muse_books",
    "wmdp_cyber",
    "wmdp_bio",
)
METHODS = ("control", "orthograd", "gu_euclidean", "gu_adam")


def _summary_for(benchmark, offset):
    if benchmark.startswith("tofu"):
        return {
            "retain_extraction_strength": 0.8 + offset,
            "extraction_strength": 0.2 - offset,
            "privleak": -10.0 + offset,
            "model_utility": 0.6 + offset,
        }
    if benchmark.startswith("muse"):
        return {
            "forget_knowmem_ROUGE": 0.4 - offset,
            "forget_verbmem_ROUGE": 0.7 - offset,
            "privleak": -20.0 + offset,
            "retain_knowmem_ROUGE": 0.65 + offset,
        }
    task = "wmdp_cyber" if benchmark.endswith("cyber") else "wmdp_bio"
    return {f"{task}/acc": 0.5 - offset, "mmlu/acc": 0.6 + offset}


def _write_analysis_fixture(root):
    rows = []
    for benchmark in DATASETS:
        for method_index, method in enumerate(METHODS):
            arm = root / benchmark / method
            arm.mkdir(parents=True)
            summary_name = (
                "TOFU_SUMMARY.json"
                if benchmark.startswith("tofu")
                else "MUSE_SUMMARY.json"
                if benchmark.startswith("muse")
                else "LMEval_SUMMARY.json"
            )
            (arm / summary_name).write_text(
                json.dumps(_summary_for(benchmark, 0.01 * method_index)) + "\n"
            )
            config = {
                "task_name": f"{benchmark}_{method}",
                "paths": {"output_dir": str(arm)},
                "model": {"name": benchmark},
                "trainer": {
                    "handler": (
                        "OrthogradUnlearn"
                        if method == "orthograd"
                        else "GeometricUnlearn"
                    ),
                    "args": {
                        "max_steps": 10,
                        "learning_rate": 1e-5,
                        "gradient_accumulation_steps": 1,
                    },
                    "method_args": {
                        "orthograd_config": (
                            {"basis_device": "cpu"} if method == "orthograd" else None
                        ),
                        "geometric_config": {
                            "gu_enabled": method != "control",
                            "optimizer_geometry": (
                                "euclidean"
                                if method in {"orthograd", "gu_euclidean"}
                                else "adam"
                            ),
                            "diagnostics_path": str(arm / "diagnostics.jsonl"),
                            "resource_profile_path": str(
                                arm / "training_resource_profile.json"
                            ),
                        },
                    },
                },
            }
            hydra = arm / ".hydra"
            hydra.mkdir()
            (hydra / "config.yaml").write_text(yaml.safe_dump(config))
            (arm / "training_resource_profile.json").write_text(
                json.dumps(
                    {
                        "record_type": "training_resource_profile",
                        "update_count": 10,
                        "train_wall_seconds": 20.0 + method_index,
                        "first_update_ms": 1100.0,
                        "steady_update_mean_ms": 1000.0 + method_index,
                        "steady_update_p50_ms": 1000.0,
                        "steady_update_p95_ms": 1010.0,
                        "cuda_max_allocated_bytes": 10_000 + method_index,
                        "cuda_max_reserved_bytes": 12_000 + method_index,
                        "phase_totals_ms": {"projection_writeback": method_index},
                        "phase_counts": {"projection_writeback": 10},
                    }
                )
                + "\n"
            )
            (arm / "process_resource_profile.json").write_text(
                json.dumps(
                    {
                        "record_type": "process_tree_resource_profile",
                        "exit_code": 0,
                        "poll_ms": 100,
                        "sample_count": 20,
                        "wall_seconds": 30.0 + method_index,
                        "peak_process_tree_rss_bytes": 20_000 + method_index,
                        "gpu_baseline_used_bytes": 100,
                        "gpu_peak_used_bytes": 20_100 + method_index,
                        "gpu_peak_delta_bytes": 20_000 + method_index,
                    }
                )
                + "\n"
            )
            if method != "control":
                records = []
                for step in range(1, 11):
                    if method == "orthograd":
                        records.append(
                            {
                                "record_type": "orthograd_geometry",
                                "update_step": step,
                                "effective_rank": 2,
                                "requested_rank": 2,
                                "basis_bytes": 8000,
                                "estimated_projection_flops": 4000,
                                "preserved_signal_ratio": 0.9,
                                "maximum_relative_residual": 1e-7,
                            }
                        )
                    else:
                        records.append(
                            {
                                "record_type": "geometry",
                                "update_step": step,
                                "relative_orthogonality_residual": 1e-7,
                                "optimizer_geometry": (
                                    "euclidean"
                                    if method == "gu_euclidean"
                                    else "paged_adamw_32bit"
                                ),
                            }
                        )
                (arm / "diagnostics.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
            if benchmark.startswith("wmdp"):
                (arm / "wmdp_provenance.json").write_text(
                    json.dumps(
                        {
                            "dataset": "cais/wmdp-corpora",
                            "revision": "daf89fa9b618b63a624228061a9cebacca88009c",
                            "files": {
                                "forget": {"sha256": "a" * 64},
                                "retain": {"sha256": "b" * 64},
                            },
                        }
                    )
                    + "\n"
                )
            rows.append(
                [
                    benchmark,
                    method,
                    "0",
                    "start",
                    "end",
                    "0",
                    f"run {benchmark} {method}",
                    str(arm),
                ]
            )
    header = [
        "benchmark",
        "method",
        "gpu",
        "start_utc",
        "end_utc",
        "exit_code",
        "command",
        "arm_dir",
    ]
    (root / "RUN_MANIFEST.tsv").write_text(
        "\t".join(header) + "\n" + "".join("\t".join(row) + "\n" for row in rows)
    )


def test_analyzer_emits_family_specific_tables_and_overheads(tmp_path):
    analyzer = load_analyzer()
    root = tmp_path / "matrix"
    root.mkdir()
    _write_analysis_fixture(root)

    result = analyzer.analyze_matrix(root)
    markdown = analyzer.render_markdown(result)

    assert set(result["benchmarks"]) == set(DATASETS)
    assert result["benchmarks"]["tofu01"]["methods"]["control"]["metrics"][
        "ES Re."
    ] == pytest.approx(0.8)
    assert result["benchmarks"]["muse_news"]["methods"]["control"]["metrics"][
        "Forget KnowMem"
    ] == pytest.approx(0.4)
    assert result["benchmarks"]["wmdp_bio"]["methods"]["control"]["metrics"][
        "WMDP accuracy"
    ] == pytest.approx(0.5)
    overhead = result["benchmarks"]["tofu01"]["methods"]["orthograd"]["overhead"]
    assert overhead["wall_clock"] == pytest.approx(31.0 / 30.0 - 1.0)
    assert math.isfinite(overhead["allocated"])
    assert "| SimNPO (base) |" in markdown
    assert "| + OrthoGrad-style (per-sample QR, Euclidean) |" in markdown
    assert "| + GU, Euclidean metric (Table 5) |" in markdown
    assert "| + GU, Adam metric (ours) |" in markdown
    assert "Forget KnowMem ↓" in markdown
    assert "WMDP accuracy ↓" in markdown


def test_analyzer_rejects_config_drift_missing_metrics_and_payloads(tmp_path):
    analyzer = load_analyzer()
    root = tmp_path / "matrix"
    root.mkdir()
    _write_analysis_fixture(root)

    config_path = root / "tofu05" / "orthograd" / ".hydra" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["trainer"]["args"]["learning_rate"] = 2e-5
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="configuration drift"):
        analyzer.analyze_matrix(root)

    config["trainer"]["args"]["learning_rate"] = 1e-5
    config_path.write_text(yaml.safe_dump(config))
    summary = root / "muse_books" / "gu_adam" / "MUSE_SUMMARY.json"
    summary.write_text(json.dumps({"forget_knowmem_ROUGE": 0.1}) + "\n")
    with pytest.raises(ValueError, match="metric keys"):
        analyzer.analyze_matrix(root)

    summary.write_text(json.dumps(_summary_for("muse_books", 0.03)) + "\n")
    (root / "wmdp_bio" / "gu_adam" / "optimizer.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checkpoint payload"):
        analyzer.analyze_matrix(root)


def test_analyzer_writes_byte_deterministic_json_and_markdown(tmp_path):
    analyzer = load_analyzer()
    root = tmp_path / "matrix"
    root.mkdir()
    _write_analysis_fixture(root)
    result = analyzer.analyze_matrix(root)
    markdown = tmp_path / "report.md"
    output_json = tmp_path / "analysis.json"

    analyzer.write_outputs(result, markdown, output_json)
    first = (markdown.read_bytes(), output_json.read_bytes())
    analyzer.write_outputs(result, markdown, output_json)
    second = (markdown.read_bytes(), output_json.read_bytes())

    assert first == second
    assert json.loads(output_json.read_text())["provenance"]["manifest_rows"] == 28
    assert markdown.read_text().endswith("\n")
