import importlib.util
import json
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "scripts/run_gu_full_matrix.py"

METHODS = (
    "GradAscent",
    "GradDiff",
    "NPO",
    "DPO",
    "SimNPO",
    "RMU",
    "UNDIAL",
    "CEU",
    "WGA",
    "SatImp",
    "PDU",
)
BENCHMARKS = (
    "tofu_forget01",
    "tofu_forget05",
    "tofu_forget10",
    "muse_news",
    "muse_books",
    "wmdp_cyber",
)


def load_registry():
    spec = importlib.util.spec_from_file_location("run_gu_full_matrix", REGISTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_zero_manifest_is_json_safe_complete_and_sorted():
    manifest = load_registry().build_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["protocol"] == "gu_full_matrix_20260724"
    assert manifest["stage"] == "seed0"
    assert manifest["seed"] == 0
    assert len(manifest["jobs"]) == 63
    assert len(manifest["not_applicable"]) == 3
    assert [job["id"] for job in manifest["jobs"]] == sorted(
        job["id"] for job in manifest["jobs"]
    )
    assert manifest["not_applicable"] == sorted(
        manifest["not_applicable"], key=lambda item: (item["method"], item["benchmark"])
    )
    assert json.loads(json.dumps(manifest)) == manifest


def test_compatibility_matrix_has_the_exact_methods_and_settings():
    jobs = load_registry().build_manifest()["jobs"]
    by_method = Counter(job["method"] for job in jobs)
    by_benchmark = Counter(job["benchmark"] for job in jobs)

    assert set(by_method) == set(METHODS)
    assert set(by_benchmark) == set(BENCHMARKS)
    assert by_method == {
        "GradAscent": 6,
        "GradDiff": 6,
        "NPO": 6,
        "DPO": 3,
        "SimNPO": 6,
        "RMU": 6,
        "UNDIAL": 6,
        "CEU": 6,
        "WGA": 6,
        "SatImp": 6,
        "PDU": 6,
    }
    assert by_benchmark == {
        "tofu_forget01": 11,
        "tofu_forget05": 11,
        "tofu_forget10": 11,
        "muse_news": 10,
        "muse_books": 10,
        "wmdp_cyber": 10,
    }


def test_dpo_only_uses_shipped_tofu_preference_pairs():
    manifest = load_registry().build_manifest()
    dpo_jobs = [job for job in manifest["jobs"] if job["method"] == "DPO"]

    assert {job["benchmark"] for job in dpo_jobs} == {
        "tofu_forget01",
        "tofu_forget05",
        "tofu_forget10",
    }
    assert {job["experiment_config"] for job in dpo_jobs} == {"unlearn/tofu/idk"}
    assert manifest["not_applicable"] == [
        {
            "method": "DPO",
            "benchmark": "muse_books",
            "status": "not_applicable",
            "reason": "missing_shipped_preference_pairs",
        },
        {
            "method": "DPO",
            "benchmark": "muse_news",
            "status": "not_applicable",
            "reason": "missing_shipped_preference_pairs",
        },
        {
            "method": "DPO",
            "benchmark": "wmdp_cyber",
            "status": "not_applicable",
            "reason": "missing_shipped_preference_pairs",
        },
    ]


def test_jobs_have_unique_safe_ids_and_relative_output_directories():
    jobs = load_registry().build_manifest()["jobs"]
    ids = [job["id"] for job in jobs]

    assert len(ids) == len(set(ids))
    for job in jobs:
        expected_id = f'{job["method"]}__{job["benchmark"]}__seed{job["seed"]}'
        assert job["id"] == expected_id
        assert "/" not in job["id"] and "\\" not in job["id"]
        assert job["output_dir"] == f"jobs/{expected_id}"
        output_dir = PurePosixPath(job["output_dir"])
        assert not output_dir.is_absolute()
        assert ".." not in output_dir.parts
        assert job["status"] == "pending"


def test_benchmarks_resolve_exact_splits_models_evaluators_and_regexes():
    jobs = load_registry().build_manifest()["jobs"]
    settings = {}
    for job in jobs:
        settings.setdefault(
            job["benchmark"],
            {
                key: job[key]
                for key in (
                    "split",
                    "retain_split",
                    "holdout_split",
                    "experiment_config",
                    "model",
                    "evaluator_kind",
                    "selected_parameter_regex",
                )
                if key in job
            },
        )

    tofu_regex = r"model[.]layers[.](29|30|31)[.]mlp[.]down_proj[.]weight"
    seven_b_regex = r"model[.]layers[.](5|6|7)[.]mlp[.]down_proj[.]weight"
    tofu_model = {
        "config": "Llama-3.1-8B-Instruct",
        "pretrained_model_name_or_path": (
            "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
        ),
    }
    assert settings == {
        "tofu_forget01": {
            "split": "forget01",
            "retain_split": "retain99",
            "holdout_split": "holdout01",
            "experiment_config": "unlearn/tofu/default",
            "model": tofu_model,
            "evaluator_kind": "tofu",
            "selected_parameter_regex": tofu_regex,
        },
        "tofu_forget05": {
            "split": "forget05",
            "retain_split": "retain95",
            "holdout_split": "holdout05",
            "experiment_config": "unlearn/tofu/default",
            "model": tofu_model,
            "evaluator_kind": "tofu",
            "selected_parameter_regex": tofu_regex,
        },
        "tofu_forget10": {
            "split": "forget10",
            "retain_split": "retain90",
            "holdout_split": "holdout10",
            "experiment_config": "unlearn/tofu/default",
            "model": tofu_model,
            "evaluator_kind": "tofu",
            "selected_parameter_regex": tofu_regex,
        },
        "muse_news": {
            "split": "News",
            "retain_split": "retain1",
            "experiment_config": "unlearn/muse/default",
            "model": {
                "config": "Llama-2-7b-hf",
                "pretrained_model_name_or_path": "muse-bench/MUSE-News_target",
            },
            "evaluator_kind": "muse",
            "selected_parameter_regex": seven_b_regex,
        },
        "muse_books": {
            "split": "Books",
            "retain_split": "retain1",
            "experiment_config": "unlearn/muse/default",
            "model": {
                "config": "Llama-2-7b-hf",
                "pretrained_model_name_or_path": "muse-bench/MUSE-Books_target",
            },
            "evaluator_kind": "muse",
            "selected_parameter_regex": seven_b_regex,
        },
        "wmdp_cyber": {
            "split": "cyber",
            "experiment_config": "unlearn/wmdp/default",
            "model": {
                "config": "zephyr-7b-beta",
                "pretrained_model_name_or_path": "HuggingFaceH4/zephyr-7b-beta",
            },
            "evaluator_kind": "lm_eval",
            "selected_parameter_regex": seven_b_regex,
        },
    }


def test_every_job_keeps_its_shipped_trainer_and_unpinned_provenance_explicit():
    jobs = load_registry().build_manifest()["jobs"]

    assert all(job["trainer_config"] == job["method"] for job in jobs)
    assert all(
        job["provenance"]
        == {
            "model_revision": None,
            "tokenizer_revision": None,
            "dataset_revision": None,
            "local_corpus_sha256": None,
        }
        for job in jobs
    )


def test_seed_variation_is_deterministic_and_never_duplicates_jobs():
    registry = load_registry()
    first = registry.build_manifest(seed=2)
    second = registry.build_manifest(seed=2)

    assert first == second
    assert first["stage"] == "seed2"
    assert first["seed"] == 2
    assert {job["seed"] for job in first["jobs"]} == {2}
    assert len({job["id"] for job in first["jobs"]}) == 63
    assert not (
        {job["id"] for job in first["jobs"]}
        & {job["id"] for job in registry.build_manifest(seed=0)["jobs"]}
    )
