import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "scripts/run_gu_full_matrix.py"
WMDP_CORPUS_ROOT = Path("/workspace/re/GU/geo-unlearning/data/wmdp/wmdp-corpora")

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
    assert manifest["stage"] == "stage1"
    assert manifest["seed"] == 0
    assert len(manifest["jobs"]) == 60
    assert len(manifest["not_applicable"]) == 6
    assert [job["job_id"] for job in manifest["jobs"]] == sorted(
        job["job_id"] for job in manifest["jobs"]
    )
    assert manifest["not_applicable"] == sorted(
        manifest["not_applicable"], key=lambda item: (item["method"], item["benchmark"])
    )
    assert json.loads(json.dumps(manifest)) == manifest


def test_compatibility_matrix_has_the_exact_methods_and_settings():
    registry = load_registry()
    jobs = registry.build_manifest()["jobs"]
    by_method = Counter(job["method"] for job in jobs)
    by_benchmark = Counter(job["benchmark"] for job in jobs)

    assert registry.METHODS == METHODS
    assert set(by_method) == set(METHODS) - {"DPO"}
    assert set(by_benchmark) == set(BENCHMARKS)
    assert by_method == {
        "GradAscent": 6,
        "GradDiff": 6,
        "NPO": 6,
        "SimNPO": 6,
        "RMU": 6,
        "UNDIAL": 6,
        "CEU": 6,
        "WGA": 6,
        "SatImp": 6,
        "PDU": 6,
    }
    assert by_benchmark == {
        "tofu_forget01": 10,
        "tofu_forget05": 10,
        "tofu_forget10": 10,
        "muse_news": 10,
        "muse_books": 10,
        "wmdp_cyber": 10,
    }


def test_dpo_is_not_compatible_without_the_shipped_idk_artifact():
    manifest = load_registry().build_manifest()
    dpo_jobs = [job for job in manifest["jobs"] if job["method"] == "DPO"]

    assert dpo_jobs == []
    assert manifest["not_applicable"] == [
        {
            "method": "DPO",
            "benchmark": "muse_books",
            "status": "not_applicable",
            "reason": "missing_shipped_idk_artifact",
            "provenance": {
                "required_artifact": {"path": "./data/idk.jsonl", "status": "missing"}
            },
        },
        {
            "method": "DPO",
            "benchmark": "muse_news",
            "status": "not_applicable",
            "reason": "missing_shipped_idk_artifact",
            "provenance": {
                "required_artifact": {"path": "./data/idk.jsonl", "status": "missing"}
            },
        },
        {
            "method": "DPO",
            "benchmark": "tofu_forget01",
            "status": "not_applicable",
            "reason": "missing_shipped_idk_artifact",
            "provenance": {
                "required_artifact": {"path": "./data/idk.jsonl", "status": "missing"}
            },
        },
        {
            "method": "DPO",
            "benchmark": "tofu_forget05",
            "status": "not_applicable",
            "reason": "missing_shipped_idk_artifact",
            "provenance": {
                "required_artifact": {"path": "./data/idk.jsonl", "status": "missing"}
            },
        },
        {
            "method": "DPO",
            "benchmark": "tofu_forget10",
            "status": "not_applicable",
            "reason": "missing_shipped_idk_artifact",
            "provenance": {
                "required_artifact": {"path": "./data/idk.jsonl", "status": "missing"}
            },
        },
        {
            "method": "DPO",
            "benchmark": "wmdp_cyber",
            "status": "not_applicable",
            "reason": "missing_shipped_idk_artifact",
            "provenance": {
                "required_artifact": {"path": "./data/idk.jsonl", "status": "missing"}
            },
        },
    ]
    assert "sha" not in json.dumps(manifest["not_applicable"])


def test_jobs_have_unique_safe_ids_and_relative_output_directories():
    jobs = load_registry().build_manifest()["jobs"]
    ids = [job["job_id"] for job in jobs]

    assert len(ids) == len(set(ids))
    for job in jobs:
        expected_id = f'{job["method"]}__{job["benchmark"]}__seed{job["seed"]}'
        assert job["job_id"] == expected_id
        assert "id" not in job
        assert "/" not in job["job_id"] and "\\" not in job["job_id"]
        assert job["output_dir"] == f"jobs/{expected_id}"
        output_dir = PurePosixPath(job["output_dir"])
        assert not output_dir.is_absolute()
        assert ".." not in output_dir.parts
        assert job["status"] == "pending"
        assert job["stage"] == "stage1"


def test_benchmarks_resolve_exact_splits_models_evaluators_and_regexes():
    jobs = load_registry().build_manifest()["jobs"]
    tofu_regex = r"model[.]layers[.](29|30|31)[.]mlp[.]down_proj[.]weight"
    seven_b_regex = r"model[.]layers[.](5|6|7)[.]mlp[.]down_proj[.]weight"
    tofu_model = {
        "config": "Llama-3.1-8B-Instruct",
        "pretrained_model_name_or_path": (
            "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
        ),
    }
    expected_settings = {
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
    keys = (
        "split",
        "retain_split",
        "holdout_split",
        "experiment_config",
        "model",
        "evaluator_kind",
        "selected_parameter_regex",
    )
    for job in jobs:
        assert {key: job[key] for key in keys if key in job} == expected_settings[
            job["benchmark"]
        ]


def test_every_job_keeps_its_shipped_trainer_and_pinned_provenance_explicit():
    jobs = load_registry().build_manifest()["jobs"]

    assert all(job["trainer_config"] == job["method"] for job in jobs)
    assert "null" not in json.dumps([job["provenance"] for job in jobs])

    tofu_provenance = {
        "model": {
            "artifact": "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
            "revision": "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e",
        },
        "tokenizer": {
            "artifact": "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
            "revision": "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e",
        },
        "dataset": {
            "artifact": "locuslab/TOFU",
            "revision": "324592d84ae4f482ac7249b9285c2ecdb53e3a68",
        },
        "gibberish_classifier": {
            "artifact": "madhurjindal/autonlp-Gibberish-Detector-492513457",
            "revision": "76672dd7d3575f68ab980705bcec975cc62de71c",
        },
    }
    expected_provenance = {
        "tofu_forget01": tofu_provenance,
        "tofu_forget05": tofu_provenance,
        "tofu_forget10": tofu_provenance,
        "muse_news": {
            "model": {
                "artifact": "muse-bench/MUSE-News_target",
                "revision": "a2f39769e9a0b98ec1cdd12f65e9962502208935",
            },
            "tokenizer": {
                "artifact": "NousResearch/Llama-2-7b-hf",
                "revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
            },
            "dataset": {
                "artifact": "muse-bench/MUSE-News",
                "revision": "506bd5b150b92814d45e4404a82f120ab2d748bf",
            },
            "reference_model": {
                "artifact": "muse-bench/MUSE-News_retrain",
                "revision": "324ef49ee0a038078aba7d8de831edf57235c9b3",
            },
            "gibberish_classifier": {
                "artifact": "madhurjindal/autonlp-Gibberish-Detector-492513457",
                "revision": "76672dd7d3575f68ab980705bcec975cc62de71c",
            },
        },
        "muse_books": {
            "model": {
                "artifact": "muse-bench/MUSE-Books_target",
                "revision": "c8dd3fb23a726762ec66d277933c7cff6767f3c2",
            },
            "tokenizer": {
                "artifact": "NousResearch/Llama-2-7b-hf",
                "revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
            },
            "dataset": {
                "artifact": "muse-bench/MUSE-Books",
                "revision": "051ba90319e920d410d87cfdbd61f25843c1b892",
            },
            "reference_model": {
                "artifact": "muse-bench/MUSE-Books_retrain",
                "revision": "1d67430e4e8bdf2a65823740e909792519175ac2",
            },
            "gibberish_classifier": {
                "artifact": "madhurjindal/autonlp-Gibberish-Detector-492513457",
                "revision": "76672dd7d3575f68ab980705bcec975cc62de71c",
            },
        },
        "wmdp_cyber": {
            "model": {
                "artifact": "HuggingFaceH4/zephyr-7b-beta",
                "revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c",
            },
            "tokenizer": {
                "artifact": "HuggingFaceH4/zephyr-7b-beta",
                "revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c",
            },
            "forget_corpus": {
                "path": str(WMDP_CORPUS_ROOT / "cyber-forget-corpus.jsonl"),
                "size_bytes": 21_580_792,
                "sha256": (
                    "b5d339ed7f42a9e0dfc00708e516b288"
                    "363a87512ec9cbdad8703f0ea8f5ea9a"
                ),
                "upstream": (
                    "cais/wmdp-corpora@daf89fa9b618b63a624228061a9cebacca88009c"
                ),
            },
            "retain_corpus": {
                "path": str(WMDP_CORPUS_ROOT / "cyber-retain-corpus.jsonl"),
                "size_bytes": 62_240_540,
                "sha256": (
                    "e3164a1402c381cb50b104fd06a91b50"
                    "68ec650ee9b4bd2f119716ecded2add8"
                ),
                "upstream": (
                    "cais/wmdp-corpora@daf89fa9b618b63a624228061a9cebacca88009c"
                ),
            },
            "utility_corpus": {
                "artifact": "wikitext",
                "subset": "wikitext-2-raw-v1",
                "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            },
            "evaluation_datasets": {
                "wmdp_cyber": {
                    "artifact": "cais/wmdp",
                    "revision": "7125571f22f032c56415e7980f48d877dd830ff8",
                    "cache_builder_sha": ("7125571f22f032c56415e7980f48d877dd830ff8"),
                },
                "mmlu": {
                    "artifact": "hails/mmlu_no_train",
                    "revision": "b2e1ec9aa795adafe68e8e983248dbd4b52a1c60",
                    "cache_builder_sha": ("b2e1ec9aa795adafe68e8e983248dbd4b52a1c60"),
                },
            },
        },
    }

    hex40 = re.compile(r"[0-9a-f]{40}")
    hex64 = re.compile(r"[0-9a-f]{64}")
    for job in jobs:
        benchmark_provenance = job["provenance"]
        assert benchmark_provenance == expected_provenance[job["benchmark"]]
        for name, source in benchmark_provenance.items():
            if "revision" in source:
                assert hex40.fullmatch(source["revision"]), (name, source)
            if "sha256" in source:
                assert hex64.fullmatch(source["sha256"]), (name, source)


def test_seed_zero_manifest_is_deterministic_and_never_duplicates_jobs():
    registry = load_registry()
    first = registry.build_manifest(seed=0)
    second = registry.build_manifest(seed=0)

    assert first == second
    assert first["stage"] == "stage1"
    assert first["seed"] == 0
    assert {job["seed"] for job in first["jobs"]} == {0}
    assert {job["stage"] for job in first["jobs"]} == {"stage1"}
    assert len({job["job_id"] for job in first["jobs"]}) == 60


@pytest.mark.parametrize(
    "seed",
    ["0", Path("seed0"), 0.0, 1.5, True, False, -1, 1, 2, 3],
)
def test_manifest_rejects_invalid_seed_before_building_job_ids(seed):
    with pytest.raises((TypeError, ValueError)):
        load_registry().build_manifest(seed=seed)


def test_wmdp_corpus_paths_are_runtime_reachable_and_content_addressed():
    wmdp_jobs = [
        job
        for job in load_registry().build_manifest()["jobs"]
        if job["benchmark"] == "wmdp_cyber"
    ]

    assert len(wmdp_jobs) == 10
    observed_hashes = {}
    for job in wmdp_jobs:
        for corpus_name in ("forget_corpus", "retain_corpus"):
            source = job["provenance"][corpus_name]
            path = Path(source["path"])
            assert path.is_absolute()
            assert path.parent == WMDP_CORPUS_ROOT
            assert path.is_file()
            assert path.stat().st_size == source["size_bytes"]
            if path not in observed_hashes:
                observed_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert observed_hashes[path] == source["sha256"]


def test_wmdp_evaluation_dataset_provenance_matches_cached_builders():
    wmdp_job = next(
        job
        for job in load_registry().build_manifest()["jobs"]
        if job["benchmark"] == "wmdp_cyber"
    )
    datasets = wmdp_job["provenance"]["evaluation_datasets"]

    assert datasets == {
        "wmdp_cyber": {
            "artifact": "cais/wmdp",
            "revision": "7125571f22f032c56415e7980f48d877dd830ff8",
            "cache_builder_sha": "7125571f22f032c56415e7980f48d877dd830ff8",
        },
        "mmlu": {
            "artifact": "hails/mmlu_no_train",
            "revision": "b2e1ec9aa795adafe68e8e983248dbd4b52a1c60",
            "cache_builder_sha": "b2e1ec9aa795adafe68e8e983248dbd4b52a1c60",
        },
    }

    cache_root = Path("/root/.cache/huggingface/datasets")
    wmdp_builder = (
        cache_root
        / "cais___wmdp"
        / "wmdp-cyber"
        / "0.0.0"
        / datasets["wmdp_cyber"]["cache_builder_sha"]
    )
    assert wmdp_builder.is_dir()

    mmlu_builders = sorted(
        (cache_root / "hails___mmlu_no_train").glob(
            f"*/0.0.0/{datasets['mmlu']['cache_builder_sha']}"
        )
    )
    assert mmlu_builders
    assert all(path.is_dir() for path in mmlu_builders)
