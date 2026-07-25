import fcntl
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from collections import Counter
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

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
            "experiment_config": "unlearn/tofu/gu_matrix_8b",
            "model": tofu_model,
            "evaluator_kind": "tofu",
            "selected_parameter_regex": tofu_regex,
        },
        "tofu_forget05": {
            "split": "forget05",
            "retain_split": "retain95",
            "holdout_split": "holdout05",
            "experiment_config": "unlearn/tofu/gu_matrix_8b",
            "model": tofu_model,
            "evaluator_kind": "tofu",
            "selected_parameter_regex": tofu_regex,
        },
        "tofu_forget10": {
            "split": "forget10",
            "retain_split": "retain90",
            "holdout_split": "holdout10",
            "experiment_config": "unlearn/tofu/gu_matrix_8b",
            "model": tofu_model,
            "evaluator_kind": "tofu",
            "selected_parameter_regex": tofu_regex,
        },
        "muse_news": {
            "split": "News",
            "retain_split": "retain1",
            "experiment_config": "unlearn/muse/gu_matrix",
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
            "experiment_config": "unlearn/muse/gu_matrix",
            "model": {
                "config": "Llama-2-7b-hf",
                "pretrained_model_name_or_path": "muse-bench/MUSE-Books_target",
            },
            "evaluator_kind": "muse",
            "selected_parameter_regex": seven_b_regex,
        },
        "wmdp_cyber": {
            "split": "cyber",
            "experiment_config": "unlearn/wmdp/gu_matrix_cyber",
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


TOFU_DATASET_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"
GIBBERISH_REVISION = "76672dd7d3575f68ab980705bcec975cc62de71c"
RETAIN_LOG_HASHES = {
    "tofu_forget01": "d307bb83ea3e3409fd01cc50908501ea16aba49e9ed6a779f8ceb75ec2e5503f",
    "tofu_forget05": "d94b259bc878e1a38f97cc0fac6937147d09d5958cd24d7fe0437d4be7d3a29f",
    "tofu_forget10": "aa4192bc0e60ee97f47aa514e39510e70f3a8cdc8aa6330591a3d7a0f88e95f2",
    "muse_news": "11f8c9338f955a9fc1ce4148daa0fbd2867807c648a43f99dc240442d2135ec2",
    "muse_books": "502d38cd7da4fda841e4ee411b488cce3be0d882a1e2c07a0117a83045be8b76",
}
RUNTIME_ROOTS = {
    "tofu": Path("/dev/shm/gu-matrix-tofu"),
    "muse_news": Path("/dev/shm/gu-matrix-muse-news"),
    "muse_books": Path("/dev/shm/gu-matrix-muse-books"),
    "wmdp": Path("/dev/shm/gu-matrix-wmdp"),
}

def benchmark_family(job):
    return job["benchmark"].split("_", 1)[0]


def compose_command(command):
    from hydra import compose, initialize_config_dir

    config_index = command.index("--config-name=unlearn.yaml")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            config_name="unlearn.yaml",
            overrides=command[config_index + 1 :],
        )


def direct_shipped_config(job, output_dir):
    provenance = job["provenance"]
    return compose_command(
        [
            sys.executable,
            "src/train.py",
            "--config-name=unlearn.yaml",
            f'model={job["model"]["config"]}',
            f'trainer={job["trainer_config"]}',
            f'task_name={job["job_id"]}',
            f"paths.output_dir={output_dir}",
            (
                "model.model_args.pretrained_model_name_or_path="
                f'{provenance["model"]["artifact"]}'
            ),
            f'+model.model_args.revision={provenance["model"]["revision"]}',
            (
                "model.tokenizer_args.pretrained_model_name_or_path="
                f'{provenance["tokenizer"]["artifact"]}'
            ),
            f'+model.tokenizer_args.revision={provenance["tokenizer"]["revision"]}',
        ]
    )


def without_keys(mapping, ignored):
    return {key: value for key, value in mapping.items() if key not in ignored}


def test_all_60_seed_zero_commands_are_exact_and_checkpoint_free(tmp_path):
    registry = load_registry()
    jobs = registry.build_manifest(seed=0)["jobs"]

    assert len(jobs) == 60
    for job in jobs:
        output_dir = tmp_path / job["job_id"]
        command = registry.build_command(job, output_dir)
        assert command[:3] == [
            sys.executable,
            str(ROOT / "src/train.py"),
            "--config-name=unlearn.yaml",
        ]
        assert command.count("save_model_after_train=false") == 1
        assert command.count("trainer.args.save_strategy=no") == 1
        assert command.count("trainer.args.save_only_model=false") == 1
        assert command.count("trainer.args.do_eval=true") == 1
        assert command.count("trainer.args.eval_on_start=false") == 1
        assert command.count("trainer.args.eval_strategy=no") == 1
        assert command.count("trainer.args.report_to=none") == 1
        assert command.count("trainer.args.logging_steps=1") == 1


def test_all_60_commands_hydra_compose_without_missing_or_unresolved_values(tmp_path):
    from omegaconf import OmegaConf

    jobs = load_registry().build_manifest(seed=0)["jobs"]
    for job in jobs:
        config = compose_command(
            load_registry().build_command(job, tmp_path / job["job_id"])
        )
        config.paths.work_dir = str(ROOT)
        OmegaConf.resolve(config)
        rendered = OmegaConf.to_yaml(config, resolve=True)
        assert "???" not in rendered
        assert "${" not in rendered
        assert config.trainer.handler == job["method"]
        assert config.save_model_after_train is False
        assert config.trainer.args.save_strategy == "no"
        assert config.trainer.args.save_only_model is False
        assert config.trainer.args.do_eval is True
        assert config.trainer.args.eval_on_start is False
        assert config.trainer.args.eval_strategy == "no"
        assert config.trainer.args.report_to == "none"
        assert config.trainer.args.seed == config.trainer.args.data_seed == 0
        assert dict(config.trainer.method_args.gu) == {
            "enabled": True,
            "parameter_regex": [job["selected_parameter_regex"]],
            "retain_history_rank": 8,
            "projection_eps": pytest.approx(1.0e-6),
            "retain_filter": "first_order",
            "retain_budget": pytest.approx(1.0e-4),
            "backtracking_scales": [1.0, 0.5, 0.25, 0.125],
            "diagnostics_path": "gu_diagnostics.jsonl",
        }


def test_non_rmu_commands_keep_direct_shipped_model_and_trainer_defaults(tmp_path):
    from omegaconf import OmegaConf

    jobs = load_registry().build_manifest(seed=0)["jobs"]
    config_paths = (
        ROOT / "configs/experiment/unlearn/tofu/gu_matrix_8b.yaml",
        ROOT / "configs/experiment/unlearn/muse/gu_matrix.yaml",
        ROOT / "configs/experiment/unlearn/wmdp/gu_matrix_cyber.yaml",
    )
    for path in config_paths:
        assert "trainer" not in OmegaConf.load(path)

    representatives = {}
    for job in jobs:
        if job["method"] != "RMU":
            representatives.setdefault((benchmark_family(job), job["method"]), job)

    operational_args = {
        "output_dir",
        "logging_dir",
        "report_to",
        "save_strategy",
        "save_only_model",
        "do_eval",
        "eval_on_start",
        "eval_strategy",
        "logging_steps",
        "seed",
        "data_seed",
    }
    identity_args = {"pretrained_model_name_or_path", "revision"}
    for job in representatives.values():
        output_dir = tmp_path / job["job_id"]
        actual = compose_command(load_registry().build_command(job, output_dir))
        shipped = direct_shipped_config(job, output_dir)
        actual_model = OmegaConf.to_container(actual.model, resolve=True)
        shipped_model = OmegaConf.to_container(shipped.model, resolve=True)
        assert without_keys(actual_model["model_args"], identity_args) == without_keys(
            shipped_model["model_args"], identity_args
        )
        assert without_keys(
            actual_model["tokenizer_args"], identity_args
        ) == without_keys(shipped_model["tokenizer_args"], identity_args)
        actual_trainer = OmegaConf.to_container(actual.trainer.args, resolve=True)
        shipped_trainer = OmegaConf.to_container(shipped.trainer.args, resolve=True)
        assert without_keys(actual_trainer, operational_args) == without_keys(
            shipped_trainer, operational_args
        )

    for job in jobs:
        if job["method"] == "RMU":
            continue
        command = load_registry().build_command(job, tmp_path / job["job_id"])
        assert not any(
            argument.startswith(
                (
                    "trainer.args.learning_rate=",
                    "trainer.args.num_train_epochs=",
                    "trainer.args.optim=",
                    "trainer.args.bf16=",
                    "trainer.args.gradient_checkpointing=",
                )
            )
            for argument in command
        )


def test_rmu_commands_translate_branch_recipes_to_current_rmu_keys(tmp_path):
    jobs = [
        job
        for job in load_registry().build_manifest(seed=0)["jobs"]
        if job["method"] == "RMU"
    ]

    assert len(jobs) == 6
    for job in jobs:
        family = benchmark_family(job)
        command = load_registry().build_command(job, tmp_path / job["job_id"])
        config = compose_command(command)
        expected_module = "model[.]layers[.]31" if family == "tofu" else "model[.]layers[.]7"
        expected_alpha = {"tofu": 100.0, "muse": 10.0, "wmdp": 1200.0}[family]
        assert config.trainer.method_args.gamma == pytest.approx(1.0)
        assert config.trainer.method_args.alpha == pytest.approx(expected_alpha)
        assert config.trainer.method_args.retain_loss_type == "EMBED_DIFF"
        assert config.trainer.method_args.steering_coeff == pytest.approx(6.5)
        assert config.trainer.method_args.module_regex == expected_module
        assert list(config.trainer.method_args.trainable_params_regex) == [
            job["selected_parameter_regex"]
        ]
        assert config.trainer.args.gradient_checkpointing is False
        if job["benchmark"] == "muse_books":
            assert config.trainer.args.num_train_epochs == pytest.approx(1.0)
        elif job["benchmark"] == "muse_news":
            assert config.trainer.args.num_train_epochs == pytest.approx(10.0)


def test_tofu_rmu_intentionally_translates_branch_scope_to_common_gu_scope(tmp_path):
    job = next(
        job
        for job in load_registry().build_manifest(seed=0)["jobs"]
        if job["method"] == "RMU" and job["benchmark"] == "tofu_forget01"
    )
    config = compose_command(
        load_registry().build_command(job, tmp_path / job["job_id"])
    )

    translation = config.protocol.rmu_scope_translation
    assert translation.branch_recipe.module_regex == "model.layers.15"
    assert translation.branch_recipe.parameter_regex == (
        "model[.]layers[.](13|14|15)[.]mlp[.]down_proj[.]weight"
    )
    assert translation.matrix_protocol.module_regex == "model.layers.31"
    assert translation.matrix_protocol.parameter_regex == (
        "model[.]layers[.](29|30|31)[.]mlp[.]down_proj[.]weight"
    )
    assert translation.reason == "preserve_gradient_path_to_approved_common_gu_scope"
    assert config.trainer.method_args.module_regex == "model[.]layers[.]31"
    assert list(config.trainer.method_args.gu.parameter_regex) == [
        translation.matrix_protocol.parameter_regex
    ]


def test_tofu_and_muse_rmu_apply_branch_model_runtime_overrides_only(tmp_path):
    jobs = [
        job
        for job in load_registry().build_manifest(seed=0)["jobs"]
        if job["method"] == "RMU" and job["benchmark"] != "wmdp_cyber"
    ]

    assert len(jobs) == 5
    for job in jobs:
        command = load_registry().build_command(job, tmp_path / job["job_id"])
        config = compose_command(command)
        expected_backend = (
            "flash_attention_2"
            if job["benchmark"].startswith("tofu_")
            else "sdpa"
        )
        assert (
            f"model.model_args.attn_implementation={expected_backend}" in command
        )
        assert config.model.model_args.attn_implementation == expected_backend
        assert config.model.model_args.use_cache is False
        assert config.model.model_args.output_attentions is False


def test_wmdp_rmu_uses_expressible_branch_recipe_and_declares_sampling_deviation(
    tmp_path,
):
    from omegaconf import OmegaConf

    dataset_path = ROOT / "configs/data/datasets/WMDP_wikitext_retain.yaml"
    assert dataset_path.is_file()
    dataset = OmegaConf.load(dataset_path)
    assert OmegaConf.to_container(dataset, resolve=True) == {
        "WMDP_wikitext_retain": {
            "handler": "PretrainingDataset",
            "args": {
                "hf_args": {
                    "path": "wikitext",
                    "name": "wikitext-2-raw-v1",
                    "split": "train",
                    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                },
                "text_key": "text",
                "max_length": 512,
            },
        }
    }
    job = next(
        job
        for job in load_registry().build_manifest(seed=0)["jobs"]
        if job["method"] == "RMU" and job["benchmark"] == "wmdp_cyber"
    )
    command = load_registry().build_command(job, tmp_path / job["job_id"])
    config = compose_command(command)

    assert set(config.data.retain) == {"WMDP_wikitext_retain"}
    retain = config.data.retain.WMDP_wikitext_retain.args
    assert retain.hf_args.revision == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert config.data.forget.WMDP_forget.args.max_length == 768
    assert config.model.model_args.attn_implementation == "sdpa"
    assert config.model.model_args.use_cache is False
    assert config.model.model_args.output_attentions is False
    assert config.eval.lm_eval.simple_evaluate_args.batch_size == 8
    assert config.protocol.rmu_sampling.effective == "shipped_deterministic_random"
    assert config.protocol.rmu_sampling.branch_only == "sequential_sampler"
    assert config.protocol.rmu_sampling.claims_branch_sampler_parity is False
    assert "sequential_sampler" not in config.trainer.method_args


def test_pinned_benchmark_configs_resolve_evaluators_and_provenance(tmp_path):
    jobs = load_registry().build_manifest(seed=0)["jobs"]
    representatives = {}
    for job in jobs:
        representatives.setdefault(job["benchmark"], job)

    for benchmark, job in representatives.items():
        config = compose_command(
            load_registry().build_command(job, tmp_path / job["job_id"])
        )
        provenance = job["provenance"]
        assert config.model.model_args.revision == provenance["model"]["revision"]
        assert config.model.tokenizer_args.revision == provenance["tokenizer"]["revision"]
        assert config.protocol.source_provenance.model.artifact == provenance["model"]["artifact"]
        assert config.protocol.source_provenance.model.revision == provenance["model"]["revision"]
        assert config.protocol.source_provenance.tokenizer == provenance["tokenizer"]
        if benchmark.startswith("tofu_"):
            assert set(config.eval) == {"tofu"}
            assert config.data.forget.TOFU_QA_forget.args.hf_args.revision == TOFU_DATASET_REVISION
            assert config.data.retain.TOFU_QA_retain.args.hf_args.revision == TOFU_DATASET_REVISION
            assert config.eval.tofu.metrics.forget_Q_A_gibberish.classifier_model_args.revision == GIBBERISH_REVISION
            assert config.eval.tofu.metrics.forget_Q_A_gibberish.classifier_tokenization_args.revision == GIBBERISH_REVISION
            assert Path(config.retain_logs_path).is_file()
        elif benchmark.startswith("muse_"):
            assert set(config.eval) == {"muse"}
            assert config.data.forget.MUSE_forget.args.hf_args.revision == provenance["dataset"]["revision"]
            assert config.data.retain.MUSE_retain.args.hf_args.revision == provenance["dataset"]["revision"]
            assert config.protocol.source_provenance.reference_model == provenance["reference_model"]
            assert config.eval.muse.metrics.mia_reference.reference_model_path == config.reference_model_snapshot
            assert Path(config.reference_model_snapshot).name == provenance[
                "reference_model"
            ]["revision"]
            assert Path(config.retain_logs_path).is_file()
            assert config.eval.muse.metrics.forget_gibberish.classifier_model_args.revision == GIBBERISH_REVISION
            assert config.eval.muse.metrics.forget_gibberish.classifier_tokenization_args.revision == GIBBERISH_REVISION
        else:
            assert set(config.eval) == {"lm_eval"}
            assert list(config.eval.lm_eval.tasks) == ["wmdp_cyber", "mmlu"]


def iter_mappings(value):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from iter_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_mappings(child)


def assert_metric_hub_references_are_revision_pinned(config):
    metrics = OmegaConf_to_container(config)["eval"]
    observed = []
    for mapping in iter_mappings(metrics):
        for key in ("path", "pretrained_model_name_or_path"):
            artifact = mapping.get(key)
            if isinstance(artifact, str) and artifact.startswith(
                ("locuslab/", "muse-bench/", "madhurjindal/")
            ):
                observed.append(artifact)
                assert re.fullmatch(r"[0-9a-f]{40}", mapping.get("revision", "")), (
                    key,
                    artifact,
                    mapping,
                )
    assert observed


def test_tofu_and_muse_pin_every_enabled_metric_hub_reference(tmp_path):
    jobs = load_registry().build_manifest(seed=0)["jobs"]
    representatives = {}
    for job in jobs:
        if job["benchmark"] != "wmdp_cyber":
            representatives.setdefault(job["benchmark"], job)

    for benchmark, job in representatives.items():
        config = compose_command(
            load_registry().build_command(job, tmp_path / job["job_id"])
        )
        assert_metric_hub_references_are_revision_pinned(config)
        retain_logs = Path(config.retain_logs_path)
        retain_hash = hashlib.sha256(retain_logs.read_bytes()).hexdigest()
        evaluator = config.eval.tofu if benchmark.startswith("tofu_") else config.eval.muse
        assert (
            evaluator.metrics.privleak.reference_logs.retain_model_logs.sha256
            == retain_hash
            == RETAIN_LOG_HASHES[benchmark]
        )
        if benchmark.startswith("muse_"):
            reference = evaluator.metrics.mia_reference
            assert reference.reference_model_path == config.reference_model_snapshot
            assert reference.reference_model_revision == config.reference_model_revision
        gibberish_name = (
            "forget_Q_A_gibberish" if benchmark.startswith("tofu_") else "forget_gibberish"
        )
        gibberish = evaluator.metrics[gibberish_name]
        assert gibberish.classifier_model_args.revision == GIBBERISH_REVISION
        assert gibberish.classifier_tokenization_args.revision == GIBBERISH_REVISION


def test_wmdp_config_is_cyber_only_with_absolute_verified_corpora_and_eval_cache_metadata(tmp_path):
    job = next(
        job
        for job in load_registry().build_manifest(seed=0)["jobs"]
        if job["benchmark"] == "wmdp_cyber"
    )
    config = compose_command(
        load_registry().build_command(job, tmp_path / job["job_id"])
    )
    rendered = json.dumps(OmegaConf_to_container(config)).lower()

    assert not re.search(r"\\bbio(?:logy)?\\b", rendered)
    assert config.data_split == "cyber"
    for corpus_name, data_name, dataset_name in (
        ("forget_corpus", "forget", "WMDP_forget"),
        ("retain_corpus", "retain", "WMDP_retain"),
    ):
        metadata = config.protocol.source_provenance[corpus_name]
        path = Path(metadata.path)
        assert path.is_absolute() and path.is_file()
        assert config.data[data_name][dataset_name].args.hf_args.data_files == str(path)
        assert path.stat().st_size == metadata.size_bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata.sha256
    assert config.protocol.evaluation_pin.pin_mechanism == "verified_offline_cache"
    assert config.protocol.evaluation_pin.datasets_cache == (
        "/dev/shm/gu-matrix-wmdp/datasets"
    )
    assert config.protocol.evaluation_datasets.wmdp_cyber.cache_builder_sha == (
        "7125571f22f032c56415e7980f48d877dd830ff8"
    )
    assert config.protocol.evaluation_datasets.mmlu.cache_builder_sha == (
        "b2e1ec9aa795adafe68e8e983248dbd4b52a1c60"
    )
    assert all(
        "revision" not in mapping
        for mapping in iter_mappings(OmegaConf_to_container(config)["eval"]["lm_eval"])
    )


def test_build_environment_enforces_wmdp_verified_offline_dataset_cache(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/registered/model-cache")
    monkeypatch.setenv("MATRIX_CALLER_VALUE", "preserved")
    registry = load_registry()
    wmdp_job = next(
        job
        for job in registry.build_manifest(seed=0)["jobs"]
        if job["benchmark"] == "wmdp_cyber"
    )
    environment = registry.build_environment(wmdp_job)
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_DATASETS_CACHE"] == "/dev/shm/gu-matrix-wmdp/datasets"
    assert environment["HF_HUB_CACHE"] == "/dev/shm/gu-matrix-wmdp/hub"
    assert environment["HF_HOME"] == "/dev/shm/gu-matrix-wmdp/hub"
    assert environment["HUGGINGFACE_HUB_CACHE"] == "/dev/shm/gu-matrix-wmdp/hub"
    assert environment["TRANSFORMERS_CACHE"] == "/dev/shm/gu-matrix-wmdp/hub"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["MATRIX_CALLER_VALUE"] == "preserved"


def test_every_job_binds_environment_requirements_and_reference_to_one_root():
    registry = load_registry()
    for job in registry.build_manifest(seed=0)["jobs"]:
        family = benchmark_family(job)
        key = job["benchmark"] if family == "muse" else family
        root = RUNTIME_ROOTS[key]
        environment = registry.environment_overrides(job)
        assert environment == {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_DATASETS_CACHE": str(root / "datasets"),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(root / "hub"),
            "HF_HUB_CACHE": str(root / "hub"),
            "HF_HUB_OFFLINE": "1",
            "HUGGINGFACE_HUB_CACHE": str(root / "hub"),
            "TRANSFORMERS_CACHE": str(root / "hub"),
        }
        for requirement in registry.source_requirements(job)["content_requirements"]:
            expected_root = root / (
                "hub" if requirement["kind"] == "hub_snapshot" else "datasets"
            )
            assert Path(requirement["path"]).is_relative_to(expected_root)
        if family == "muse":
            reference_argument = next(
                argument
                for argument in registry.build_command(job, job["output_dir"])
                if argument.startswith("reference_model_snapshot=")
            )
            reference_path = Path(reference_argument.split("=", 1)[1])
            assert reference_path.is_relative_to(root / "hub")


def test_fresh_model_import_and_loaders_use_validated_hub_cache():
    registry = load_registry()
    job = next(
        job
        for job in registry.build_manifest(seed=0)["jobs"]
        if job["benchmark"] == "wmdp_cyber"
    )
    environment = registry.build_environment(job)
    environment["PYTHONPATH"] = str(ROOT / "src")
    script = """
import json
from omegaconf import OmegaConf
import model

captured = {}

class FakeModel:
    @classmethod
    def from_pretrained(cls, **kwargs):
        captured["model_cache_dir"] = kwargs["cache_dir"]
        return object()

class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 1

def fake_tokenizer_from_pretrained(**kwargs):
    captured["tokenizer_cache_dir"] = kwargs["cache_dir"]
    return FakeTokenizer()

model.MODEL_REGISTRY["AutoModelForCausalLM"] = FakeModel
model.AutoTokenizer.from_pretrained = fake_tokenizer_from_pretrained
config = OmegaConf.create({
    "model_args": {
        "pretrained_model_name_or_path": "example/model",
        "torch_dtype": "float32",
    },
    "tokenizer_args": {
        "pretrained_model_name_or_path": "example/tokenizer",
    },
})
model.get_model(config)
print(json.dumps({"hf_home": model.hf_home, **captured}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    hub = str(RUNTIME_ROOTS["wmdp"] / "hub")
    assert json.loads(result.stdout) == {
        "hf_home": hub,
        "model_cache_dir": hub,
        "tokenizer_cache_dir": hub,
    }


def fixture_content_manifest(path, registered_suffixes=None):
    entries = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if registered_suffixes is not None and candidate.suffix not in registered_suffixes:
            continue
        contents = candidate.read_bytes()
        entries.append(
            {
                "path": candidate.relative_to(path).as_posix(),
                "size": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_dataset_cache_manifest_detects_copied_arrow_and_metadata_mutation(tmp_path):
    registry = load_registry()
    source = Path(
        "/dev/shm/ungu-hf-datasets-wmdp/cais___wmdp/wmdp-cyber/0.0.0/"
        "7125571f22f032c56415e7980f48d877dd830ff8"
    )
    copied = tmp_path / "wmdp-cache"
    shutil.copytree(source, copied)
    requirement = {
        "kind": "dataset_cache",
        "source_name": "wmdp_cyber_eval_cache",
        "path": str(copied),
        "content_manifest_sha256": fixture_content_manifest(
            copied, {".arrow", ".json"}
        ),
        "registered_suffixes": [".arrow", ".json"],
    }
    memo = {}

    assert registry.validate_content_requirement(requirement, memo) == requirement[
        "content_manifest_sha256"
    ]
    arrow_or_metadata = next(
        path
        for path in copied.rglob("*")
        if path.is_file() and path.suffix in {".arrow", ".json"}
    )
    arrow_or_metadata.write_bytes(arrow_or_metadata.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="wmdp_cyber_eval_cache"):
        registry.validate_content_requirement(requirement, {})

    shutil.rmtree(copied)
    shutil.copytree(source, copied)
    (copied / "incidental_builder.lock").write_text("ignored")
    assert registry.validate_content_requirement(requirement, {}) == requirement[
        "content_manifest_sha256"
    ]


def test_hub_snapshot_validation_checks_blob_hash_and_symlink_containment(tmp_path):
    registry = load_registry()
    repository = tmp_path / "models--example--tiny"
    blobs = repository / "blobs"
    snapshot = repository / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob_contents = b'{"model_type":"tiny"}'
    blob_name = hashlib.sha256(blob_contents).hexdigest()
    blob = blobs / blob_name
    blob.write_bytes(blob_contents)
    (snapshot / "config.json").symlink_to(Path("../../blobs") / blob_name)
    requirement = {
        "kind": "hub_snapshot",
        "source_name": "example/tiny",
        "path": str(snapshot),
        "revision": "a" * 40,
        "content_manifest_sha256": fixture_content_manifest(snapshot),
    }

    assert registry.validate_content_requirement(requirement, {}) == requirement[
        "content_manifest_sha256"
    ]
    blob.write_bytes(blob_contents + b"tamper")
    with pytest.raises(ValueError, match="example/tiny"):
        registry.validate_content_requirement(requirement, {})

    outside = tmp_path / "outside"
    outside.write_bytes(blob_contents)
    blob.write_bytes(blob_contents)
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(outside)
    with pytest.raises(ValueError, match="example/tiny"):
        registry.validate_content_requirement(requirement, {})


def test_content_fingerprint_memo_is_keyed_by_requirement_json(tmp_path, monkeypatch):
    registry = load_registry()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "dataset_info.json").write_text("{}")
    (cache / "data.arrow").write_bytes(b"arrow")
    requirement = {
        "kind": "dataset_cache",
        "source_name": "shared_cache",
        "path": str(cache),
        "content_manifest_sha256": fixture_content_manifest(
            cache, {".arrow", ".json"}
        ),
        "registered_suffixes": [".arrow", ".json"],
    }
    observed_calls = 0
    original = registry.canonical_directory_fingerprint

    def counting_fingerprint(path, registered_suffixes=None):
        nonlocal observed_calls
        observed_calls += 1
        return original(path, registered_suffixes)

    monkeypatch.setattr(registry, "canonical_directory_fingerprint", counting_fingerprint)
    memo = {}
    for _ in range(60):
        registry.validate_content_requirement(requirement, memo)

    assert observed_calls == 1
    assert len(memo) == 1


def test_source_requirements_register_content_manifests_and_missing_snapshot(monkeypatch):
    registry = load_registry()
    jobs = registry.build_manifest(seed=0)["jobs"]
    representatives = {}
    for job in jobs:
        key = (job["benchmark"], job["method"] == "RMU")
        representatives.setdefault(key, job)

    for job in representatives.values():
        requirements = registry.source_requirements(job)
        assert requirements["model"] == job["provenance"]["model"]
        assert requirements["tokenizer"] == job["provenance"]["tokenizer"]
        content = requirements["content_requirements"]
        assert content
        for requirement in content:
            assert requirement["kind"] in {"dataset_cache", "hub_snapshot"}
            assert requirement["source_name"]
            assert Path(requirement["path"]).is_absolute()
            expected = requirement["content_manifest_sha256"]
            assert re.fullmatch(r"[0-9a-f]{64}", expected)
        if job["benchmark"].startswith(("tofu_", "muse_")):
            retain_log = requirements["retain_log"]
            assert Path(retain_log["path"]).is_file()
            assert retain_log["sha256"] == RETAIN_LOG_HASHES[job["benchmark"]]
        if job["benchmark"] == "wmdp_cyber":
            cache = requirements["evaluation_cache"]
            assert cache["root"] == "/dev/shm/gu-matrix-wmdp/datasets"
            assert cache["builders"]["wmdp_cyber"]["builder_id"] == (
                "7125571f22f032c56415e7980f48d877dd830ff8"
            )
            assert cache["builders"]["mmlu"]["builder_id"] == (
                "b2e1ec9aa795adafe68e8e983248dbd4b52a1c60"
            )
            assert set(requirements["corpora"]) == {"forget", "retain"}
            if job["method"] == "RMU":
                assert requirements["retain_dataset"] == {
                    "artifact": "wikitext",
                    "subset": "wikitext-2-raw-v1",
                    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                    "cache_dir": (
                        "/dev/shm/gu-matrix-wmdp/datasets/wikitext/"
                        "wikitext-2-raw-v1/0.0.0/"
                        "b08601e04326c79dfdd32d625aee71d232d685c3"
                    ),
                }

    books_job = next(job for job in jobs if job["benchmark"] == "muse_books")
    with pytest.raises(ValueError, match="muse-bench/MUSE-Books_target"):
        registry.validate_sources(books_job)

    tampered = json.loads(json.dumps(jobs[0]))
    tampered["provenance"]["model"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="model revision"):
        registry.validate_sources(tampered)

    monkeypatch.setitem(
        registry.RETAIN_LOG_HASHES,
        "tofu_forget01",
        "0" * 64,
    )
    tofu_job = next(job for job in jobs if job["benchmark"] == "tofu_forget01")
    with pytest.raises(ValueError, match="retain log"):
        registry.validate_sources(tofu_job)


def test_manifest_dry_run_stops_on_source_mismatch(monkeypatch):
    registry = load_registry()

    def reject_sources(job, fingerprint_memo=None):
        del fingerprint_memo
        raise ValueError(f'source mismatch: {job["job_id"]}')

    monkeypatch.setattr(registry, "validate_sources", reject_sources, raising=False)
    payload = registry._manifest_dry_run(seed=0)
    assert "source mismatch" in payload["source_validation_error"]
    assert len(payload["jobs"]) == 1
    assert payload["jobs"][0]["fingerprints"] == {}


def test_manifest_dry_run_emits_validation_fingerprints(monkeypatch):
    registry = load_registry()

    monkeypatch.setattr(
        registry,
        "validate_sources",
        lambda job, fingerprint_memo=None: {"shared": "a" * 64},
    )
    payload = registry._manifest_dry_run(seed=0)

    assert "source_validation_error" not in payload
    assert len(payload["jobs"]) == 60
    assert all(job["fingerprints"] == {"shared": "a" * 64} for job in payload["jobs"])


def test_design_discloses_branch_only_rmu_sampling_deviation():
    design = (
        ROOT / "docs/superpowers/specs/2026-07-24-gu-full-matrix-design.md"
    ).read_text()

    assert "branch-compatible recipe values and provenance" in design
    assert "common shipped `RMU` trainer" in design
    assert "`shipped_deterministic_random`" in design
    assert "does not claim sequential-sampler parity" in design


def OmegaConf_to_container(config):
    from omegaconf import OmegaConf

    copied = config.copy()
    copied.paths.work_dir = str(ROOT)
    return OmegaConf.to_container(copied, resolve=True, throw_on_missing=True)


def test_manifest_seed_zero_dry_run_prints_commands_without_creating_outputs(tmp_path):
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = subprocess.run(
        [
            sys.executable,
            str(REGISTRY),
            "manifest",
            "--seed0",
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert "muse-bench/MUSE-Books_target" in payload["source_validation_error"]
    assert len(payload["jobs"]) == 1
    for job in payload["jobs"]:
        assert job["argv"][:3] == [
            sys.executable,
            str(ROOT / "src/train.py"),
            "--config-name=unlearn.yaml",
        ]
        assert job["source_requirements"]["model"] == job["provenance"]["model"]
        assert job["fingerprints"] == {}
        assert job["benchmark"] == "muse_books"
        assert job["environment"] == {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_DATASETS_CACHE": "/dev/shm/gu-matrix-muse-books/datasets",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": "/dev/shm/gu-matrix-muse-books/hub",
            "HF_HUB_CACHE": "/dev/shm/gu-matrix-muse-books/hub",
            "HF_HUB_OFFLINE": "1",
            "HUGGINGFACE_HUB_CACHE": "/dev/shm/gu-matrix-muse-books/hub",
            "TRANSFORMERS_CACHE": "/dev/shm/gu-matrix-muse-books/hub",
        }
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before


JOB_DIAGNOSTIC_FIELDS = {
    "step",
    "objective",
    "selected_parameter_count",
    "proposal_norm",
    "corrected_norm",
    "correction_ratio",
    "constraint_count",
    "active_constraints",
    "max_violation_before",
    "max_violation_after",
    "kkt_residual",
    "projection_tolerance",
    "applied_scale",
    "retain_loss_before",
    "retain_loss_after",
    "zero_step",
    "zero_step_reason",
    "optimizer_state_semantics",
    "projection_seconds",
    "filter_seconds",
}


def job_diagnostic(job, step, **overrides):
    record = {
        "step": step,
        "objective": job["method"],
        "selected_parameter_count": 3,
        "proposal_norm": 2.0,
        "corrected_norm": 1.0,
        "correction_ratio": 0.5,
        "constraint_count": 2,
        "active_constraints": [1],
        "max_violation_before": 0.2,
        "max_violation_after": 5.0e-8,
        "kkt_residual": 1.0e-8,
        "projection_tolerance": 1.0e-6,
        "applied_scale": 0.5,
        "retain_loss_before": None,
        "retain_loss_after": None,
        "zero_step": False,
        "zero_step_reason": None,
        "optimizer_state_semantics": "proposal_state_committed",
        "projection_seconds": 0.01,
        "filter_seconds": 0.02,
    }
    record.update(overrides)
    assert set(record) == JOB_DIAGNOSTIC_FIELDS
    return record


def job_endpoint_prefix(job):
    return {
        "tofu": "TOFU",
        "muse": "MUSE",
        "lm_eval": "LMEval",
    }[job["evaluator_kind"]]


def install_job_subprocess_stub(
    monkeypatch,
    registry,
    job,
    output_dir,
    *,
    returncode=0,
    log_text=None,
    diagnostics=None,
    endpoint_count=1,
    forbidden_path=None,
    extra_files=None,
    extra_dirs=None,
    nvml_results=None,
    tamper_command=False,
):
    if log_text is None:
        log_text = "".join(
            json.dumps(record) + "\n"
            for record in (
                {"loss": 2.0, "grad_norm": 1.0, "epoch": 0.1},
                {"loss": 1.5, "grad_norm": 0.8, "epoch": 0.2},
                {"train_runtime": 3.0, "train_loss": 1.75, "global_step": 2},
            )
        )
    if diagnostics is None:
        diagnostics = [job_diagnostic(job, 1), job_diagnostic(job, 2)]
    if extra_files is None:
        extra_files = {}
    if extra_dirs is None:
        extra_dirs = []
    captured = {"nvidia_commands": [], "sleep_seconds": []}

    class StubPopen:
        pid = 4242

        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            self._polls = 0
            kwargs["stdout"].write(log_text)
            kwargs["stdout"].flush()
            if diagnostics is not False:
                (output_dir / "gu_diagnostics.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in diagnostics)
                )
            prefix = job_endpoint_prefix(job)
            for index in range(endpoint_count):
                eval_dir = output_dir / f"checkpoint-{index + 1}" / "evals"
                eval_dir.mkdir(parents=True)
                (eval_dir / f"{prefix}_SUMMARY.json").write_text(
                    json.dumps({"metric": 0.25})
                )
                (eval_dir / f"{prefix}_EVAL.json").write_text(
                    json.dumps({"metric": {"agg_value": 0.25}})
                )
            if forbidden_path is not None:
                path = output_dir / forbidden_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"forbidden")
            for relative_path, contents in extra_files.items():
                path = output_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            for relative_path in extra_dirs:
                (output_dir / relative_path).mkdir(parents=True, exist_ok=True)
            if tamper_command:
                payload = json.loads((output_dir / "command.json").read_text())
                payload["argv"].append("trainer.args.learning_rate=999")
                (output_dir / "command.json").write_text(json.dumps(payload))

        def poll(self):
            self._polls += 1
            return None if self._polls == 1 else returncode

    if nvml_results is None:
        nvml_results = [
            (0, "4242, 128 MiB\n", ""),
            (0, "4242, 384 MiB\n", ""),
        ]
    nvml_samples = iter(nvml_results)

    def stub_run(argv, **kwargs):
        captured["nvidia_commands"].append((argv, kwargs))
        outcome = next(nvml_samples, (0, "", ""))
        if isinstance(outcome, BaseException):
            raise outcome
        returncode, stdout, stderr = outcome
        return SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    clock_ticks = iter((10.0, 10.25, 11.0, 11.5, 12.0))
    monkeypatch.setattr(
        registry,
        "subprocess",
        SimpleNamespace(Popen=StubPopen, run=stub_run, STDOUT=subprocess.STDOUT),
        raising=False,
    )
    monkeypatch.setattr(
        registry,
        "time",
        SimpleNamespace(
            monotonic=lambda: next(clock_ticks, 12.0),
            perf_counter=lambda: next(clock_ticks, 12.0),
            sleep=lambda seconds: captured["sleep_seconds"].append(seconds),
        ),
        raising=False,
    )
    return captured


def first_matrix_job(registry, evaluator_kind="muse"):
    return next(
        job
        for job in registry.build_manifest(seed=0)["jobs"]
        if job["evaluator_kind"] == evaluator_kind
    )


def test_completed_job_requires_exact_gu_endpoint_and_resource_evidence(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    captured = install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "completed"
    assert result["returncode"] == 0
    assert result["projection_count"] == result["optimizer_update_count"] == 2
    assert result["selected_parameter_changed"] is True
    assert result["zero_step_count"] == 0
    assert result["applied_scale_distribution"] == {"0.5": 2}
    assert result["correction_ratio"] == {"min": 0.5, "max": 0.5, "mean": 0.5}
    assert result["max_violation_after"] == pytest.approx(5.0e-8)
    assert result["peak_nvml_mib"] == 384
    assert result["peak_gpu_memory_mib"] == 384
    assert result["gpu_memory_sample_count"] == 2
    assert result["wall_clock_seconds"] >= 0.0
    assert result["forbidden_artifacts"] == []
    assert result["endpoint_summary_path"].endswith("MUSE_SUMMARY.json")
    assert result["endpoint_raw_path"].endswith("MUSE_EVAL.json")
    assert captured["argv"] == registry.build_command(job, output_dir)
    assert captured["kwargs"]["env"] == registry.build_environment(job)
    assert captured["kwargs"]["stderr"] == subprocess.STDOUT
    assert captured["nvidia_commands"]
    assert captured["nvidia_commands"][0][0] == [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader",
    ]
    monitor_kwargs = captured["nvidia_commands"][0][1]
    assert math.isfinite(monitor_kwargs["timeout"])
    assert monitor_kwargs["timeout"] > 0
    assert captured["sleep_seconds"] == [1.0]
    command_record = json.loads((output_dir / "command.json").read_text())
    assert command_record["argv"] == registry.build_command(job, output_dir)
    assert command_record["environment_overrides"] == registry.environment_overrides(job)
    assert command_record["provenance"] == job["provenance"]
    assert json.loads((output_dir / "JOB_RESULT.json").read_text()) == result


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("method", "NPO"),
        ("benchmark", "wmdp_cyber"),
        ("split", "forget99"),
        ("retain_split", "retain00"),
        ("holdout_split", "holdout99"),
        ("seed", 1),
        ("stage", "stage2"),
        ("trainer_config", "NPO"),
        ("experiment_config", "unlearn/tampered"),
        ("model", {}),
        ("evaluator_kind", "muse"),
        ("selected_parameter_regex", "tampered"),
        ("provenance", {}),
        ("output_dir", "jobs/tampered"),
    ],
)
def test_run_job_rejects_tampered_registry_identity_before_output_or_launch(
    tmp_path,
    monkeypatch,
    field,
    tampered_value,
):
    registry = load_registry()
    job = first_matrix_job(registry, evaluator_kind="tofu")
    tampered = json.loads(json.dumps(job))
    tampered[field] = tampered_value
    output_dir = tmp_path / job["job_id"]
    launched = []

    def forbidden_popen(*args, **kwargs):
        launched.append((args, kwargs))
        raise AssertionError("tampered jobs must not launch")

    monkeypatch.setattr(
        registry,
        "subprocess",
        SimpleNamespace(Popen=forbidden_popen, STDOUT=subprocess.STDOUT),
    )

    with pytest.raises(ValueError, match="registered Stage-1 job"):
        registry.run_job(tampered, output_dir)

    assert launched == []
    assert not output_dir.exists()


def test_run_job_rejects_unknown_job_without_erasing_existing_output(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    unknown = json.loads(json.dumps(job))
    unknown["job_id"] = "unknown__muse_books__seed0"
    output_dir = tmp_path / job["job_id"]
    output_dir.mkdir()
    sentinel = output_dir / "preserve.txt"
    sentinel.write_text("preserve")
    monkeypatch.setattr(
        registry.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unknown jobs must not launch")
        ),
    )

    with pytest.raises(ValueError, match="registered Stage-1 job"):
        registry.run_job(unknown, output_dir)

    assert sentinel.read_text() == "preserve"


def test_stage_two_identity_requires_the_exact_seed_zero_parent():
    registry = load_registry()
    parent = first_matrix_job(registry)
    derived = json.loads(json.dumps(parent))
    derived.update(
        {
            "job_id": derived["job_id"].replace("seed0", "seed1"),
            "seed": 1,
            "stage": "stage2",
            "status": "pending",
            "output_dir": derived["output_dir"].replace("seed0", "seed1"),
            "parent_seed_zero": parent["job_id"],
        }
    )

    assert registry.validate_job_identity(derived, parent_job=parent) == derived

    tampered = json.loads(json.dumps(derived))
    tampered["provenance"] = {}
    with pytest.raises(ValueError, match="derived|Stage-2"):
        registry.validate_job_identity(tampered, parent_job=parent)

    wrong_parent = json.loads(json.dumps(parent))
    wrong_parent["job_id"] = "unknown__muse_books__seed0"
    with pytest.raises(ValueError, match="registered Stage-1"):
        registry.validate_job_identity(derived, parent_job=wrong_parent)


def test_run_job_executes_the_current_worktree_with_one_absolute_output(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    monkeypatch.chdir(tmp_path)
    relative_output = Path("relative-jobs") / job["job_id"]
    resolved_output = (tmp_path / relative_output).resolve()
    captured = install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        resolved_output,
    )

    result = registry.run_job(job, relative_output)

    assert result["status"] == "completed"
    assert registry.CODE_ROOT == ROOT
    assert registry.CODE_ROOT != registry.SHARED_ROOT
    assert captured["kwargs"]["cwd"] == ROOT
    assert captured["kwargs"]["cwd"] != registry.SHARED_ROOT
    assert captured["argv"][1] == str(ROOT / "src/train.py")
    assert f"paths.output_dir={resolved_output}" in captured["argv"]
    assert captured["argv"] == registry.build_command(job, resolved_output)
    assert json.loads((resolved_output / "command.json").read_text())["argv"] == (
        captured["argv"]
    )


def test_run_job_rejects_nonfinite_diagnostic(tmp_path, monkeypatch):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    records = [
        job_diagnostic(job, 1),
        job_diagnostic(job, 2, corrected_norm=math.nan),
    ]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        diagnostics=records,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert "nonfinite" in " ".join(result["issues"]).lower()


def test_run_job_accepts_exact_first_order_producer_diagnostics(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    records = [
        job_diagnostic(
            job,
            1,
            active_constraints=[],
            retain_loss_before=None,
            retain_loss_after=None,
        ),
        job_diagnostic(
            job,
            2,
            active_constraints=[0, 1],
            retain_loss_before=None,
            retain_loss_after=None,
        ),
    ]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        diagnostics=records,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "completed"
    assert result["projection_count"] == 2


@pytest.mark.parametrize(
    "active_constraints",
    [[True], [2], [1, 1], [1, 0], ["0"]],
)
def test_run_job_rejects_malformed_active_constraint_lists(
    tmp_path,
    monkeypatch,
    active_constraints,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    records = [
        job_diagnostic(job, 1),
        job_diagnostic(job, 2, active_constraints=active_constraints),
    ]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        diagnostics=records,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert "active_constraints" in " ".join(result["issues"])


@pytest.mark.parametrize(
    "record_overrides",
    [
        {"constraint_count": 0, "active_constraints": []},
        {"kkt_residual": -1.0e-8},
        {"kkt_residual": 2.0e-6},
    ],
)
def test_run_job_rejects_impossible_producer_invariants(
    tmp_path,
    monkeypatch,
    record_overrides,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    records = [
        job_diagnostic(job, 1),
        job_diagnostic(job, 2, **record_overrides),
    ]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        diagnostics=records,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"


@pytest.mark.parametrize(
    ("retain_before", "retain_after"),
    [
        (None, 1.0),
        ("missing", "missing"),
        (math.nan, 1.0),
        (1.0, math.inf),
    ],
)
def test_run_job_rejects_malformed_or_nonfinite_retain_losses(
    tmp_path,
    monkeypatch,
    retain_before,
    retain_after,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    records = [
        job_diagnostic(job, 1),
        job_diagnostic(
            job,
            2,
            retain_loss_before=retain_before,
            retain_loss_after=retain_after,
        ),
    ]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        diagnostics=records,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"


@pytest.mark.parametrize("endpoint_count", [0, 2])
def test_run_job_requires_exactly_one_endpoint_pair(
    tmp_path,
    monkeypatch,
    endpoint_count,
):
    registry = load_registry()
    job = first_matrix_job(registry, evaluator_kind="tofu")
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        endpoint_count=endpoint_count,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert "endpoint" in " ".join(result["issues"]).lower()


def test_run_job_rejects_projection_update_mismatch(tmp_path, monkeypatch):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        diagnostics=[job_diagnostic(job, 1)],
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert result["projection_count"] == 1
    assert result["optimizer_update_count"] == 2
    assert "mismatch" in " ".join(result["issues"]).lower()


def test_run_job_rejects_unchanged_selected_parameters(tmp_path, monkeypatch):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    records = [
        job_diagnostic(
            job,
            step,
            proposal_norm=0.0,
            corrected_norm=0.0,
            correction_ratio=0.0,
            applied_scale=0.0,
            zero_step=True,
            zero_step_reason="zero_delta",
        )
        for step in (1, 2)
    ]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        diagnostics=records,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert result["selected_parameter_changed"] is False
    assert result["zero_step_count"] == 2
    assert "unchanged" in " ".join(result["issues"]).lower()


@pytest.mark.parametrize(
    "rejected_record",
    [
        {"zero_step": True, "zero_step_reason": "zero_delta"},
        {"zero_step": False, "zero_step_reason": "retain_budget_exceeded"},
    ],
)
def test_run_job_rejects_any_mixed_zero_or_rejection_record(
    tmp_path,
    monkeypatch,
    rejected_record,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    records = [
        job_diagnostic(job, 1),
        job_diagnostic(
            job,
            2,
            proposal_norm=0.0,
            corrected_norm=0.0,
            correction_ratio=0.0,
            applied_scale=0.0,
            **rejected_record,
        ),
    ]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        diagnostics=records,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert result["selected_parameter_changed"] is True
    assert "zero" in " ".join(result["issues"]).lower() or "reject" in " ".join(
        result["issues"]
    ).lower()


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "weights.npz",
        "model.pkl",
        "arbitrary_state/trace.log",
        "unknown.json",
        "payload.log",
        "trainer_state.log",
        "logs/payload.log",
        ".hydra/extra.yaml",
        "checkpoint-2/evals/EXTRA.json",
        "checkpoint-3/notes.txt",
    ],
)
def test_run_job_rejects_every_file_outside_the_evidence_allowlist(
    tmp_path,
    monkeypatch,
    forbidden_path,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        forbidden_path=forbidden_path,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert forbidden_path in result["forbidden_artifacts"]


def test_run_job_rejects_checkpoint_config_outside_the_live_eval_pair(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        forbidden_path="checkpoint-9/config.json",
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert result["forbidden_artifacts"] == ["checkpoint-9/config.json"]


def test_run_job_accepts_only_the_exact_valid_evidence_tree(tmp_path, monkeypatch):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        extra_files={
            ".hydra/config.yaml": b"config\n",
            ".hydra/hydra.yaml": b"hydra\n",
            ".hydra/overrides.yaml": b"overrides\n",
        },
        extra_dirs=["logs", "checkpoint-9", "checkpoint-9/evals"],
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "completed"
    assert result["forbidden_artifacts"] == []


def test_run_job_accepts_only_its_exact_hydra_handler_log(tmp_path, monkeypatch):
    registry = load_registry()
    job = next(
        candidate
        for candidate in registry.build_manifest(seed=0)["jobs"]
        if candidate["method"] == "SimNPO"
        and candidate["evaluator_kind"] == "muse"
    )
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        extra_files={"SimNPO.log": b"hydra log\n"},
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "completed"
    assert result["forbidden_artifacts"] == []


def test_run_job_rejects_a_different_methods_hydra_log(tmp_path, monkeypatch):
    registry = load_registry()
    job = next(
        candidate
        for candidate in registry.build_manifest(seed=0)["jobs"]
        if candidate["method"] == "SimNPO"
        and candidate["evaluator_kind"] == "muse"
    )
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        extra_files={"NPO.log": b"wrong handler log\n"},
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert result["forbidden_artifacts"] == ["NPO.log"]


def test_run_job_classifies_oom_as_scientific_without_retry(tmp_path, monkeypatch):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    captured = install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        returncode=1,
        log_text="torch.cuda.OutOfMemoryError: CUDA out of memory\n",
        diagnostics=False,
        endpoint_count=0,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert result["failure_kind"] == "oom"
    assert captured["argv"] == registry.build_command(job, output_dir)


def test_scientific_failure_precedes_gpu_monitor_failure(tmp_path, monkeypatch):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        returncode=1,
        log_text="torch.cuda.OutOfMemoryError: CUDA out of memory\n",
        diagnostics=False,
        endpoint_count=0,
        nvml_results=[
            (1, "", "NVML unavailable"),
            (1, "", "NVML unavailable"),
        ],
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert result["failure_kind"] == "oom"
    assert "nvml" in " ".join(result["issues"]).lower()


def test_gpu_monitor_timeout_is_infrastructure_failure(tmp_path, monkeypatch):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    timeout = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        nvml_results=[timeout, timeout],
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "failed_infrastructure"
    assert result["failure_kind"] == "resource_monitor"
    assert "timed out" in " ".join(result["issues"]).lower()


def test_scientific_failure_precedes_gpu_monitor_timeout(tmp_path, monkeypatch):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    timeout = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        returncode=1,
        log_text="torch.cuda.OutOfMemoryError: CUDA out of memory\n",
        diagnostics=False,
        endpoint_count=0,
        nvml_results=[timeout, timeout],
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert result["failure_kind"] == "oom"
    assert "timed out" in " ".join(result["issues"]).lower()


@pytest.mark.parametrize(
    "nvml_results",
    [
        [(0, "", ""), (0, "", "")],
        [(0, "9999, 800 MiB\n", ""), (0, "9999, 900 MiB\n", "")],
    ],
)
def test_completed_evidence_requires_observing_child_gpu_memory(
    tmp_path,
    monkeypatch,
    nvml_results,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        nvml_results=nvml_results,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "failed_infrastructure"
    assert result["failure_kind"] == "resource_monitor"
    assert result["peak_gpu_memory_mib"] == 0
    assert result["gpu_memory_sample_count"] == 0


@pytest.mark.parametrize(
    ("message", "failure_kind"),
    [
        ("OSError: [Errno 5] Input/output error\n", "host_io"),
        ("required cache directory is absent\n", "cache"),
    ],
)
def test_run_job_classifies_host_io_and_cache_as_infrastructure(
    tmp_path,
    monkeypatch,
    message,
    failure_kind,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        returncode=1,
        log_text=message,
        diagnostics=False,
        endpoint_count=0,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "failed_infrastructure"
    assert result["failure_kind"] == failure_kind


def test_run_job_rejects_command_record_identity_tampering(tmp_path, monkeypatch):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(
        monkeypatch,
        registry,
        job,
        output_dir,
        tamper_command=True,
    )

    result = registry.run_job(job, output_dir)

    assert result["status"] == "invalid_scientific"
    assert "command identity" in " ".join(result["issues"]).lower()


def test_run_job_writes_json_records_with_atomic_replace(tmp_path, monkeypatch):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["job_id"]
    install_job_subprocess_stub(monkeypatch, registry, job, output_dir)
    original_replace = registry.os.replace
    replacements = []

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        assert Path(source).is_file()
        return original_replace(source, destination)

    monkeypatch.setattr(registry.os, "replace", recording_replace)

    result = registry.run_job(job, output_dir)

    assert result["status"] == "completed"
    assert {destination.name for _, destination in replacements} >= {
        "command.json",
        "JOB_RESULT.json",
    }
    assert not tuple(output_dir.glob(".*.tmp"))


TERMINAL_QUEUE_STATES = {
    "completed",
    "invalid_scientific",
    "failed_infrastructure",
}


def queue_manifest(registry, output_root):
    manifest = registry.build_manifest(seed=0)
    manifest["output_root"] = str(Path(output_root).resolve())
    return manifest


def queue_result(registry, job, output_root, status="completed", failure_kind=None):
    output_dir = (Path(output_root) / job["output_dir"]).resolve()
    prefix = job_endpoint_prefix(job)
    endpoint_dir = output_dir / "checkpoint-1" / "evals"
    if status == "completed":
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        (endpoint_dir / f"{prefix}_SUMMARY.json").write_text('{"metric": 0.25}')
        (endpoint_dir / f"{prefix}_EVAL.json").write_text('{"metric": 0.25}')
    command_record = {
        "schema_version": registry.SCHEMA_VERSION,
        "protocol": registry.PROTOCOL,
        "job_id": job["job_id"],
        "method": job["method"],
        "benchmark": job["benchmark"],
        "seed": job["seed"],
        "stage": job["stage"],
        "argv": registry.build_command(job, output_dir),
        "environment_overrides": registry.environment_overrides(job),
        "provenance": job["provenance"],
    }
    identity = hashlib.sha256(
        json.dumps(command_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": registry.SCHEMA_VERSION,
        "protocol": registry.PROTOCOL,
        "job_id": job["job_id"],
        "method": job["method"],
        "benchmark": job["benchmark"],
        "seed": job["seed"],
        "stage": job["stage"],
        "status": status,
        "failure_kind": failure_kind,
        "returncode": 0 if status == "completed" else 1,
        "command_path": "command.json",
        "command_identity": identity,
        "environment_overrides": registry.environment_overrides(job),
        "provenance": job["provenance"],
        "run_log_path": "run.log",
        "diagnostics_path": "gu_diagnostics.jsonl",
        "endpoint_summary_path": f"checkpoint-1/evals/{prefix}_SUMMARY.json",
        "endpoint_raw_path": f"checkpoint-1/evals/{prefix}_EVAL.json",
        "wall_clock_seconds": 1.0,
        "peak_nvml_mib": 256,
        "peak_gpu_memory_mib": 256,
        "gpu_memory_sample_count": 2,
        "optimizer_update_count": 2,
        "final_global_step": 2,
        "projection_count": 2,
        "zero_step_count": 0,
        "applied_scale_distribution": {"0.5": 2},
        "correction_ratio": {"min": 0.5, "max": 0.5, "mean": 0.5},
        "max_violation_after": 5.0e-8,
        "selected_parameter_changed": True,
        "forbidden_artifacts": [],
        "issues": [],
    }


def finish_queue_job(registry, job, output_root, status, failure_kind=None):
    result = queue_result(
        registry,
        job,
        output_root,
        status=status,
        failure_kind=failure_kind,
    )
    job.update(status=status, attempt_count=1, pid=None)
    job["attempt_history"] = [
        {
            "attempt": 1,
            "status": status,
            "pid": 4242,
            "argv": registry.build_command(
                job, Path(output_root) / job["output_dir"]
            ),
            "command_identity": result["command_identity"],
            "evidence": result,
        }
    ]
    return result


def mark_queue_job_running(registry, job, output_root, pid=4242):
    result = queue_result(registry, job, output_root)
    job.update(status="running", attempt_count=1, pid=pid)
    job["attempt_history"] = [
        {
            "attempt": 1,
            "status": "running",
            "pid": pid,
            "argv": registry.build_command(
                job, Path(output_root) / job["output_dir"]
            ),
            "command_identity": result["command_identity"],
        }
    ]


def leave_only_pending(state, count, registry, output_root):
    for index, job in enumerate(state["jobs"]):
        if index >= count:
            finish_queue_job(
                registry,
                job,
                output_root,
                "invalid_scientific",
                "evidence_validation",
            )


def install_queue_admission_stub(monkeypatch, registry, used_memory=0):
    monkeypatch.setattr(
        registry,
        "live_matrix_children",
        lambda state, output_root: [],
    )

    def gpu_query(argv, **kwargs):
        assert argv == [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ]
        assert math.isfinite(kwargs["timeout"])
        assert kwargs["timeout"] > 0
        return SimpleNamespace(returncode=0, stdout=f"0, {used_memory}\n", stderr="")

    monkeypatch.setattr(registry.subprocess, "run", gpu_query)


def test_queue_state_creation_is_atomic_and_starts_with_no_running_job(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    manifest = queue_manifest(registry, tmp_path)
    original_replace = registry.os.replace
    replacements = []

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        assert Path(source).is_file()
        return original_replace(source, destination)

    monkeypatch.setattr(registry.os, "replace", recording_replace)

    state = registry.create_queue_state(manifest, state_path)

    assert json.loads(state_path.read_text()) == state
    assert len(state["jobs"]) == 60
    assert {job["status"] for job in state["jobs"]} == {"pending"}
    assert all(job["attempt_count"] == 0 for job in state["jobs"])
    assert all(job["attempt_history"] == [] for job in state["jobs"])
    assert replacements[-1][1] == state_path
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_queue_refuses_state_with_more_than_one_running_job(tmp_path, monkeypatch):
    registry = load_registry()
    manifest_path = tmp_path / "manifest.json"
    state_path = tmp_path / "queue_state.json"
    manifest = queue_manifest(registry, tmp_path)
    manifest_path.write_text(json.dumps(manifest))
    state = registry.create_queue_state(manifest, state_path)
    for job in state["jobs"][:2]:
        mark_queue_job_running(registry, job, tmp_path, pid=123)
    state_path.write_text(json.dumps(state))
    launched = []
    monkeypatch.setattr(
        registry,
        "run_job",
        lambda *args, **kwargs: launched.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="at most one running"):
        registry.run_queue(manifest_path, state_path)

    assert launched == []


def test_queue_requires_device_zero_below_500_mib_before_launch(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    manifest_path = tmp_path / "manifest.json"
    state_path = tmp_path / "queue_state.json"
    manifest = queue_manifest(registry, tmp_path)
    manifest_path.write_text(json.dumps(manifest))
    state = registry.create_queue_state(manifest, state_path)
    leave_only_pending(state, 1, registry, tmp_path)
    state_path.write_text(json.dumps(state))
    install_queue_admission_stub(monkeypatch, registry, used_memory=500)
    launched = []
    monkeypatch.setattr(
        registry,
        "run_job",
        lambda *args, **kwargs: launched.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="500 MiB"):
        registry.run_queue(manifest_path, state_path)

    assert launched == []
    persisted = json.loads(state_path.read_text())
    assert persisted["jobs"][0]["status"] == "pending"


def test_queue_resume_monitors_exact_live_child_then_adopts_and_continues(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    manifest_path = tmp_path / "manifest.json"
    state_path = tmp_path / "queue_state.json"
    manifest = queue_manifest(registry, tmp_path)
    manifest_path.write_text(json.dumps(manifest))
    state = registry.create_queue_state(manifest, state_path)
    leave_only_pending(state, 2, registry, tmp_path)
    running, next_job = state["jobs"][:2]
    mark_queue_job_running(registry, running, tmp_path)
    output = tmp_path / running["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "JOB_RESULT.json").write_text(
        json.dumps(queue_result(registry, running, tmp_path))
    )
    state_path.write_text(json.dumps(state))
    observations = iter(
        [
            [{"job_id": running["job_id"], "pid": 4242, "exact": True}],
            [{"job_id": running["job_id"], "pid": 4242, "exact": True}],
            [],
        ]
    )
    monkeypatch.setattr(
        registry,
        "live_matrix_children",
        lambda state, output_root: next(observations, []),
    )
    sleeps = []
    monkeypatch.setattr(registry.time, "sleep", sleeps.append)
    monkeypatch.setattr(registry, "admit_launch", lambda state, output_root: 0)
    launched = []

    def stub_run_job(job, output_dir, **kwargs):
        launched.append(job["job_id"])
        return queue_result(registry, job, tmp_path)

    monkeypatch.setattr(registry, "run_job", stub_run_job)

    registry.run_queue(manifest_path, state_path)

    persisted = json.loads(state_path.read_text())
    assert [job["status"] for job in persisted["jobs"][:2]] == [
        "completed",
        "completed",
    ]
    assert launched == [next_job["job_id"]]
    assert sleeps == [1.0, 1.0]


def test_queue_resume_adopts_valid_result_but_never_an_endpoint_alone(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    manifest_path = tmp_path / "manifest.json"
    state_path = tmp_path / "queue_state.json"
    manifest = queue_manifest(registry, tmp_path)
    manifest_path.write_text(json.dumps(manifest))
    state = registry.create_queue_state(manifest, state_path)
    leave_only_pending(state, 0, registry, tmp_path)
    first, second = state["jobs"][:2]
    mark_queue_job_running(registry, first, tmp_path, pid=9999)
    first_output = tmp_path / first["output_dir"]
    first_output.mkdir(parents=True, exist_ok=True)
    (first_output / "JOB_RESULT.json").write_text(
        json.dumps(
            queue_result(
                registry,
                first,
                tmp_path,
                "invalid_scientific",
                "oom",
            )
        )
    )
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(registry, "live_matrix_children", lambda *args: [])
    install_queue_admission_stub(monkeypatch, registry)
    attempts = []

    def stub_run_job(job, output_dir, **kwargs):
        attempts.append(job["job_id"])
        return queue_result(registry, job, tmp_path, "failed_infrastructure", "host_io")

    monkeypatch.setattr(registry, "run_job", stub_run_job)

    registry.run_queue(manifest_path, state_path)

    state = json.loads(state_path.read_text())
    second = state["jobs"][1]
    mark_queue_job_running(registry, second, tmp_path, pid=9999)
    second_output = tmp_path / second["output_dir"] / "checkpoint-1" / "evals"
    second_output.mkdir(parents=True, exist_ok=True)
    (second_output / "MUSE_SUMMARY.json").write_text('{"metric": 1.0}')
    state_path.write_text(json.dumps(state))
    registry.run_queue(manifest_path, state_path)

    persisted = json.loads(state_path.read_text())
    assert persisted["jobs"][0]["status"] == "invalid_scientific"
    assert persisted["jobs"][0]["attempt_count"] == 1
    assert persisted["jobs"][1]["status"] == "failed_infrastructure"
    assert persisted["jobs"][1]["attempt_count"] == 2
    assert attempts == [second["job_id"]]


def test_queue_retries_infrastructure_once_with_identical_command_then_continues(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    manifest_path = tmp_path / "manifest.json"
    state_path = tmp_path / "queue_state.json"
    manifest = queue_manifest(registry, tmp_path)
    manifest_path.write_text(json.dumps(manifest))
    state = registry.create_queue_state(manifest, state_path)
    leave_only_pending(state, 2, registry, tmp_path)
    state_path.write_text(json.dumps(state))
    install_queue_admission_stub(monkeypatch, registry)
    calls = []

    def stub_run_job(job, output_dir, launched=None, **kwargs):
        if launched is not None:
            launched(6000 + len(calls))
        calls.append((job["job_id"], registry.build_command(job, output_dir)))
        first_id = state["jobs"][0]["job_id"]
        if job["job_id"] == first_id:
            return queue_result(
                registry,
                job,
                tmp_path,
                "failed_infrastructure",
                "host_io",
            )
        return queue_result(registry, job, tmp_path)

    monkeypatch.setattr(registry, "run_job", stub_run_job)

    registry.run_queue(manifest_path, state_path)

    persisted = json.loads(state_path.read_text())
    first, second = persisted["jobs"][:2]
    assert first["status"] == "failed_infrastructure"
    assert first["attempt_count"] == 2
    assert [entry["status"] for entry in first["attempt_history"]] == [
        "failed_infrastructure",
        "failed_infrastructure",
    ]
    assert second["status"] == "completed"
    assert second["attempt_count"] == 1
    assert [job_id for job_id, _ in calls] == [
        first["job_id"],
        first["job_id"],
        second["job_id"],
    ]
    assert calls[0][1] == calls[1][1]


def test_queue_never_retries_scientific_failure_and_continues(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    manifest_path = tmp_path / "manifest.json"
    state_path = tmp_path / "queue_state.json"
    manifest = queue_manifest(registry, tmp_path)
    manifest_path.write_text(json.dumps(manifest))
    state = registry.create_queue_state(manifest, state_path)
    leave_only_pending(state, 2, registry, tmp_path)
    state_path.write_text(json.dumps(state))
    install_queue_admission_stub(monkeypatch, registry)
    calls = []

    def stub_run_job(job, output_dir, **kwargs):
        calls.append(job["job_id"])
        if job["job_id"] == state["jobs"][0]["job_id"]:
            return queue_result(
                registry,
                job,
                tmp_path,
                "invalid_scientific",
                "oom",
            )
        return queue_result(registry, job, tmp_path)

    monkeypatch.setattr(registry, "run_job", stub_run_job)

    registry.run_queue(manifest_path, state_path)

    persisted = json.loads(state_path.read_text())
    assert [job["status"] for job in persisted["jobs"][:2]] == [
        "invalid_scientific",
        "completed",
    ]
    assert [job["attempt_count"] for job in persisted["jobs"][:2]] == [1, 1]
    assert calls == [job["job_id"] for job in persisted["jobs"][:2]]


def test_stage_two_expands_every_valid_seed_zero_parent_exactly_once(tmp_path):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    manifest = queue_manifest(registry, tmp_path)
    state = registry.create_queue_state(manifest, state_path)
    for job in state["jobs"]:
        finish_queue_job(
            registry,
            job,
            tmp_path,
            "invalid_scientific",
            "evidence_validation",
        )
    completed = state["jobs"][:2]
    for job in completed:
        result = finish_queue_job(registry, job, tmp_path, "completed")
        output = tmp_path / job["output_dir"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "JOB_RESULT.json").write_text(json.dumps(result))
    state_path.write_text(json.dumps(state))

    expanded = registry.expand_seeds(state_path)
    expanded_again = registry.expand_seeds(state_path)

    derived = [job for job in expanded["jobs"] if job["seed"] in {1, 2}]
    assert len(derived) == 4
    assert expanded_again == expanded
    assert {job["seed"] for job in derived} == {1, 2}
    assert {job["parent_seed_zero"] for job in derived} == {
        job["job_id"] for job in completed
    }
    for job in derived:
        parent = next(
            candidate
            for candidate in completed
            if candidate["job_id"] == job["parent_seed_zero"]
        )
        registry.validate_job_identity(job, parent_job=parent)
        assert job["status"] == "pending"
        assert job["attempt_count"] == 0
        assert job["attempt_history"] == []


def test_stage_two_refuses_expansion_before_stage_one_is_terminal(tmp_path):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    registry.create_queue_state(queue_manifest(registry, tmp_path), state_path)

    with pytest.raises(ValueError, match="Stage 1.*terminal"):
        registry.expand_seeds(state_path)


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "injected", "reordered"])
def test_stage_two_requires_the_exact_complete_unique_stage_one_registry(
    tmp_path,
    tamper,
):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    state = registry.create_queue_state(queue_manifest(registry, tmp_path), state_path)
    leave_only_pending(state, 0, registry, tmp_path)
    if tamper == "missing":
        state["jobs"].pop()
    elif tamper == "duplicate":
        state["jobs"].append(json.loads(json.dumps(state["jobs"][0])))
    elif tamper == "injected":
        state["jobs"][-1] = json.loads(json.dumps(state["jobs"][0]))
        state["jobs"][-1]["job_id"] = "Injected__muse_books__seed0"
    else:
        state["jobs"][0], state["jobs"][1] = state["jobs"][1], state["jobs"][0]
    state_path.write_text(json.dumps(state))

    with pytest.raises(ValueError, match="Stage-1|canonical|registry"):
        registry.expand_seeds(state_path)


@pytest.mark.parametrize("evidence", ["missing", "corrupt", "wrong_status"])
def test_stage_two_fails_for_invalid_completed_parent_evidence(tmp_path, evidence):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    state = registry.create_queue_state(queue_manifest(registry, tmp_path), state_path)
    leave_only_pending(state, 0, registry, tmp_path)
    parent = state["jobs"][0]
    finish_queue_job(registry, parent, tmp_path, "completed")
    result_path = tmp_path / parent["output_dir"] / "JOB_RESULT.json"
    if evidence != "missing":
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if evidence == "corrupt":
            result_path.write_text("not-json")
        else:
            result_path.write_text(
                json.dumps(
                    queue_result(
                        registry,
                        parent,
                        tmp_path,
                        "invalid_scientific",
                        "oom",
                    )
                )
            )
    state_path.write_text(json.dumps(state))

    with pytest.raises((OSError, ValueError, json.JSONDecodeError)):
        registry.expand_seeds(state_path)
    assert registry.main(["expand-seeds", "--state", str(state_path)]) != 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("projection_count", 0),
        ("optimizer_update_count", 0),
        ("zero_step_count", 1),
        ("selected_parameter_changed", False),
        ("forbidden_artifacts", ["checkpoint-1/model.safetensors"]),
        ("issues", ["contract mismatch"]),
        ("peak_gpu_memory_mib", float("nan")),
        ("correction_ratio", {"min": 0.5, "max": 0.5, "mean": float("inf")}),
        ("endpoint_summary_path", None),
        ("endpoint_summary_path", "checkpoint-1/evals/WRONG_SUMMARY.json"),
    ],
)
def test_completed_result_validation_reuses_the_full_task_three_contract(
    tmp_path,
    field,
    value,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    result = queue_result(registry, job, tmp_path)
    result[field] = value

    with pytest.raises(ValueError, match="completed|evidence|status"):
        registry.validate_job_result(
            result,
            job,
            tmp_path / job["output_dir"],
            expected_status="completed",
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "injected",
        "duplicate",
        "reordered",
        "attempt_count",
        "attempt_number",
        "argv",
        "evidence_failure_kind",
        "running_pid_mismatch",
        "terminal_pid",
        "pending_pid",
    ],
)
def test_status_rejects_queue_state_tampering_before_process_inspection(
    tmp_path,
    monkeypatch,
    tamper,
):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    state = registry.create_queue_state(queue_manifest(registry, tmp_path), state_path)
    leave_only_pending(state, 0, registry, tmp_path)
    first = state["jobs"][0]
    if tamper == "injected":
        injected = json.loads(json.dumps(first))
        injected["job_id"] = "Injected__muse_books__seed0"
        state["jobs"].append(injected)
    elif tamper == "duplicate":
        state["jobs"].append(json.loads(json.dumps(first)))
    elif tamper == "reordered":
        state["jobs"][0], state["jobs"][1] = state["jobs"][1], state["jobs"][0]
    elif tamper == "attempt_count":
        first["attempt_count"] = 2
    elif tamper == "attempt_number":
        first["attempt_history"][0]["attempt"] = 2
    elif tamper == "argv":
        first["attempt_history"][0]["argv"].append("tampered=true")
    elif tamper == "evidence_failure_kind":
        first["attempt_history"][0]["evidence"]["failure_kind"] = None
    elif tamper == "running_pid_mismatch":
        mark_queue_job_running(registry, first, tmp_path, pid=4242)
        first["pid"] = None
    elif tamper == "terminal_pid":
        first["pid"] = 123
    else:
        pending = state["jobs"][1]
        pending.update(status="pending", attempt_count=0, attempt_history=[], pid=123)
    state_path.write_text(json.dumps(state))
    inspected = []
    monkeypatch.setattr(
        registry,
        "live_matrix_children",
        lambda *args: inspected.append(args),
    )

    with pytest.raises(ValueError, match="queue|Stage-1|canonical|attempt|pid"):
        registry.queue_status(state_path)

    assert inspected == []
    assert registry.main(["status", "--state", str(state_path)]) != 0


def test_status_rejects_duplicate_stage_two_job_identity(tmp_path):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    state = registry.create_queue_state(queue_manifest(registry, tmp_path), state_path)
    leave_only_pending(state, 0, registry, tmp_path)
    parent = state["jobs"][0]
    derived = json.loads(json.dumps(parent))
    derived.update(
        job_id=parent["job_id"].replace("seed0", "seed1"),
        seed=1,
        stage="stage2",
        status="pending",
        output_dir=parent["output_dir"].replace("seed0", "seed1"),
        parent_seed_zero=parent["job_id"],
        attempt_count=0,
        attempt_history=[],
        pid=None,
    )
    state["jobs"].extend([derived, json.loads(json.dumps(derived))])
    state_path.write_text(json.dumps(state))

    with pytest.raises(ValueError, match="duplicate|canonical"):
        registry.queue_status(state_path)


@pytest.mark.parametrize("parent_status", ["invalid_scientific", "failed_infrastructure"])
def test_status_and_expansion_reject_derived_job_from_invalid_parent(
    tmp_path,
    parent_status,
):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    state = registry.create_queue_state(queue_manifest(registry, tmp_path), state_path)
    leave_only_pending(state, 0, registry, tmp_path)
    parent = state["jobs"][0]
    if parent_status == "invalid_scientific":
        finish_queue_job(
            registry,
            parent,
            tmp_path,
            "invalid_scientific",
            "oom",
        )
    else:
        result = queue_result(
            registry,
            parent,
            tmp_path,
            "failed_infrastructure",
            "host_io",
        )
        command = registry.build_command(parent, tmp_path / parent["output_dir"])
        parent.update(
            status="failed_infrastructure",
            attempt_count=2,
            attempt_history=[
                {
                    "attempt": attempt,
                    "status": "failed_infrastructure",
                    "pid": 4000 + attempt,
                    "argv": command,
                    "command_identity": result["command_identity"],
                    "evidence": result,
                }
                for attempt in (1, 2)
            ],
            pid=None,
        )
    derived = json.loads(json.dumps(parent))
    derived.update(
        job_id=parent["job_id"].replace("seed0", "seed1"),
        seed=1,
        stage="stage2",
        status="pending",
        output_dir=parent["output_dir"].replace("seed0", "seed1"),
        parent_seed_zero=parent["job_id"],
        attempt_count=0,
        attempt_history=[],
        pid=None,
    )
    state["jobs"].append(derived)
    state_path.write_text(json.dumps(state))

    with pytest.raises(ValueError, match="parent|completed"):
        registry.queue_status(state_path)
    with pytest.raises(ValueError, match="parent|completed"):
        registry.expand_seeds(state_path)


@pytest.mark.parametrize(
    "tamper",
    ["different_checkpoint", "violation", "traversal", "missing_file"],
)
def test_completed_evidence_requires_safe_existing_endpoint_pair_and_tolerance(
    tmp_path,
    tamper,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["output_dir"]
    result = queue_result(registry, job, tmp_path)
    prefix = job_endpoint_prefix(job)
    if tamper == "different_checkpoint":
        other = output_dir / "checkpoint-2" / "evals"
        other.mkdir(parents=True)
        (other / f"{prefix}_EVAL.json").write_text('{"metric": 0.25}')
        result["endpoint_raw_path"] = f"checkpoint-2/evals/{prefix}_EVAL.json"
    elif tamper == "violation":
        result["max_violation_after"] = 999.0
    elif tamper == "traversal":
        result["endpoint_summary_path"] = (
            f"../checkpoint-1/evals/{prefix}_SUMMARY.json"
        )
    else:
        (output_dir / result["endpoint_raw_path"]).unlink()

    with pytest.raises(ValueError, match="completed|endpoint|projection"):
        registry.validate_job_result(
            result,
            job,
            output_dir,
            expected_status="completed",
        )


@pytest.mark.parametrize(
    "tamper",
    ["extra_pair", "corrupt", "empty", "nonnumeric", "nonfinite"],
)
def test_completed_evidence_revalidates_exact_endpoint_file_contents(
    tmp_path,
    tamper,
):
    registry = load_registry()
    job = first_matrix_job(registry)
    output_dir = tmp_path / job["output_dir"]
    result = queue_result(registry, job, tmp_path)
    prefix = job_endpoint_prefix(job)
    summary = output_dir / result["endpoint_summary_path"]
    raw = output_dir / result["endpoint_raw_path"]
    if tamper == "extra_pair":
        extra = output_dir / "checkpoint-2" / "evals"
        extra.mkdir(parents=True)
        (extra / f"{prefix}_SUMMARY.json").write_text('{"metric": 0.5}')
        (extra / f"{prefix}_EVAL.json").write_text('{"metric": 0.5}')
    elif tamper == "corrupt":
        summary.write_text("not-json")
    elif tamper == "empty":
        raw.write_text("{}")
    elif tamper == "nonnumeric":
        raw.write_text('{"metric": "not measured"}')
    else:
        raw.write_text('{"metric": NaN}')

    with pytest.raises(ValueError, match="endpoint|completed|numeric|JSON"):
        registry.validate_job_result(
            result,
            job,
            output_dir,
            expected_status="completed",
        )


@pytest.mark.parametrize("seed", [1, 2])
def test_queue_state_rejects_singleton_replication_seed(tmp_path, seed):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    state = registry.create_queue_state(queue_manifest(registry, tmp_path), state_path)
    leave_only_pending(state, 0, registry, tmp_path)
    parent = state["jobs"][0]
    finish_queue_job(registry, parent, tmp_path, "completed")
    derived = json.loads(json.dumps(parent))
    derived.update(
        job_id=parent["job_id"].replace("seed0", f"seed{seed}"),
        seed=seed,
        stage="stage2",
        status="pending",
        output_dir=parent["output_dir"].replace("seed0", f"seed{seed}"),
        parent_seed_zero=parent["job_id"],
        attempt_count=0,
        attempt_history=[],
        pid=None,
    )
    state["jobs"].append(derived)
    state_path.write_text(json.dumps(state))

    with pytest.raises(ValueError, match="replication|both|pair"):
        registry.queue_status(state_path)


def test_queue_state_rejects_partial_global_replication(tmp_path):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    state = registry.create_queue_state(queue_manifest(registry, tmp_path), state_path)
    leave_only_pending(state, 0, registry, tmp_path)
    replicated_parent, omitted_parent = state["jobs"][:2]
    finish_queue_job(registry, replicated_parent, tmp_path, "completed")
    finish_queue_job(registry, omitted_parent, tmp_path, "completed")
    for seed in (1, 2):
        derived = json.loads(json.dumps(replicated_parent))
        derived.update(
            job_id=replicated_parent["job_id"].replace("seed0", f"seed{seed}"),
            seed=seed,
            stage="stage2",
            status="pending",
            output_dir=replicated_parent["output_dir"].replace(
                "seed0", f"seed{seed}"
            ),
            parent_seed_zero=replicated_parent["job_id"],
            attempt_count=0,
            attempt_history=[],
            pid=None,
        )
        state["jobs"].append(derived)
    state_path.write_text(json.dumps(state))

    with pytest.raises(ValueError, match="global|every completed|atomic"):
        registry.queue_status(state_path)


def test_seed_expansion_writes_both_replicates_in_one_atomic_replace(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    state_path = tmp_path / "queue_state.json"
    state = registry.create_queue_state(queue_manifest(registry, tmp_path), state_path)
    leave_only_pending(state, 0, registry, tmp_path)
    parent = state["jobs"][0]
    result = finish_queue_job(registry, parent, tmp_path, "completed")
    result_path = tmp_path / parent["output_dir"] / "JOB_RESULT.json"
    result_path.write_text(json.dumps(result))
    state_path.write_text(json.dumps(state))
    original_replace = registry.os.replace
    replacements = []

    def recording_replace(source, destination):
        replacements.append(Path(destination))
        return original_replace(source, destination)

    monkeypatch.setattr(registry.os, "replace", recording_replace)

    expanded = registry.expand_seeds(state_path)
    expanded_again = registry.expand_seeds(state_path)

    children = [
        job for job in expanded["jobs"] if job.get("parent_seed_zero") == parent["job_id"]
    ]
    assert {job["seed"] for job in children} == {1, 2}
    assert replacements == [state_path]
    assert expanded_again == expanded


@pytest.mark.parametrize("controller", ["queue", "smoke", "run-job"])
def test_global_dev0_lock_blocks_controllers_from_a_different_output_root(
    tmp_path,
    monkeypatch,
    controller,
):
    registry = load_registry()
    output_root = tmp_path / "second-output-root"
    output_root.mkdir()
    manifest_path = output_root / "manifest.json"
    state_path = output_root / "queue_state.json"
    manifest = queue_manifest(registry, output_root)
    manifest_path.write_text(json.dumps(manifest))
    state = registry.create_queue_state(manifest, state_path)
    leave_only_pending(state, 1, registry, output_root)
    state_path.write_text(json.dumps(state))
    admitted = []
    launched = []
    monkeypatch.setattr(
        registry,
        "admit_launch",
        lambda *args: admitted.append(controller) or 0,
    )
    monkeypatch.setattr(registry, "preflight_manifest", lambda *args: {})

    def stub_run_job(job, output_dir, **kwargs):
        launched.append(job["job_id"])
        return queue_result(registry, job, output_root)

    monkeypatch.setattr(registry, "run_job", stub_run_job)
    lock_path = Path("/tmp/geo_unlearning_gu_matrix_dev0.lock")
    with lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if controller == "queue":
            with pytest.raises(RuntimeError, match="controller.*active|lock"):
                registry.run_queue(manifest_path, state_path)
        elif controller == "smoke":
            with pytest.raises(RuntimeError, match="controller.*active|lock"):
                registry.smoke_manifest(manifest_path)
        else:
            assert registry.main(
                [
                    "run-job",
                    "--manifest",
                    str(manifest_path),
                    "--job-id",
                    manifest["jobs"][0]["job_id"],
                ]
            ) != 0

    assert admitted == []
    assert launched == []


def test_resume_adopts_first_infrastructure_result_then_runs_exact_retry(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    manifest_path = tmp_path / "manifest.json"
    state_path = tmp_path / "queue_state.json"
    manifest = queue_manifest(registry, tmp_path)
    manifest_path.write_text(json.dumps(manifest))
    state = registry.create_queue_state(manifest, state_path)
    leave_only_pending(state, 0, registry, tmp_path)
    job = state["jobs"][0]
    mark_queue_job_running(registry, job, tmp_path, pid=9999)
    result_path = tmp_path / job["output_dir"] / "JOB_RESULT.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            queue_result(
                registry,
                job,
                tmp_path,
                "failed_infrastructure",
                "host_io",
            )
        )
    )
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(registry, "live_matrix_children", lambda *args: [])
    monkeypatch.setattr(registry, "admit_launch", lambda *args: 0)
    calls = []

    def stub_run_job(retry_job, output_dir, **kwargs):
        calls.append(registry.build_command(retry_job, output_dir))
        return queue_result(registry, retry_job, tmp_path)

    monkeypatch.setattr(registry, "run_job", stub_run_job)

    registry.run_queue(manifest_path, state_path)

    persisted = json.loads(state_path.read_text())["jobs"][0]
    assert persisted["status"] == "completed"
    assert persisted["attempt_count"] == 2
    assert [entry["status"] for entry in persisted["attempt_history"]] == [
        "failed_infrastructure",
        "completed",
    ]
    assert len(calls) == 1
    assert persisted["attempt_history"][0]["argv"] == calls[0]
    assert persisted["attempt_history"][1]["argv"] == calls[0]


def test_manifest_cli_writes_only_after_validation_and_run_job_rejects_unknown(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    output_root = tmp_path / "matrix"

    assert registry.main(
        ["manifest", "--seed", "0", "--output-root", str(output_root)]
    ) == 0
    manifest_path = output_root / "manifest.json"
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text())["output_root"] == str(
        output_root.resolve()
    )

    launched = []
    monkeypatch.setattr(
        registry,
        "run_job",
        lambda *args, **kwargs: launched.append((args, kwargs)),
    )
    assert registry.main(
        ["run-job", "--manifest", str(manifest_path), "--job-id", "unknown"]
    ) != 0
    assert launched == []
    assert not (output_root / "jobs" / "unknown").exists()


def test_preflight_and_smoke_call_real_validation_and_run_seams(
    tmp_path,
    monkeypatch,
):
    registry = load_registry()
    manifest_path = tmp_path / "manifest.json"
    manifest = queue_manifest(registry, tmp_path)
    manifest_path.write_text(json.dumps(manifest))
    validated = []
    launched = []

    def stub_validate_sources(job, fingerprint_memo=None):
        validated.append(job["job_id"])
        return {"validated": job["job_id"]}

    def stub_run_job(job, output_dir, **kwargs):
        launched.append(job["job_id"])
        return queue_result(registry, job, tmp_path)

    monkeypatch.setattr(registry, "validate_sources", stub_validate_sources)
    monkeypatch.setattr(registry, "run_job", stub_run_job)
    monkeypatch.setattr(registry, "admit_launch", lambda state, output_root: 0)

    assert registry.main(["preflight", "--manifest", str(manifest_path)]) == 0
    assert validated == [job["job_id"] for job in manifest["jobs"]]
    validated.clear()
    assert registry.main(["smoke", "--manifest", str(manifest_path)]) == 0
    assert validated == [job["job_id"] for job in manifest["jobs"]]
    assert launched
    assert {job_id.split("__", 1)[0] for job_id in launched} == set(
        registry.STAGE_ONE_METHODS
    )
    assert sum("__RMU__" in f"__{job_id}__" for job_id in launched) == len(
        registry.BENCHMARKS
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["preflight", "--manifest", "missing.json"],
        ["smoke", "--manifest", "missing.json"],
        ["queue", "--manifest", "missing.json", "--state", "state.json"],
        ["expand-seeds", "--state", "missing.json"],
        ["status", "--state", "missing.json"],
    ],
)
def test_cli_paths_return_nonzero_for_missing_inputs(argv):
    assert load_registry().main(argv) != 0
