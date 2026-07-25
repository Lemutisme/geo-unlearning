import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
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
            "src/train.py",
            "--config-name=unlearn.yaml",
        ]
        assert command.count("save_model_after_train=false") == 1
        assert command.count("trainer.args.save_strategy=no") == 1
        assert command.count("trainer.args.save_only_model=false") == 1
        assert command.count("trainer.args.do_eval=true") == 1
        assert command.count("trainer.args.eval_on_start=false") == 1
        assert command.count("trainer.args.eval_strategy=no") == 1
        assert command.count("trainer.args.report_to=none") == 1


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
    assert environment["HF_HOME"] == "/registered/model-cache"
    assert environment["MATRIX_CALLER_VALUE"] == "preserved"


def test_every_job_binds_environment_requirements_and_reference_to_one_root():
    registry = load_registry()
    for job in registry.build_manifest(seed=0)["jobs"]:
        family = benchmark_family(job)
        key = job["benchmark"] if family == "muse" else family
        root = RUNTIME_ROOTS[key]
        environment = registry.environment_overrides(job)
        assert environment == {
            "HF_DATASETS_CACHE": str(root / "datasets"),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_CACHE": str(root / "hub"),
            "HF_HUB_OFFLINE": "1",
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
            "src/train.py",
            "--config-name=unlearn.yaml",
        ]
        assert job["source_requirements"]["model"] == job["provenance"]["model"]
        assert job["fingerprints"] == {}
        assert job["benchmark"] == "muse_books"
        assert job["environment"] == {
            "HF_DATASETS_CACHE": "/dev/shm/gu-matrix-muse-books/datasets",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_CACHE": "/dev/shm/gu-matrix-muse-books/hub",
            "HF_HUB_OFFLINE": "1",
        }
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before
