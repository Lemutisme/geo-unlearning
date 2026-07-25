"""Registry for the GU full-matrix experiment jobs."""

import argparse
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path


SCHEMA_VERSION = 1
PROTOCOL = "gu_full_matrix_20260724"
STAGE = "stage1"
SHARED_ROOT = Path("/workspace/re/GU/geo-unlearning")
WMDP_EVALUATION_CACHE = "/dev/shm/ungu-hf-datasets-wmdp"
WMDP_OFFLINE_ENVIRONMENT = {
    "HF_DATASETS_CACHE": WMDP_EVALUATION_CACHE,
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
}

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

STAGE_ONE_METHODS = tuple(method for method in METHODS if method != "DPO")

TOFU_PARAMETER_REGEX = r"model[.]layers[.](29|30|31)[.]mlp[.]down_proj[.]weight"
SEVEN_B_PARAMETER_REGEX = r"model[.]layers[.](5|6|7)[.]mlp[.]down_proj[.]weight"

TOFU_MODEL = {
    "config": "Llama-3.1-8B-Instruct",
    "pretrained_model_name_or_path": (
        "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
    ),
}
GIBBERISH_CLASSIFIER_PROVENANCE = {
    "artifact": "madhurjindal/autonlp-Gibberish-Detector-492513457",
    "revision": "76672dd7d3575f68ab980705bcec975cc62de71c",
}
TOFU_PROVENANCE = {
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
    "gibberish_classifier": GIBBERISH_CLASSIFIER_PROVENANCE,
}
MUSE_TOKENIZER_PROVENANCE = {
    "artifact": "NousResearch/Llama-2-7b-hf",
    "revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
}

BENCHMARKS = {
    "tofu_forget01": {
        "split": "forget01",
        "retain_split": "retain99",
        "holdout_split": "holdout01",
        "experiment_config": "unlearn/tofu/gu_matrix_8b",
        "model": TOFU_MODEL,
        "evaluator_kind": "tofu",
        "selected_parameter_regex": TOFU_PARAMETER_REGEX,
        "provenance": TOFU_PROVENANCE,
    },
    "tofu_forget05": {
        "split": "forget05",
        "retain_split": "retain95",
        "holdout_split": "holdout05",
        "experiment_config": "unlearn/tofu/gu_matrix_8b",
        "model": TOFU_MODEL,
        "evaluator_kind": "tofu",
        "selected_parameter_regex": TOFU_PARAMETER_REGEX,
        "provenance": TOFU_PROVENANCE,
    },
    "tofu_forget10": {
        "split": "forget10",
        "retain_split": "retain90",
        "holdout_split": "holdout10",
        "experiment_config": "unlearn/tofu/gu_matrix_8b",
        "model": TOFU_MODEL,
        "evaluator_kind": "tofu",
        "selected_parameter_regex": TOFU_PARAMETER_REGEX,
        "provenance": TOFU_PROVENANCE,
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
        "selected_parameter_regex": SEVEN_B_PARAMETER_REGEX,
        "provenance": {
            "model": {
                "artifact": "muse-bench/MUSE-News_target",
                "revision": "a2f39769e9a0b98ec1cdd12f65e9962502208935",
            },
            "tokenizer": MUSE_TOKENIZER_PROVENANCE,
            "dataset": {
                "artifact": "muse-bench/MUSE-News",
                "revision": "506bd5b150b92814d45e4404a82f120ab2d748bf",
            },
            "reference_model": {
                "artifact": "muse-bench/MUSE-News_retrain",
                "revision": "324ef49ee0a038078aba7d8de831edf57235c9b3",
            },
            "gibberish_classifier": GIBBERISH_CLASSIFIER_PROVENANCE,
        },
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
        "selected_parameter_regex": SEVEN_B_PARAMETER_REGEX,
        "provenance": {
            "model": {
                "artifact": "muse-bench/MUSE-Books_target",
                "revision": "c8dd3fb23a726762ec66d277933c7cff6767f3c2",
            },
            "tokenizer": MUSE_TOKENIZER_PROVENANCE,
            "dataset": {
                "artifact": "muse-bench/MUSE-Books",
                "revision": "051ba90319e920d410d87cfdbd61f25843c1b892",
            },
            "reference_model": {
                "artifact": "muse-bench/MUSE-Books_retrain",
                "revision": "1d67430e4e8bdf2a65823740e909792519175ac2",
            },
            "gibberish_classifier": GIBBERISH_CLASSIFIER_PROVENANCE,
        },
    },
    "wmdp_cyber": {
        "split": "cyber",
        "experiment_config": "unlearn/wmdp/gu_matrix_cyber",
        "model": {
            "config": "zephyr-7b-beta",
            "pretrained_model_name_or_path": "HuggingFaceH4/zephyr-7b-beta",
        },
        "evaluator_kind": "lm_eval",
        "selected_parameter_regex": SEVEN_B_PARAMETER_REGEX,
        "provenance": {
            "model": {
                "artifact": "HuggingFaceH4/zephyr-7b-beta",
                "revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c",
            },
            "tokenizer": {
                "artifact": "HuggingFaceH4/zephyr-7b-beta",
                "revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c",
            },
            "forget_corpus": {
                "path": (
                    "/workspace/re/GU/geo-unlearning/data/wmdp/wmdp-corpora/"
                    "cyber-forget-corpus.jsonl"
                ),
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
                "path": (
                    "/workspace/re/GU/geo-unlearning/data/wmdp/wmdp-corpora/"
                    "cyber-retain-corpus.jsonl"
                ),
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
    },
}

COMMON_RUNTIME_ARGUMENTS = (
    "trainer.args.seed={seed}",
    "+trainer.args.data_seed={seed}",
    "save_model_after_train=false",
    "trainer.args.save_strategy=no",
    "trainer.args.save_only_model=false",
    "trainer.args.do_eval=true",
    "trainer.args.eval_on_start=false",
    "trainer.args.eval_strategy=no",
    "trainer.args.report_to=none",
)
COMMON_GU_ARGUMENTS = (
    "+trainer.method_args.gu.enabled=true",
    '+trainer.method_args.gu.parameter_regex=["{parameter_regex}"]',
    "+trainer.method_args.gu.retain_history_rank=8",
    "+trainer.method_args.gu.projection_eps=1e-6",
    "+trainer.method_args.gu.retain_filter=first_order",
    "+trainer.method_args.gu.retain_budget=1e-4",
    "+trainer.method_args.gu.backtracking_scales=[1.0,0.5,0.25,0.125]",
    "+trainer.method_args.gu.diagnostics_path=gu_diagnostics.jsonl",
)
RETAIN_LOG_HASHES = {
    "tofu_forget01": "d307bb83ea3e3409fd01cc50908501ea16aba49e9ed6a779f8ceb75ec2e5503f",
    "tofu_forget05": "d94b259bc878e1a38f97cc0fac6937147d09d5958cd24d7fe0437d4be7d3a29f",
    "tofu_forget10": "aa4192bc0e60ee97f47aa514e39510e70f3a8cdc8aa6330591a3d7a0f88e95f2",
    "muse_news": "11f8c9338f955a9fc1ce4148daa0fbd2867807c648a43f99dc240442d2135ec2",
    "muse_books": "502d38cd7da4fda841e4ee411b488cce3be0d882a1e2c07a0117a83045be8b76",
}
RMU_OVERRIDES = {
    "tofu": (
        "trainer.method_args.gamma=1.0",
        "trainer.method_args.alpha=100.0",
        "trainer.method_args.retain_loss_type=EMBED_DIFF",
        "trainer.method_args.steering_coeff=6.5",
        'trainer.method_args.module_regex="model[.]layers[.]31"',
        'trainer.method_args.trainable_params_regex=["model[.]layers[.](29|30|31)[.]mlp[.]down_proj[.]weight"]',
        "trainer.args.optim=adamw_torch",
        "+trainer.args.adam_beta1=0.9",
        "+trainer.args.adam_beta2=0.999",
        "+trainer.args.adam_epsilon=1e-6",
        "trainer.args.learning_rate=2e-4",
        "trainer.args.weight_decay=0.0",
        "+trainer.args.max_grad_norm=0.0",
        "trainer.args.per_device_train_batch_size=1",
        "trainer.args.per_device_eval_batch_size=1",
        "trainer.args.gradient_accumulation_steps=4",
        "+trainer.args.max_steps=-1",
        "trainer.args.num_train_epochs=10.0",
        "+trainer.args.lr_scheduler_type=constant",
        "+trainer.args.warmup_steps=0",
        "+trainer.args.warmup_epochs=0.0",
        "trainer.args.bf16=true",
        "trainer.args.bf16_full_eval=true",
        "+trainer.args.fp16=false",
        "trainer.args.gradient_checkpointing=false",
        "+trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
        "trainer.args.logging_steps=1",
    ),
    "muse_news": (
        "trainer.method_args.gamma=1.0",
        "trainer.method_args.alpha=10.0",
        "trainer.method_args.retain_loss_type=EMBED_DIFF",
        "trainer.method_args.steering_coeff=6.5",
        'trainer.method_args.module_regex="model[.]layers[.]7"',
        'trainer.method_args.trainable_params_regex=["model[.]layers[.](5|6|7)[.]mlp[.]down_proj[.]weight"]',
        "trainer.args.optim=adamw_torch",
        "+trainer.args.adam_beta1=0.9",
        "+trainer.args.adam_beta2=0.999",
        "+trainer.args.adam_epsilon=1e-6",
        "trainer.args.learning_rate=1e-3",
        "trainer.args.weight_decay=0.0",
        "+trainer.args.max_grad_norm=0.0",
        "trainer.args.per_device_train_batch_size=4",
        "trainer.args.per_device_eval_batch_size=1",
        "trainer.args.gradient_accumulation_steps=1",
        "trainer.args.num_train_epochs=10.0",
        "+trainer.args.lr_scheduler_type=constant",
        "+trainer.args.warmup_steps=0",
        "+trainer.args.warmup_epochs=0.0",
        "trainer.args.bf16=true",
        "trainer.args.bf16_full_eval=true",
        "+trainer.args.fp16=false",
        "trainer.args.gradient_checkpointing=false",
        "+trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
        "trainer.args.logging_steps=1",
    ),
    "muse_books": (
        "trainer.method_args.gamma=1.0",
        "trainer.method_args.alpha=10.0",
        "trainer.method_args.retain_loss_type=EMBED_DIFF",
        "trainer.method_args.steering_coeff=6.5",
        'trainer.method_args.module_regex="model[.]layers[.]7"',
        'trainer.method_args.trainable_params_regex=["model[.]layers[.](5|6|7)[.]mlp[.]down_proj[.]weight"]',
        "trainer.args.optim=adamw_torch",
        "+trainer.args.adam_beta1=0.9",
        "+trainer.args.adam_beta2=0.999",
        "+trainer.args.adam_epsilon=1e-6",
        "trainer.args.learning_rate=1e-3",
        "trainer.args.weight_decay=0.0",
        "+trainer.args.max_grad_norm=0.0",
        "trainer.args.per_device_train_batch_size=4",
        "trainer.args.per_device_eval_batch_size=1",
        "trainer.args.gradient_accumulation_steps=1",
        "trainer.args.num_train_epochs=1.0",
        "+trainer.args.lr_scheduler_type=constant",
        "+trainer.args.warmup_steps=0",
        "+trainer.args.warmup_epochs=0.0",
        "trainer.args.bf16=true",
        "trainer.args.bf16_full_eval=true",
        "+trainer.args.fp16=false",
        "trainer.args.gradient_checkpointing=false",
        "+trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
        "trainer.args.logging_steps=1",
    ),
    "wmdp": (
        "trainer.method_args.gamma=1.0",
        "trainer.method_args.alpha=1200.0",
        "trainer.method_args.retain_loss_type=EMBED_DIFF",
        "trainer.method_args.steering_coeff=6.5",
        'trainer.method_args.module_regex="model[.]layers[.]7"',
        'trainer.method_args.trainable_params_regex=["model[.]layers[.](5|6|7)[.]mlp[.]down_proj[.]weight"]',
        "trainer.args.optim=adamw_torch",
        "+trainer.args.adam_beta1=0.9",
        "+trainer.args.adam_beta2=0.999",
        "+trainer.args.adam_epsilon=1e-6",
        "trainer.args.learning_rate=5e-5",
        "trainer.args.weight_decay=0.0",
        "+trainer.args.max_grad_norm=0.0",
        "trainer.args.per_device_train_batch_size=4",
        "trainer.args.per_device_eval_batch_size=1",
        "trainer.args.gradient_accumulation_steps=1",
        "+trainer.args.max_steps=150",
        "+trainer.args.lr_scheduler_type=constant",
        "+trainer.args.warmup_steps=0",
        "+trainer.args.warmup_epochs=0.0",
        "trainer.args.bf16=true",
        "trainer.args.bf16_full_eval=true",
        "+trainer.args.fp16=false",
        "trainer.args.gradient_checkpointing=false",
        "+trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
        "trainer.args.logging_steps=1",
    ),
}
WMDP_RMU_OVERRIDES = (
    "data/datasets@data.retain=WMDP_wikitext_retain",
    "~data.retain.WMDP_retain",
    "data.forget.WMDP_forget.args.max_length=768",
    "model.model_args.attn_implementation=sdpa",
    "+model.model_args.use_cache=false",
    "+model.model_args.output_attentions=false",
    "eval.lm_eval.simple_evaluate_args.batch_size=8",
)


def _make_job(method, benchmark, seed, stage):
    job_id = f"{method}__{benchmark}__seed{seed}"
    job = {
        "job_id": job_id,
        "method": method,
        "benchmark": benchmark,
        "seed": seed,
        "stage": stage,
        "status": "pending",
        "trainer_config": method,
        **deepcopy(BENCHMARKS[benchmark]),
        "output_dir": f"jobs/{job_id}",
    }
    return job


def build_manifest(seed=0):
    """Return the deterministic manifest for one GU matrix seed."""
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    if seed != 0:
        raise ValueError("build_manifest only supports the Stage-1 seed 0")

    compatible_pairs = [
        (method, benchmark) for method in STAGE_ONE_METHODS for benchmark in BENCHMARKS
    ]
    jobs = sorted(
        (
            _make_job(method, benchmark, seed, STAGE)
            for method, benchmark in compatible_pairs
        ),
        key=lambda job: job["job_id"],
    )
    not_applicable = sorted(
        (
            {
                "method": "DPO",
                "benchmark": benchmark,
                "status": "not_applicable",
                "reason": "missing_shipped_idk_artifact",
                "provenance": {
                    "required_artifact": {
                        "path": "./data/idk.jsonl",
                        "status": "missing",
                    }
                },
            }
            for benchmark in BENCHMARKS
        ),
        key=lambda item: (item["method"], item["benchmark"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "stage": STAGE,
        "seed": seed,
        "jobs": jobs,
        "not_applicable": not_applicable,
    }


def _benchmark_family(job):
    return job["benchmark"].split("_", 1)[0]


def _benchmark_arguments(job):
    provenance = job["provenance"]
    family = _benchmark_family(job)
    if family == "tofu":
        retain_logs = (
            SHARED_ROOT
            / "saves/eval"
            / f'tofu_Llama-3.1-8B-Instruct_{job["retain_split"]}'
            / "TOFU_EVAL.json"
        )
        return (
            f'forget_split={job["split"]}',
            f'retain_split={job["retain_split"]}',
            f'holdout_split={job["holdout_split"]}',
            f"retain_logs_path={retain_logs}",
            f"tofu_dataset_revision={provenance['dataset']['revision']}",
            f"retain_logs_sha256={RETAIN_LOG_HASHES[job['benchmark']]}",
        )
    if family == "muse":
        split = job["split"]
        reference = provenance["reference_model"]
        reference_snapshot = (
            Path("/root/.cache/huggingface/hub")
            / f"models--muse-bench--MUSE-{split}_retrain"
            / "snapshots"
            / reference["revision"]
        )
        retain_logs = (
            SHARED_ROOT
            / "saves/eval"
            / f"muse_Llama-2-7b-hf_{split}_retrain"
            / "MUSE_EVAL.json"
        )
        return (
            f"data_split={split}",
            f"dataset_revision={provenance['dataset']['revision']}",
            f"reference_model_artifact={reference['artifact']}",
            f"reference_model_revision={reference['revision']}",
            f"reference_model_snapshot={reference_snapshot}",
            f"retain_logs_path={retain_logs}",
            f"muse_retain_logs_sha256={RETAIN_LOG_HASHES[job['benchmark']]}",
        )
    return ("data_split=cyber",)


def build_command(job, output_dir):
    """Build one checkpoint-free train-plus-live-evaluation command."""
    provenance = job["provenance"]
    command = [
        sys.executable,
        "src/train.py",
        "--config-name=unlearn.yaml",
        f'experiment={job["experiment_config"]}',
        f'trainer={job["trainer_config"]}',
        f'task_name={job["job_id"]}',
        f"paths.output_dir={output_dir}",
        (
            "model.model_args.pretrained_model_name_or_path="
            f'{provenance["model"]["artifact"]}'
        ),
        f'model.model_args.revision={provenance["model"]["revision"]}',
        (
            "model.tokenizer_args.pretrained_model_name_or_path="
            f'{provenance["tokenizer"]["artifact"]}'
        ),
        f'model.tokenizer_args.revision={provenance["tokenizer"]["revision"]}',
        *_benchmark_arguments(job),
        *(
            argument.format(seed=job["seed"])
            for argument in COMMON_RUNTIME_ARGUMENTS
        ),
        *(
            argument.format(parameter_regex=job["selected_parameter_regex"])
            for argument in COMMON_GU_ARGUMENTS
        ),
    ]
    if job["method"] == "RMU":
        family = _benchmark_family(job)
        recipe = job["benchmark"] if family == "muse" else family
        command.extend(RMU_OVERRIDES[recipe])
        if job["benchmark"] == "wmdp_cyber":
            command.extend(WMDP_RMU_OVERRIDES)
    return command


def environment_overrides(job):
    """Return the environment values enforced beyond the caller environment."""
    if job["benchmark"] == "wmdp_cyber":
        return dict(WMDP_OFFLINE_ENVIRONMENT)
    return {}


def build_environment(job):
    """Return the inherited process environment with benchmark enforcement."""
    environment = dict(os.environ)
    environment.update(environment_overrides(job))
    return environment


def source_requirements(job):
    """Return the immutable source evidence required before launching a job."""
    provenance = job["provenance"]
    requirements = {
        "model": deepcopy(provenance["model"]),
        "tokenizer": deepcopy(provenance["tokenizer"]),
    }
    family = _benchmark_family(job)
    if family == "tofu":
        retain_log = (
            SHARED_ROOT
            / "saves/eval"
            / f'tofu_Llama-3.1-8B-Instruct_{job["retain_split"]}'
            / "TOFU_EVAL.json"
        )
        requirements["retain_log"] = {
            "path": str(retain_log),
            "sha256": RETAIN_LOG_HASHES[job["benchmark"]],
        }
    elif family == "muse":
        retain_log = (
            SHARED_ROOT
            / "saves/eval"
            / f'muse_Llama-2-7b-hf_{job["split"]}_retrain'
            / "MUSE_EVAL.json"
        )
        requirements["retain_log"] = {
            "path": str(retain_log),
            "sha256": RETAIN_LOG_HASHES[job["benchmark"]],
        }
    else:
        evaluation_datasets = provenance["evaluation_datasets"]
        requirements["corpora"] = {
            "forget": deepcopy(provenance["forget_corpus"]),
            "retain": deepcopy(provenance["retain_corpus"]),
        }
        requirements["evaluation_cache"] = {
            "root": WMDP_EVALUATION_CACHE,
            "builders": {
                "wmdp_cyber": {
                    "config_root": str(
                        Path(WMDP_EVALUATION_CACHE)
                        / "cais___wmdp/wmdp-cyber/0.0.0"
                    ),
                    "builder_id": evaluation_datasets["wmdp_cyber"][
                        "cache_builder_sha"
                    ],
                    "required_config_count": 1,
                },
                "mmlu": {
                    "config_root": str(
                        Path(WMDP_EVALUATION_CACHE) / "hails___mmlu_no_train"
                    ),
                    "builder_id": evaluation_datasets["mmlu"]["cache_builder_sha"],
                    "required_config_count": 57,
                },
            },
        }
        if job["method"] == "RMU":
            requirements["retain_dataset"] = {
                "artifact": "wikitext",
                "subset": "wikitext-2-raw-v1",
                "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                "cache_dir": str(
                    Path(WMDP_EVALUATION_CACHE)
                    / "wikitext/wikitext-2-raw-v1/0.0.0"
                    / "b08601e04326c79dfdd32d625aee71d232d685c3"
                ),
            }
    return requirements


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(job):
    """Validate source evidence without creating or modifying job output."""
    requirements = source_requirements(job)
    registered_provenance = BENCHMARKS[job["benchmark"]]["provenance"]
    for label in ("model", "tokenizer"):
        source = requirements[label]
        if not source.get("artifact"):
            raise ValueError(f"{label} artifact is missing")
        if re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")) is None:
            raise ValueError(f"{label} revision must be a pinned 40-hex commit")
        if source != registered_provenance[label]:
            raise ValueError(f"{label} revision does not match the registry")
    if job["model"]["pretrained_model_name_or_path"] != requirements["model"][
        "artifact"
    ]:
        raise ValueError("model artifact does not match the registered job")

    retain_log = requirements.get("retain_log")
    if retain_log is not None:
        path = Path(retain_log["path"])
        if not path.is_file() or _file_sha256(path) != retain_log["sha256"]:
            raise ValueError(f"retain log source mismatch: {path}")

    for corpus in requirements.get("corpora", {}).values():
        path = Path(corpus["path"])
        if (
            not path.is_file()
            or path.stat().st_size != corpus["size_bytes"]
            or _file_sha256(path) != corpus["sha256"]
        ):
            raise ValueError(f"corpus source mismatch: {path}")

    builders = requirements.get("evaluation_cache", {}).get("builders", {})
    for name, builder in builders.items():
        config_root = Path(builder["config_root"])
        if name == "mmlu":
            directories = tuple(
                config_root.glob(f"*/0.0.0/{builder['builder_id']}")
            )
        else:
            directories = (config_root / builder["builder_id"],)
        if len(directories) != builder["required_config_count"] or not all(
            path.is_dir() for path in directories
        ):
            raise ValueError(f"evaluation cache builder mismatch: {name}")
    retain_dataset = requirements.get("retain_dataset")
    if retain_dataset is not None and not Path(retain_dataset["cache_dir"]).is_dir():
        raise ValueError("retain dataset cache mismatch: wikitext")
    return requirements


def _manifest_dry_run(seed):
    manifest = build_manifest(seed=seed)
    jobs = []
    validated = set()
    for job in manifest["jobs"]:
        validation_key = (job["benchmark"], job["method"] == "RMU")
        if validation_key not in validated:
            validate_sources(job)
            validated.add(validation_key)
        rendered = deepcopy(job)
        rendered["argv"] = build_command(job, job["output_dir"])
        rendered["environment"] = environment_overrides(job)
        rendered["source_requirements"] = source_requirements(job)
        jobs.append(rendered)
    manifest["jobs"] = jobs
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--seed", type=int, default=0)
    manifest_parser.add_argument("--seed0", action="store_const", const=0, dest="seed")
    manifest_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "manifest":
        payload = _manifest_dry_run(args.seed) if args.dry_run else build_manifest(args.seed)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
