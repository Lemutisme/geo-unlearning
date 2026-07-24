"""Registry for the GU full-matrix experiment jobs."""

from copy import deepcopy


SCHEMA_VERSION = 1
PROTOCOL = "gu_full_matrix_20260724"
SEED_STAGES = {0: "stage1", 1: "stage2", 2: "stage2"}

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

GENERAL_METHODS = (
    "GradAscent",
    "GradDiff",
    "NPO",
    "SimNPO",
    "UNDIAL",
    "CEU",
    "WGA",
    "SatImp",
    "PDU",
)

TOFU_PARAMETER_REGEX = r"model[.]layers[.](29|30|31)[.]mlp[.]down_proj[.]weight"
SEVEN_B_PARAMETER_REGEX = r"model[.]layers[.](5|6|7)[.]mlp[.]down_proj[.]weight"

TOFU_MODEL = {
    "config": "Llama-3.1-8B-Instruct",
    "pretrained_model_name_or_path": (
        "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
    ),
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
        "experiment_config": "unlearn/tofu/default",
        "model": TOFU_MODEL,
        "evaluator_kind": "tofu",
        "selected_parameter_regex": TOFU_PARAMETER_REGEX,
        "provenance": TOFU_PROVENANCE,
    },
    "tofu_forget05": {
        "split": "forget05",
        "retain_split": "retain95",
        "holdout_split": "holdout05",
        "experiment_config": "unlearn/tofu/default",
        "model": TOFU_MODEL,
        "evaluator_kind": "tofu",
        "selected_parameter_regex": TOFU_PARAMETER_REGEX,
        "provenance": TOFU_PROVENANCE,
    },
    "tofu_forget10": {
        "split": "forget10",
        "retain_split": "retain90",
        "holdout_split": "holdout10",
        "experiment_config": "unlearn/tofu/default",
        "model": TOFU_MODEL,
        "evaluator_kind": "tofu",
        "selected_parameter_regex": TOFU_PARAMETER_REGEX,
        "provenance": TOFU_PROVENANCE,
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
        "selected_parameter_regex": SEVEN_B_PARAMETER_REGEX,
        "provenance": {
            "model": {
                "artifact": "muse-bench/MUSE-News_target",
                "revision": "a2f39769e9a0b98ec1cdd12f65e9962502208935",
            },
            "tokenizer": MUSE_TOKENIZER_PROVENANCE,
        },
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
        "selected_parameter_regex": SEVEN_B_PARAMETER_REGEX,
        "provenance": {
            "model": {
                "artifact": "muse-bench/MUSE-Books_target",
                "revision": "c8dd3fb23a726762ec66d277933c7cff6767f3c2",
            },
            "tokenizer": MUSE_TOKENIZER_PROVENANCE,
        },
    },
    "wmdp_cyber": {
        "split": "cyber",
        "experiment_config": "unlearn/wmdp/default",
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
                "sha256": (
                    "b5d339ed7f42a9e0dfc00708e516b288"
                    "363a87512ec9cbdad8703f0ea8f5ea9a"
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
        },
    },
}


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
    if method == "DPO":
        job["experiment_config"] = "unlearn/tofu/idk"
    return job


def build_manifest(seed=0):
    """Return the deterministic manifest for one GU matrix seed."""
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    if seed not in SEED_STAGES:
        raise ValueError("seed must be one of 0, 1, or 2")
    stage = SEED_STAGES[seed]

    compatible_pairs = [
        *(
            (method, benchmark)
            for method in GENERAL_METHODS
            for benchmark in BENCHMARKS
        ),
        *(("RMU", benchmark) for benchmark in BENCHMARKS),
        *(
            ("DPO", benchmark)
            for benchmark in BENCHMARKS
            if benchmark.startswith("tofu_")
        ),
    ]
    jobs = sorted(
        (
            _make_job(method, benchmark, seed, stage)
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
                "reason": "missing_shipped_preference_pairs",
            }
            for benchmark in BENCHMARKS
            if not benchmark.startswith("tofu_")
        ),
        key=lambda item: (item["method"], item["benchmark"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "stage": stage,
        "seed": seed,
        "jobs": jobs,
        "not_applicable": not_applicable,
    }
