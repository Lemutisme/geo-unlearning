"""Registry for the GU full-matrix experiment jobs."""

from copy import deepcopy


SCHEMA_VERSION = 1
PROTOCOL = "gu_full_matrix_20260724"

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

BENCHMARKS = {
    "tofu_forget01": {
        "split": "forget01",
        "retain_split": "retain99",
        "holdout_split": "holdout01",
        "experiment_config": "unlearn/tofu/default",
        "model": TOFU_MODEL,
        "evaluator_kind": "tofu",
        "selected_parameter_regex": TOFU_PARAMETER_REGEX,
    },
    "tofu_forget05": {
        "split": "forget05",
        "retain_split": "retain95",
        "holdout_split": "holdout05",
        "experiment_config": "unlearn/tofu/default",
        "model": TOFU_MODEL,
        "evaluator_kind": "tofu",
        "selected_parameter_regex": TOFU_PARAMETER_REGEX,
    },
    "tofu_forget10": {
        "split": "forget10",
        "retain_split": "retain90",
        "holdout_split": "holdout10",
        "experiment_config": "unlearn/tofu/default",
        "model": TOFU_MODEL,
        "evaluator_kind": "tofu",
        "selected_parameter_regex": TOFU_PARAMETER_REGEX,
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
    },
}

UNPINNED_PROVENANCE = {
    "model_revision": None,
    "tokenizer_revision": None,
    "dataset_revision": None,
    "local_corpus_sha256": None,
}


def _make_job(method, benchmark, seed):
    job_id = f"{method}__{benchmark}__seed{seed}"
    job = {
        "id": job_id,
        "method": method,
        "benchmark": benchmark,
        "seed": seed,
        "status": "pending",
        "trainer_config": method,
        **deepcopy(BENCHMARKS[benchmark]),
        "provenance": dict(UNPINNED_PROVENANCE),
        "output_dir": f"jobs/{job_id}",
    }
    if method == "DPO":
        job["experiment_config"] = "unlearn/tofu/idk"
    return job


def build_manifest(seed=0):
    """Return the deterministic manifest for one GU matrix seed."""
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
        (_make_job(method, benchmark, seed) for method, benchmark in compatible_pairs),
        key=lambda job: job["id"],
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
        "stage": f"seed{seed}",
        "seed": seed,
        "jobs": jobs,
        "not_applicable": not_applicable,
    }
