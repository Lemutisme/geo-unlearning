"""Registry for the GU full-matrix experiment jobs."""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from subprocess import TimeoutExpired


SCHEMA_VERSION = 1
PROTOCOL = "gu_full_matrix_20260724"
STAGE = "stage1"
CODE_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = Path("/workspace/re/GU/geo-unlearning")
RUNTIME_ROOTS = {
    "tofu": Path("/dev/shm/gu-matrix-tofu"),
    "muse_news": Path("/dev/shm/gu-matrix-muse-news"),
    "muse_books": Path("/dev/shm/gu-matrix-muse-books"),
    "wmdp": Path("/dev/shm/gu-matrix-wmdp"),
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
    "trainer.args.logging_steps=1",
)
SHIPPED_GU_RETAIN_FILTER = "first_order"
COMMON_GU_ARGUMENTS = (
    "+trainer.method_args.gu.enabled=true",
    '+trainer.method_args.gu.parameter_regex=["{parameter_regex}"]',
    "+trainer.method_args.gu.retain_history_rank=8",
    "+trainer.method_args.gu.projection_eps=1e-6",
    f"+trainer.method_args.gu.retain_filter={SHIPPED_GU_RETAIN_FILTER}",
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
TOFU_DATASET_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"
TOFU_DATASET_MANIFESTS = {
    "forget01": "47544a749780b0f09e4eb009d96669e71e73f27c0da449a89db54ea9658b71ba",
    "forget01_perturbed": "0fe55d8e1174f33eb71e7496757e5f3723ab1f739b98ba4e14e56d81fe251be7",
    "forget05": "9c49fd6c2062158a5088ce016ef66b9b6196a4a7eac8cff91649d4696076c287",
    "forget05_perturbed": "318818aaf024859ee3e2bdf185c2a45b0167b6038604a7b9c5ca00de076171a7",
    "forget10": "9356e39f2fabc7b210500bc45b5ac4c293691c8a48c7ce124ac54ff727a5a0a5",
    "forget10_perturbed": "4723022889d02ea167f3c731cfeb6cf70f533ee2d4a28e59db0110ab3037114e",
    "holdout01": "66b125103d0156efc669531c249a06b2c397ef46fcaf7552168c6616b49462d6",
    "holdout05": "55f57326e122bc07370224f79169ec71a34a9ca8a3a46d820fbc1ab8dce6527e",
    "holdout10": "b86431331dfee18abaf6ed2250b90a02522f39f3ee42c974bf50e7e24ee2e6b3",
    "real_authors_perturbed": "accbace65d282583b514925764f33d9493d9d4e06297ad752993715854fce40b",
    "retain90": "675ea11a4faedbe3d2e2559d68d422247499605172f150e82028cfc8dcc9e99e",
    "retain95": "201eb2b0f84a30a66f62752743fd6a86f19bd243e6e3d53efffb900f4fd7e179",
    "retain99": "d78777950d537d53db4b441c3857e86facf04cfb417532de5ef5dbd070a05027",
    "retain_perturbed": "ff25a0d8d85c3ba40c55a8483585b780c42e1d42ddc93a037c44815ed083c378",
    "world_facts_perturbed": "bd5f7aedc57ba20d93c9b55106cd64b6ce22cee0759fe51d790d8b9536b23c9a",
}
MUSE_DATASET_CACHE = {
    "News": {
        "revision": "506bd5b150b92814d45e4404a82f120ab2d748bf",
        "manifests": {
            "raw": "e06cff3f2f6e6539bdc93bff1e9c02ce741a6f5a1336a0d377abae557082e1c4",
            "knowmem": "fb4c8e02ab1d1ef2536213721fb6d401f6ec9a6fb6d0aba39daf9085563d4f0f",
            "verbmem": "4bcebf1eea8341165e7b86d141bc5e0d3e27a466bbab7653254b34d265ffd6ea",
            "privleak": "33063ca090ed95c289f31b813071295ff382cf67fcb65588b3876ff737d57525",
        },
    },
    "Books": {
        "revision": "051ba90319e920d410d87cfdbd61f25843c1b892",
        "manifests": {
            "raw": "971a0079974ee7ae32151e39c4e10ca9895e76ec7182245f0b5835513e993e9b",
            "knowmem": "7c6c9e55ecf9a9a615a71867529b5fd78daf9837626c438f87f0054174bd7043",
            "verbmem": "6f5a1ecf4b963557787636fb6bb804d6378e6a4c095ddd36b935f328e0531f4f",
            "privleak": "bc0dcaec1b6a91e16ef70eee98086b1a960ef3e366aeb0ff7436c0efb87d4dce",
        },
    },
}
WMDP_DATASET_MANIFESTS = {
    "wmdp_cyber": "17a1eeaaa80c6baf887d993043a271f97ce4c59f39feb6e2d5e782d30e2f3a77",
    "mmlu": "e0636c613ceaa7efb3ab181587f6bf7cf62e3c4d2ef2f32b3c4b528f93dc8228",
    "wikitext": "3b162501be71321b5c3a05ecab71bb452376295224b5060ce0108cb05a52d5ce",
}
HUB_SNAPSHOTS = {
    "tofu_target_tokenizer": {
        "source_name": "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
        "revision": "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e",
        "content_manifest_sha256": "f9879ff7688d9233da6ff1c060ad3d465aaaf026d0c9e20372add28dc2c44fd3",
    },
    "muse_news_target": {
        "source_name": "muse-bench/MUSE-News_target",
        "revision": "a2f39769e9a0b98ec1cdd12f65e9962502208935",
        "content_manifest_sha256": "7cb73ebda17432c1f48092f243eecdc7ce0165be644d4c506ba3dee86ae6011c",
    },
    "muse_books_target": {
        "source_name": "muse-bench/MUSE-Books_target",
        "revision": "c8dd3fb23a726762ec66d277933c7cff6767f3c2",
        "content_manifest_sha256": "752e2e94eb6b12f4f8e5514def431da122a25073a9b6db3da06b7801f8692795",
    },
    "muse_tokenizer": {
        "source_name": "NousResearch/Llama-2-7b-hf",
        "revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
        "content_manifest_sha256": "8f8ec553756a42d2fa5e3b630aba8cf49c8a8afa90664527d4d295c07c1000f2",
    },
    "muse_news_reference": {
        "source_name": "muse-bench/MUSE-News_retrain",
        "revision": "324ef49ee0a038078aba7d8de831edf57235c9b3",
        "content_manifest_sha256": "f7638ac028811c84e246514b838eb7cd40e39c41fea3ea518e449252186c5868",
    },
    "muse_books_reference": {
        "source_name": "muse-bench/MUSE-Books_retrain",
        "revision": "1d67430e4e8bdf2a65823740e909792519175ac2",
        "content_manifest_sha256": "6ff6a81660877991ae44d4118441ad2815c95ff8bc2d54f2256728ba7a1ea434",
    },
    "gibberish_classifier": {
        "source_name": "madhurjindal/autonlp-Gibberish-Detector-492513457",
        "revision": "76672dd7d3575f68ab980705bcec975cc62de71c",
        "content_manifest_sha256": "3d0e600d1f642bd55c16195eb8238fbb86ad17dab0a49b08bc18f4dcdd05b87c",
    },
    "wmdp_model_tokenizer": {
        "source_name": "HuggingFaceH4/zephyr-7b-beta",
        "revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c",
        "content_manifest_sha256": "64012eb3be1ba3915533f844707cb10b0ccde9fc503a6de6240ac5073907ae7b",
    },
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

GU_DIAGNOSTIC_FIELDS = {
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
NUMERIC_DIAGNOSTIC_FIELDS = GU_DIAGNOSTIC_FIELDS - {
    "step",
    "objective",
    "selected_parameter_count",
    "constraint_count",
    "active_constraints",
    "zero_step",
    "zero_step_reason",
    "optimizer_state_semantics",
    "retain_loss_before",
    "retain_loss_after",
}
INTEGER_DIAGNOSTIC_FIELDS = {
    "step",
    "selected_parameter_count",
    "constraint_count",
}
ENDPOINT_PREFIXES = {
    "tofu": "TOFU",
    "muse": "MUSE",
    "lm_eval": "LMEval",
}
NVML_QUERY_COMMAND = [
    "nvidia-smi",
    "--query-compute-apps=pid,used_memory",
    "--format=csv,noheader",
]
GPU_ADMISSION_COMMAND = [
    "nvidia-smi",
    "--query-gpu=index,memory.used",
    "--format=csv,noheader,nounits",
]
TERMINAL_STATUSES = {
    "completed",
    "invalid_scientific",
    "failed_infrastructure",
}
IMMUTABLE_JOB_FIELDS = (
    "method",
    "benchmark",
    "split",
    "retain_split",
    "holdout_split",
    "seed",
    "stage",
    "trainer_config",
    "experiment_config",
    "model",
    "evaluator_kind",
    "selected_parameter_regex",
    "provenance",
    "output_dir",
)
RMU_MODEL_OVERRIDES = {
    "tofu": (
        "model.model_args.attn_implementation=flash_attention_2",
        "+model.model_args.use_cache=false",
        "+model.model_args.output_attentions=false",
    ),
    "muse": (
        "model.model_args.attn_implementation=sdpa",
        "+model.model_args.use_cache=false",
        "+model.model_args.output_attentions=false",
    ),
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


def runtime_root(job):
    family = _benchmark_family(job)
    key = job["benchmark"] if family == "muse" else family
    return RUNTIME_ROOTS[key]


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
            runtime_root(job)
            / "hub"
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
    resolved_output = Path(output_dir).resolve()
    command = [
        sys.executable,
        str(CODE_ROOT / "src/train.py"),
        "--config-name=unlearn.yaml",
        f'experiment={job["experiment_config"]}',
        f'trainer={job["trainer_config"]}',
        f'task_name={job["job_id"]}',
        f"paths.output_dir={resolved_output}",
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
        if family in RMU_MODEL_OVERRIDES:
            command.extend(RMU_MODEL_OVERRIDES[family])
        if job["benchmark"] == "wmdp_cyber":
            command.extend(WMDP_RMU_OVERRIDES)
    return command


def environment_overrides(job):
    """Return the environment values enforced beyond the caller environment."""
    root = runtime_root(job)
    hub = str(root / "hub")
    return {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_DATASETS_CACHE": str(root / "datasets"),
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": hub,
        "HF_HUB_CACHE": hub,
        "HF_HUB_OFFLINE": "1",
        "HUGGINGFACE_HUB_CACHE": hub,
        "TRANSFORMERS_CACHE": hub,
    }


def build_environment(job):
    """Return the inherited process environment with benchmark enforcement."""
    environment = dict(os.environ)
    environment.update(environment_overrides(job))
    return environment


def _dataset_cache_requirement(source_name, path, content_manifest_sha256):
    return {
        "kind": "dataset_cache",
        "source_name": source_name,
        "path": str(path),
        "content_manifest_sha256": content_manifest_sha256,
        "registered_suffixes": [".arrow", ".json"],
    }


def _hub_snapshot_requirement(key, job):
    requirement = deepcopy(HUB_SNAPSHOTS[key])
    requirement["kind"] = "hub_snapshot"
    owner, repository = requirement["source_name"].split("/", 1)
    requirement["path"] = str(
        runtime_root(job)
        / "hub"
        / f"models--{owner}--{repository}"
        / "snapshots"
        / requirement["revision"]
    )
    return requirement


def _content_requirements(job):
    family = _benchmark_family(job)
    if family == "tofu":
        dataset_names = (
            job["split"],
            job["retain_split"],
            f'{job["split"]}_perturbed',
            "retain_perturbed",
            "real_authors_perturbed",
            "world_facts_perturbed",
            job["holdout_split"],
        )
        requirements = [
            _hub_snapshot_requirement("tofu_target_tokenizer", job),
            _hub_snapshot_requirement("gibberish_classifier", job),
        ]
        requirements.extend(
            _dataset_cache_requirement(
                f"locuslab/TOFU:{name}",
                runtime_root(job)
                / "datasets/locuslab___tofu"
                / name
                / "0.0.0"
                / TOFU_DATASET_REVISION,
                TOFU_DATASET_MANIFESTS[name],
            )
            for name in dataset_names
        )
        return requirements
    if family == "muse":
        split = job["split"]
        cache = MUSE_DATASET_CACHE[split]
        target_key = f"muse_{split.lower()}_target"
        reference_key = f"muse_{split.lower()}_reference"
        requirements = [
            _hub_snapshot_requirement(target_key, job),
            _hub_snapshot_requirement("muse_tokenizer", job),
            _hub_snapshot_requirement(reference_key, job),
            _hub_snapshot_requirement("gibberish_classifier", job),
        ]
        requirements.extend(
            _dataset_cache_requirement(
                f"muse-bench/MUSE-{split}:{name}",
                runtime_root(job)
                / "datasets"
                / f"muse-bench___muse-{split.lower()}"
                / name
                / "0.0.0"
                / cache["revision"],
                manifest,
            )
            for name, manifest in cache["manifests"].items()
        )
        return requirements
    requirements = [
        _hub_snapshot_requirement("wmdp_model_tokenizer", job),
        _dataset_cache_requirement(
            "cais/wmdp:wmdp_cyber",
            runtime_root(job)
            / "datasets"
            / "cais___wmdp/wmdp-cyber/0.0.0"
            / "7125571f22f032c56415e7980f48d877dd830ff8",
            WMDP_DATASET_MANIFESTS["wmdp_cyber"],
        ),
        _dataset_cache_requirement(
            "hails/mmlu_no_train",
            runtime_root(job) / "datasets/hails___mmlu_no_train",
            WMDP_DATASET_MANIFESTS["mmlu"],
        ),
    ]
    if job["method"] == "RMU":
        requirements.append(
            _dataset_cache_requirement(
                "wikitext:wikitext-2-raw-v1",
                runtime_root(job)
                / "datasets"
                / "wikitext/wikitext-2-raw-v1/0.0.0"
                / "b08601e04326c79dfdd32d625aee71d232d685c3",
                WMDP_DATASET_MANIFESTS["wikitext"],
            )
        )
    return requirements


def source_requirements(job):
    """Return the immutable source evidence required before launching a job."""
    provenance = job["provenance"]
    requirements = {
        "model": deepcopy(provenance["model"]),
        "tokenizer": deepcopy(provenance["tokenizer"]),
        "content_requirements": _content_requirements(job),
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
        datasets_root = runtime_root(job) / "datasets"
        requirements["corpora"] = {
            "forget": deepcopy(provenance["forget_corpus"]),
            "retain": deepcopy(provenance["retain_corpus"]),
        }
        requirements["evaluation_cache"] = {
            "root": str(datasets_root),
            "builders": {
                "wmdp_cyber": {
                    "config_root": str(
                        datasets_root
                        / "cais___wmdp/wmdp-cyber/0.0.0"
                    ),
                    "builder_id": evaluation_datasets["wmdp_cyber"][
                        "cache_builder_sha"
                    ],
                    "required_config_count": 1,
                },
                "mmlu": {
                    "config_root": str(
                        datasets_root / "hails___mmlu_no_train"
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
                    datasets_root
                    / "wikitext/wikitext-2-raw-v1/0.0.0"
                    / "b08601e04326c79dfdd32d625aee71d232d685c3"
                ),
            }
    return requirements


def _file_sha256(path):
    return _file_digests(path)[0]


def _file_digests(path):
    size = path.stat().st_size
    digest = hashlib.sha256()
    git_digest = hashlib.sha1()
    git_digest.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            git_digest.update(chunk)
    return digest.hexdigest(), git_digest.hexdigest()


def _manifest_fingerprint(entries):
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_directory_fingerprint(path, registered_suffixes=None):
    """Fingerprint relative paths, sizes, and file SHA256 values canonically."""
    root = Path(path)
    entries = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        if registered_suffixes is not None and candidate.suffix not in set(
            registered_suffixes
        ):
            continue
        entries.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "size": candidate.stat().st_size,
                "sha256": _file_sha256(candidate),
            }
        )
    if not entries:
        raise ValueError(f"cache contains no files: {root}")
    return _manifest_fingerprint(entries)


def _hub_snapshot_fingerprint(requirement):
    snapshot = Path(requirement["path"])
    if snapshot.name != requirement["revision"]:
        raise ValueError(f"{requirement['source_name']}: revision directory mismatch")
    blob_root = (snapshot.parent.parent / "blobs").resolve()
    entries = []
    for candidate in sorted(snapshot.rglob("*")):
        if not candidate.is_symlink() and not candidate.is_file():
            continue
        if not candidate.is_symlink():
            raise ValueError(
                f"{requirement['source_name']}: snapshot entry is not a symlink"
            )
        try:
            target = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                f"{requirement['source_name']}: snapshot contains a broken symlink"
            ) from error
        if not target.is_relative_to(blob_root):
            raise ValueError(
                f"{requirement['source_name']}: snapshot symlink escapes blob storage"
            )
        sha256, git_sha1 = _file_digests(target)
        blob_name = target.name
        if len(blob_name) == 64 and sha256 != blob_name:
            raise ValueError(
                f"{requirement['source_name']}: content-addressed blob hash mismatch"
            )
        if len(blob_name) == 40 and git_sha1 != blob_name:
            raise ValueError(
                f"{requirement['source_name']}: content-addressed blob hash mismatch"
            )
        if len(blob_name) not in {40, 64} or re.fullmatch(
            r"[0-9a-f]+", blob_name
        ) is None:
            raise ValueError(
                f"{requirement['source_name']}: invalid content-addressed blob name"
            )
        entries.append(
            {
                "path": candidate.relative_to(snapshot).as_posix(),
                "size": target.stat().st_size,
                "sha256": sha256,
            }
        )
    if not entries:
        raise ValueError(f"{requirement['source_name']}: snapshot contains no files")
    return _manifest_fingerprint(entries)


def validate_content_requirement(requirement, fingerprint_memo):
    """Validate one directory source, memoized by its canonical requirement JSON."""
    memo_key = json.dumps(requirement, sort_keys=True, separators=(",", ":"))
    if memo_key in fingerprint_memo:
        return fingerprint_memo[memo_key]
    source_name = requirement["source_name"]
    path = Path(requirement["path"])
    if not path.is_dir():
        raise ValueError(f"{source_name}: required cache directory is absent")
    expected = requirement.get("content_manifest_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", expected or "") is None:
        raise ValueError(f"{source_name}: expected content manifest is unavailable")
    if requirement["kind"] == "hub_snapshot":
        observed = _hub_snapshot_fingerprint(requirement)
    else:
        suffixes = set(requirement.get("registered_suffixes", ()))
        observed_suffixes = {
            candidate.suffix for candidate in path.rglob("*") if candidate.is_file()
        }
        if not suffixes.issubset(observed_suffixes):
            raise ValueError(f"{source_name}: Arrow or metadata files are missing")
        observed = canonical_directory_fingerprint(path, suffixes)
    if observed != expected:
        raise ValueError(f"{source_name}: content manifest mismatch")
    fingerprint_memo[memo_key] = observed
    return observed


def _validate_file_requirement(requirement, fingerprint_memo):
    memo_key = json.dumps(requirement, sort_keys=True, separators=(",", ":"))
    if memo_key in fingerprint_memo:
        return fingerprint_memo[memo_key]
    path = Path(requirement["path"])
    if not path.is_file():
        raise ValueError(f"{requirement['source_name']}: required file is absent")
    if "size_bytes" in requirement and path.stat().st_size != requirement["size_bytes"]:
        raise ValueError(f"{requirement['source_name']}: file size mismatch")
    observed = _file_sha256(path)
    if observed != requirement["sha256"]:
        raise ValueError(f"{requirement['source_name']}: file hash mismatch")
    fingerprint_memo[memo_key] = observed
    return observed


def validate_sources(job, fingerprint_memo=None):
    """Validate source evidence without creating or modifying job output."""
    if fingerprint_memo is None:
        fingerprint_memo = {}
    requirements = source_requirements(job)
    fingerprints = {}
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
        file_requirement = {
            "kind": "file",
            "source_name": f"retain log:{job['benchmark']}",
            **retain_log,
        }
        fingerprints[file_requirement["source_name"]] = _validate_file_requirement(
            file_requirement, fingerprint_memo
        )

    for name, corpus in requirements.get("corpora", {}).items():
        file_requirement = {
            "kind": "file",
            "source_name": f"wmdp_{name}_corpus",
            **corpus,
        }
        fingerprints[file_requirement["source_name"]] = _validate_file_requirement(
            file_requirement, fingerprint_memo
        )

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
    for requirement in requirements["content_requirements"]:
        fingerprints[requirement["source_name"]] = validate_content_requirement(
            requirement, fingerprint_memo
        )
    return fingerprints


def _atomic_write_json(path, payload):
    """Durably replace one JSON record without exposing a partial file."""
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    contents = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    try:
        with temporary.open("wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _command_record(job, output_dir):
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "job_id": job["job_id"],
        "method": job["method"],
        "benchmark": job["benchmark"],
        "seed": job["seed"],
        "stage": job["stage"],
        "argv": build_command(job, output_dir),
        "environment_overrides": environment_overrides(job),
        "provenance": deepcopy(job["provenance"]),
    }


def _command_identity(record):
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _finite_tree(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    return False


def validate_job_identity(job, *, parent_job=None):
    """Return an exact registered Stage-1 job or parent-derived Stage-2 job."""
    if parent_job is not None:
        registered_parent = validate_job_identity(parent_job)
        expected = deepcopy(registered_parent)
        seed = job.get("seed") if isinstance(job, dict) else None
        if seed not in {1, 2}:
            raise ValueError("derived Stage-2 job seed must be 1 or 2")
        expected.update(
            {
                "job_id": registered_parent["job_id"].replace("seed0", f"seed{seed}"),
                "seed": seed,
                "stage": "stage2",
                "status": "pending",
                "output_dir": registered_parent["output_dir"].replace(
                    "seed0", f"seed{seed}"
                ),
                "parent_seed_zero": registered_parent["job_id"],
            }
        )
        missing = object()
        fields = (*IMMUTABLE_JOB_FIELDS, "job_id", "parent_seed_zero")
        mismatches = [
            field
            for field in fields
            if job.get(field, missing) != expected.get(field, missing)
        ]
        if mismatches:
            raise ValueError(
                "derived Stage-2 job does not match its seed-zero parent: "
                + ",".join(mismatches)
            )
        return expected
    if not isinstance(job, dict) or not isinstance(job.get("job_id"), str):
        raise ValueError("job is not an exact registered Stage-1 job: invalid job_id")
    registered = {
        candidate["job_id"]: candidate for candidate in build_manifest(seed=0)["jobs"]
    }
    expected = registered.get(job["job_id"])
    if expected is None:
        raise ValueError("job is not an exact registered Stage-1 job: unknown job_id")
    missing = object()
    mismatches = [
        field
        for field in IMMUTABLE_JOB_FIELDS
        if job.get(field, missing) != expected.get(field, missing)
    ]
    if mismatches:
        raise ValueError(
            "job is not an exact registered Stage-1 job: immutable mismatch "
            + ",".join(mismatches)
        )
    return expected


def run_job(job, output_dir, *, parent_job=None, launched=None):
    """Run one unchanged matrix command and persist its complete evidence audit."""
    validate_job_identity(job, parent_job=parent_job)
    output = Path(output_dir).resolve()
    if output.exists():
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.mkdir(parents=True)

    command_path = output / "command.json"
    run_log_path = output / "run.log"
    diagnostics_path = output / "gu_diagnostics.jsonl"
    result_path = output / "JOB_RESULT.json"
    expected_command_record = _command_record(job, output)
    _atomic_write_json(command_path, expected_command_record)

    # Phase 1: launch once and sample only the direct child process.
    issues, infrastructure_issues = [], []
    peak_nvml_mib = gpu_memory_sample_count = 0
    launch_error = returncode = None
    started = time.monotonic()
    with run_log_path.open("w", encoding="utf-8") as run_log:
        try:
            process = subprocess.Popen(
                expected_command_record["argv"], cwd=CODE_ROOT,
                env=build_environment(job), stdout=run_log, stderr=subprocess.STDOUT,
            )
        except OSError as error:
            launch_error = str(error)
            run_log.write(f"subprocess launch failed: {error}\n")
        else:
            if launched is not None:
                launched(process.pid)
            while True:
                used_memory, observed_child, sampling_error = 0, False, None
                try:
                    sample = subprocess.run(
                        NVML_QUERY_COMMAND, capture_output=True, text=True,
                        check=False, timeout=5.0,
                    )
                except TimeoutExpired as error:
                    sampling_error = f"NVML sampling timed out after {error.timeout} seconds"
                except OSError as error:
                    sampling_error = f"NVML sampling launch failed: {error}"
                else:
                    if sample.returncode != 0:
                        detail = (sample.stderr or "").strip()
                        sampling_error = f"NVML sampling failed with exit {sample.returncode}: {detail}"
                    else:
                        for line in sample.stdout.splitlines():
                            fields = [field.strip() for field in line.split(",", 1)]
                            if len(fields) != 2:
                                continue
                            try:
                                observed_pid = int(fields[0])
                                memory = int(re.sub(r"\s*MiB\s*$", "", fields[1]))
                            except ValueError:
                                continue
                            if observed_pid == process.pid:
                                observed_child = True
                                used_memory = max(used_memory, memory)
                peak_nvml_mib = max(peak_nvml_mib, used_memory)
                gpu_memory_sample_count += int(observed_child)
                if sampling_error is not None and sampling_error not in infrastructure_issues:
                    infrastructure_issues.append(sampling_error)
                returncode = process.poll()
                if returncode is not None:
                    break
                time.sleep(1.0)
            if gpu_memory_sample_count == 0 or peak_nvml_mib <= 0:
                infrastructure_issues.append(
                    "GPU monitor never observed positive memory for the child PID"
                )
    wall_clock_seconds = max(0.0, time.monotonic() - started)
    log_text = run_log_path.read_text(errors="replace")

    # Phase 2: classify the process exit before monitor-only failures.
    exit_status = failure_kind = None
    if launch_error is not None:
        exit_status, failure_kind = "failed_infrastructure", "subprocess_launch"
        infrastructure_issues.append(f"subprocess launch failed: {launch_error}")
    elif returncode != 0:
        lowered = log_text.lower()
        if any(marker in lowered for marker in ("outofmemory", "out of memory", "cuda oom")):
            exit_status, failure_kind = "invalid_scientific", "oom"
        elif any(marker in lowered for marker in ("nan", "nonfinite", "non-finite", "infinite")):
            exit_status, failure_kind = "invalid_scientific", "nonfinite"
        elif "wrong sign" in lowered:
            exit_status, failure_kind = "invalid_scientific", "wrong_sign"
        elif any(marker in lowered for marker in ("gu rejection", "retain_budget_exceeded")):
            exit_status, failure_kind = "invalid_scientific", "gu_rejection"
        elif "zero step" in lowered or "zero_step" in lowered:
            exit_status, failure_kind = "invalid_scientific", "zero_step"
        elif any(
            marker in lowered
            for marker in (
                "input/output error", "i/o error", "no space left on device",
                "read-only file system", "stale file handle",
            )
        ):
            exit_status, failure_kind = "failed_infrastructure", "host_io"
        elif "cache" in lowered and any(
            marker in lowered
            for marker in ("absent", "missing", "corrupt", "mismatch", "failed", "error")
        ):
            exit_status, failure_kind = "failed_infrastructure", "cache"
        else:
            exit_status, failure_kind = "invalid_scientific", "subprocess_exit"

    # Phase 3: require one exact live evaluator pair with finite numeric content.
    endpoint_summary_path = endpoint_raw_path = None
    try:
        prefix = ENDPOINT_PREFIXES[job["evaluator_kind"]]
        checkpoint_pattern = re.compile(r"checkpoint-[0-9]+")
        summaries = sorted(
            path for path in output.rglob(f"{prefix}_SUMMARY.json")
            if path.parent.name == "evals"
            and checkpoint_pattern.fullmatch(path.parent.parent.name) is not None
        )
        raw_files = sorted(
            path for path in output.rglob(f"{prefix}_EVAL.json")
            if path.parent.name == "evals"
            and checkpoint_pattern.fullmatch(path.parent.parent.name) is not None
        )
        if len(summaries) != 1 or len(raw_files) != 1:
            raise ValueError(
                f"endpoint pair count mismatch: summary={len(summaries)}, raw={len(raw_files)}"
            )
        summary_path, raw_path = summaries[0], raw_files[0]
        if summary_path.parent != raw_path.parent:
            raise ValueError("endpoint summary and raw evaluation are in different checkpoints")
        for label, path in (("summary", summary_path), ("raw", raw_path)):
            payload = json.loads(
                path.read_text(),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant: {value}")
                ),
            )
            if not isinstance(payload, dict) or not payload:
                raise ValueError(f"{label} endpoint schema must be a nonempty object")
            if not _finite_tree(payload):
                raise ValueError(f"{label} endpoint contains nonfinite or unsupported values")
            pending, contains_number = list(payload.values()), False
            while pending:
                value = pending.pop()
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    contains_number = True
                    break
                if isinstance(value, dict):
                    pending.extend(value.values())
                elif isinstance(value, list):
                    pending.extend(value)
            if not contains_number:
                raise ValueError(f"{label} endpoint contains no numeric result")
        endpoint_summary_path = summary_path.relative_to(output).as_posix()
        endpoint_raw_path = raw_path.relative_to(output).as_posix()
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as error:
        issues.append(str(error))

    # Phase 4: parse and validate the producer's exact GU JSONL schema.
    diagnostics = []
    try:
        if not diagnostics_path.is_file():
            raise ValueError("GU diagnostics file is missing")
        lines = diagnostics_path.read_text().splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError("GU diagnostics JSONL is empty or contains blank records")
        for line_number, line in enumerate(lines, start=1):
            record = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant: {value}")
                ),
            )
            if not isinstance(record, dict) or set(record) != GU_DIAGNOSTIC_FIELDS:
                raise ValueError(f"GU diagnostics record {line_number} has the wrong schema")
            if not _finite_tree(record):
                raise ValueError(f"GU diagnostics record {line_number} is nonfinite")
            if any(
                isinstance(record[name], bool) or not isinstance(record[name], int)
                for name in INTEGER_DIAGNOSTIC_FIELDS
            ):
                raise ValueError(f"GU diagnostics record {line_number} has a non-integer field")
            if any(
                isinstance(record[name], bool) or not isinstance(record[name], (int, float))
                for name in NUMERIC_DIAGNOSTIC_FIELDS
            ):
                raise ValueError(f"GU diagnostics record {line_number} has a nonnumeric field")
            if record["constraint_count"] <= 0:
                raise ValueError(f"GU diagnostics record {line_number} constraint_count must be positive")
            active = record["active_constraints"]
            if (
                not isinstance(active, list)
                or any(isinstance(index, bool) or not isinstance(index, int) for index in active)
                or active != sorted(set(active))
                or any(index < 0 or index >= record["constraint_count"] for index in active)
            ):
                raise ValueError(f"GU diagnostics record {line_number} active_constraints is malformed")
            retain_losses = (record["retain_loss_before"], record["retain_loss_after"])
            if retain_losses == (None, None):
                if SHIPPED_GU_RETAIN_FILTER != "first_order":
                    raise ValueError(f"GU diagnostics record {line_number} retain losses are missing")
            elif any(
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in retain_losses
            ):
                raise ValueError(f"GU diagnostics record {line_number} retain losses are malformed")
            if record["step"] != line_number:
                raise ValueError("GU diagnostics step ids are not sequential from one")
            if record["objective"] != job["method"]:
                raise ValueError("GU diagnostics objective does not match the job")
            if record["selected_parameter_count"] <= 0:
                raise ValueError("GU selected parameter count must be positive")
            if not isinstance(record["zero_step"], bool):
                raise ValueError("GU zero_step must be boolean")
            if record["optimizer_state_semantics"] != "proposal_state_committed":
                raise ValueError("GU optimizer state semantics mismatch")
            if record["proposal_norm"] < 0 or record["corrected_norm"] < 0:
                raise ValueError("GU diagnostic norm has the wrong sign")
            if record["correction_ratio"] < 0 or record["applied_scale"] < 0:
                raise ValueError("GU correction or applied scale has the wrong sign")
            tolerance = record["projection_tolerance"]
            if tolerance < 0:
                raise ValueError("GU projection tolerance has the wrong sign")
            if not 0 <= record["kkt_residual"] <= tolerance:
                raise ValueError("GU KKT residual is outside projection tolerance")
            if record["max_violation_after"] > tolerance:
                raise ValueError("GU final projection violation exceeds tolerance")
            diagnostics.append(record)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        issues.append(f"GU diagnostics invalid: {error}")

    # Phase 5: count optimizer updates independently and summarize GU records.
    optimizer_update_count = 0
    final_global_step = None
    try:
        loss_records = []
        for line in log_text.splitlines():
            opening, closing = line.find("{"), line.rfind("}")
            record = None
            if opening >= 0 and closing >= opening:
                candidate = line[opening : closing + 1]
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        value = ast.literal_eval(candidate)
                    except (SyntaxError, ValueError):
                        value = None
                record = value if isinstance(value, dict) else None
            if record is not None and "loss" in record:
                if (
                    not _finite_tree(record)
                    or isinstance(record["loss"], bool)
                    or not isinstance(record["loss"], (int, float))
                ):
                    raise ValueError("Trainer loss log is nonfinite or nonnumeric")
                loss_records.append(record)
            if record is not None and "global_step" in record:
                step = record["global_step"]
                if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                    raise ValueError("Trainer final global_step is invalid")
                final_global_step = step
            matches = re.findall(r"global_step\s*[=:]\s*([0-9]+)", line)
            if matches:
                final_global_step = int(matches[-1])
        optimizer_update_count = len(loss_records)
        if optimizer_update_count == 0:
            raise ValueError("Trainer optimizer update loss logs are missing")
        if final_global_step is not None and final_global_step != optimizer_update_count:
            raise ValueError("Trainer final global_step and optimizer update loss-log count mismatch")
    except (OSError, UnicodeError, ValueError) as error:
        issues.append(str(error))

    projection_count = len(diagnostics)
    if projection_count != optimizer_update_count:
        issues.append(
            f"GU projection and optimizer update count mismatch: "
            f"{projection_count} != {optimizer_update_count}"
        )
    zero_step_count = sum(record["zero_step"] for record in diagnostics)
    rejected_steps = [
        record["step"]
        for record in diagnostics
        if record["zero_step"] or record["zero_step_reason"] not in (None, "")
    ]
    if rejected_steps:
        issues.append(
            "GU zero-step or rejection diagnostics are present at steps: "
            + ",".join(str(step) for step in rejected_steps)
        )
    selected_parameter_changed = any(
        not record["zero_step"]
        and record["corrected_norm"] > 0
        and record["applied_scale"] > 0
        for record in diagnostics
    )
    if diagnostics and (not selected_parameter_changed or zero_step_count == projection_count):
        issues.append("selected parameters are unchanged because every GU step is zero")
    scales = Counter(record["applied_scale"] for record in diagnostics)
    applied_scale_distribution = {str(scale): count for scale, count in sorted(scales.items())}
    ratios = [record["correction_ratio"] for record in diagnostics]
    correction_ratio = (
        {"min": min(ratios), "max": max(ratios), "mean": sum(ratios) / len(ratios)}
        if ratios
        else None
    )
    max_violation_after = max(
        (record["max_violation_after"] for record in diagnostics), default=None
    )

    # Phase 6: audit every file and nonempty directory against an allowlist.
    prefix = ENDPOINT_PREFIXES[job["evaluator_kind"]]
    checkpoint_pattern = re.compile(r"checkpoint-[0-9]+")
    root_files = {
        "command.json", "run.log", f'{job["method"]}.log',
        "gu_diagnostics.jsonl", "JOB_RESULT.json",
    }
    hydra_files = {
        ".hydra/config.yaml", ".hydra/hydra.yaml", ".hydra/overrides.yaml",
    }
    endpoint_files = {f"{prefix}_EVAL.json", f"{prefix}_SUMMARY.json"}
    forbidden = set()
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        parts, relative_name = relative.parts, relative.as_posix()
        allowed = (
            len(parts) == 1 and parts[0] in root_files
        ) or relative_name in hydra_files or (
            len(parts) == 3
            and checkpoint_pattern.fullmatch(parts[0]) is not None
            and parts[1] == "evals"
            and parts[2] in endpoint_files
        )
        if not allowed:
            forbidden.add(relative_name)
    for path in output.rglob("*"):
        if not path.is_dir() or not any(path.iterdir()):
            continue
        relative = path.relative_to(output)
        parts = relative.parts
        known = relative.as_posix() in {".hydra", "logs"} or (
            len(parts) == 1 and checkpoint_pattern.fullmatch(parts[0]) is not None
        ) or (
            len(parts) == 2
            and checkpoint_pattern.fullmatch(parts[0]) is not None
            and parts[1] == "evals"
        )
        if not known:
            forbidden.add(relative.as_posix() + "/")
    forbidden_artifacts = sorted(forbidden)
    if forbidden_artifacts:
        issues.append("forbidden persistence artifacts are present")

    # Phase 7: ensure the durable command sidecar was not altered by the child.
    try:
        observed_command_record = json.loads(
            command_path.read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        issues.append(f"command identity record is invalid: {error}")
        observed_command_record = None
    if observed_command_record != expected_command_record:
        issues.append("command identity does not match the launched command and environment")

    if exit_status == "invalid_scientific":
        status = "invalid_scientific"
    elif exit_status == "failed_infrastructure":
        status = "failed_infrastructure"
    elif issues:
        status = "invalid_scientific"
        if failure_kind is None:
            failure_kind = "evidence_validation"
    elif infrastructure_issues:
        status = "failed_infrastructure"
        if failure_kind is None:
            failure_kind = "resource_monitor"
    else:
        status = "completed"

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "job_id": job["job_id"],
        "method": job["method"],
        "benchmark": job["benchmark"],
        "seed": job["seed"],
        "stage": job["stage"],
        "status": status,
        "failure_kind": failure_kind,
        "returncode": returncode,
        "command_path": command_path.relative_to(output).as_posix(),
        "command_identity": _command_identity(expected_command_record),
        "environment_overrides": expected_command_record["environment_overrides"],
        "provenance": deepcopy(job["provenance"]),
        "run_log_path": run_log_path.relative_to(output).as_posix(),
        "diagnostics_path": diagnostics_path.relative_to(output).as_posix(),
        "endpoint_summary_path": endpoint_summary_path,
        "endpoint_raw_path": endpoint_raw_path,
        "wall_clock_seconds": wall_clock_seconds,
        "peak_nvml_mib": peak_nvml_mib,
        "peak_gpu_memory_mib": peak_nvml_mib,
        "gpu_memory_sample_count": gpu_memory_sample_count,
        "optimizer_update_count": optimizer_update_count,
        "final_global_step": final_global_step,
        "projection_count": projection_count,
        "zero_step_count": zero_step_count,
        "applied_scale_distribution": applied_scale_distribution,
        "correction_ratio": correction_ratio,
        "max_violation_after": max_violation_after,
        "selected_parameter_changed": selected_parameter_changed,
        "forbidden_artifacts": forbidden_artifacts,
        "issues": issues + infrastructure_issues,
    }
    _atomic_write_json(result_path, result)
    return result


def _read_json(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant: {value}")
        ),
    )


def validate_manifest(manifest):
    """Validate the complete deterministic Stage-1 registry before use."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    expected = build_manifest(seed=0)
    for field in ("schema_version", "protocol", "stage", "seed", "not_applicable"):
        if manifest.get(field) != expected[field]:
            raise ValueError(f"manifest mismatch: {field}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(expected["jobs"]):
        raise ValueError("manifest must contain every registered Stage-1 job")
    if [job.get("job_id") for job in jobs if isinstance(job, dict)] != [
        job["job_id"] for job in expected["jobs"]
    ]:
        raise ValueError("manifest Stage-1 job order or membership mismatch")
    for job in jobs:
        validate_job_identity(job)
    output_root = manifest.get("output_root")
    if output_root is not None and (
        not isinstance(output_root, str) or not Path(output_root).is_absolute()
    ):
        raise ValueError("manifest output_root must be an absolute path")
    return manifest


def create_queue_state(manifest, state_path):
    """Create the persistent direct queue state with every job pending."""
    validate_manifest(manifest)
    destination = Path(state_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_root = Path(manifest.get("output_root", destination.parent)).resolve()
    state = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "output_root": str(output_root),
        "jobs": [],
    }
    for registered in manifest["jobs"]:
        job = deepcopy(registered)
        job.update(
            {
                "status": "pending",
                "attempt_count": 0,
                "attempt_history": [],
                "pid": None,
            }
        )
        state["jobs"].append(job)
    _atomic_write_json(destination, state)
    return state


def validate_job_result(result, job, output_dir):
    """Require durable result evidence for this exact job and command."""
    if not isinstance(result, dict) or result.get("status") not in TERMINAL_STATUSES:
        raise ValueError("JOB_RESULT has no terminal status")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "job_id": job["job_id"],
        "method": job["method"],
        "benchmark": job["benchmark"],
        "seed": job["seed"],
        "stage": job["stage"],
        "command_identity": _command_identity(_command_record(job, output_dir)),
        "environment_overrides": environment_overrides(job),
        "provenance": job["provenance"],
    }
    mismatches = [field for field, value in expected.items() if result.get(field) != value]
    if mismatches:
        raise ValueError("JOB_RESULT identity mismatch: " + ",".join(mismatches))
    return result


def live_matrix_children(state, output_root):
    """Return live registered matrix children and whether each command is exact."""
    root = Path(output_root)
    by_id = {job["job_id"]: job for job in state["jobs"]}
    children = []
    for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            raw = cmdline_path.read_bytes()
            argv = [part.decode() for part in raw.rstrip(b"\0").split(b"\0")]
        except (OSError, UnicodeError):
            continue
        if len(argv) < 3 or argv[1:3] != [
            str(CODE_ROOT / "src/train.py"),
            "--config-name=unlearn.yaml",
        ]:
            continue
        task_arguments = [value for value in argv if value.startswith("task_name=")]
        if "+trainer.method_args.gu.enabled=true" not in argv or len(task_arguments) != 1:
            continue
        job_id = task_arguments[0].split("=", 1)[1]
        job = by_id.get(job_id)
        children.append(
            {
                "pid": int(cmdline_path.parent.name),
                "job_id": job_id,
                "exact": job is not None
                and argv == build_command(job, root / job["output_dir"]),
            }
        )
    return children


def admit_launch(state, output_root):
    """Require an idle matrix and less than 500 MiB already used on GPU 0."""
    children = live_matrix_children(state, output_root)
    if children:
        raise RuntimeError(f"matrix child already running: {children}")
    try:
        sample = subprocess.run(
            GPU_ADMISSION_COMMAND,
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except TimeoutExpired as error:
        raise RuntimeError(
            f"GPU admission timed out after {error.timeout} seconds"
        ) from error
    except OSError as error:
        raise RuntimeError(f"GPU admission launch failed: {error}") from error
    if sample.returncode != 0:
        raise RuntimeError(
            f"GPU admission failed with exit {sample.returncode}: "
            f"{(sample.stderr or '').strip()}"
        )
    device_zero = []
    for line in sample.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2 or fields[0] != "0":
            continue
        try:
            device_zero.append(int(re.sub(r"\s*MiB\s*$", "", fields[1])))
        except ValueError:
            continue
    if len(device_zero) != 1:
        raise RuntimeError("GPU admission did not return exactly one device 0 sample")
    if device_zero[0] >= 500:
        raise RuntimeError(
            f"GPU 0 has {device_zero[0]} MiB used; launch requires below 500 MiB"
        )
    return device_zero[0]


def preflight_manifest(manifest_path):
    """Validate every registered identity and pinned source before launch."""
    manifest = validate_manifest(_read_json(manifest_path))
    fingerprints = {}
    memo = {}
    for job in manifest["jobs"]:
        fingerprints[job["job_id"]] = validate_sources(job, memo)
    return fingerprints


def smoke_manifest(manifest_path):
    """Run one job per method plus every benchmark-specific RMU job."""
    manifest_path = Path(manifest_path)
    preflight_manifest(manifest_path)
    manifest = validate_manifest(_read_json(manifest_path))
    output_root = Path(manifest.get("output_root", manifest_path.parent)).resolve()
    selected = []
    seen_methods = set()
    for job in manifest["jobs"]:
        if job["method"] not in seen_methods or job["method"] == "RMU":
            selected.append(job)
            seen_methods.add(job["method"])
    smoke_state = {
        "jobs": [
            {**deepcopy(job), "status": "pending", "attempt_count": 0}
            for job in selected
        ]
    }
    results = []
    for job in selected:
        admit_launch(smoke_state, output_root)
        results.append(run_job(job, output_root / "smoke" / job["job_id"]))
    return results


def run_queue(manifest_path, state_path):
    """Resume and drain the persistent local queue sequentially on GPU 0."""
    manifest_path, state_path = Path(manifest_path), Path(state_path)
    manifest = validate_manifest(_read_json(manifest_path))
    if not state_path.exists():
        state = create_queue_state(manifest, state_path)
    else:
        state = _read_json(state_path)
    if state.get("schema_version") != SCHEMA_VERSION or state.get("protocol") != PROTOCOL:
        raise ValueError("queue state schema or protocol mismatch")
    output_root = Path(state.get("output_root", "")).resolve()
    expected_root = Path(manifest.get("output_root", manifest_path.parent)).resolve()
    if output_root != expected_root:
        raise ValueError("queue state output_root does not match manifest")
    jobs = state.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("queue state jobs must be a list")
    parents = {job["job_id"]: job for job in jobs if job.get("seed") == 0}
    if set(parents) != {job["job_id"] for job in manifest["jobs"]}:
        raise ValueError("queue state is missing registered Stage-1 jobs")
    for job in jobs:
        parent_id = job.get("parent_seed_zero")
        validate_job_identity(job, parent_job=parents.get(parent_id) if parent_id else None)
        if job.get("status") not in TERMINAL_STATUSES | {"pending", "running"}:
            raise ValueError(f"invalid queue status for {job['job_id']}")
        if not isinstance(job.get("attempt_count"), int) or not isinstance(
            job.get("attempt_history"), list
        ):
            raise ValueError(f"invalid attempt audit for {job['job_id']}")
    running = [job for job in jobs if job["status"] == "running"]
    if len(running) > 1:
        raise ValueError("queue state permits at most one running job")

    if running:
        job = running[0]
        children = live_matrix_children(state, output_root)
        if any(
            child["pid"] == job.get("pid")
            and child["job_id"] == job["job_id"]
            and child.get("exact", True)
            for child in children
        ):
            raise RuntimeError(
                f"exact matrix child already running for {job['job_id']}"
            )
        output_dir = output_root / job["output_dir"]
        result_path = output_dir / "JOB_RESULT.json"
        result = None
        if result_path.is_file():
            try:
                result = validate_job_result(_read_json(result_path), job, output_dir)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                result = None
        history = job["attempt_history"]
        if not history:
            history.append({"attempt": job["attempt_count"], "status": "running"})
        if result is not None:
            history[-1].update(status=result["status"], evidence=result)
            job.update(status=result["status"], pid=None)
            _atomic_write_json(state_path, state)
        else:
            stale = {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "job_id": job["job_id"],
                "status": "failed_infrastructure",
                "failure_kind": "stale_running_without_valid_result",
                "pid": job.get("pid"),
            }
            history[-1].update(status="failed_infrastructure", evidence=stale)
            job["pid"] = None
            if job["attempt_count"] >= 2:
                job["status"] = "failed_infrastructure"
            _atomic_write_json(state_path, state)

    for job in jobs:
        if job["status"] in TERMINAL_STATUSES:
            continue
        parent = parents.get(job.get("parent_seed_zero"))
        output_dir = output_root / job["output_dir"]
        while job["status"] in {"pending", "running"}:
            admit_launch(state, output_root)
            job["status"] = "running"
            job["attempt_count"] += 1
            command_record = _command_record(job, output_dir)
            attempt = {
                "attempt": job["attempt_count"],
                "status": "running",
                "pid": None,
                "argv": command_record["argv"],
                "command_identity": _command_identity(command_record),
            }
            job["attempt_history"].append(attempt)
            _atomic_write_json(state_path, state)

            def record_pid(pid):
                job["pid"] = pid
                attempt["pid"] = pid
                _atomic_write_json(state_path, state)

            try:
                result = run_job(
                    job,
                    output_dir,
                    parent_job=parent,
                    launched=record_pid,
                )
                validate_job_result(result, job, output_dir)
            except Exception as error:
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "job_id": job["job_id"],
                    "status": "failed_infrastructure",
                    "failure_kind": "queue_orchestration",
                    "error": str(error),
                }
            attempt.update(status=result["status"], evidence=result)
            job["pid"] = None
            if (
                result["status"] == "failed_infrastructure"
                and job["attempt_count"] < 2
            ):
                _atomic_write_json(state_path, state)
                continue
            job["status"] = result["status"]
            _atomic_write_json(state_path, state)
            break
    return state


def expand_seeds(state_path):
    """Add seeds 1 and 2 for every valid completed seed-zero parent."""
    state_path = Path(state_path)
    state = _read_json(state_path)
    jobs = state.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("queue state jobs must be a list")
    stage_one = [job for job in jobs if job.get("seed") == 0]
    if any(job.get("status") not in TERMINAL_STATUSES for job in stage_one):
        raise ValueError("Stage 1 must be terminal before seed expansion")
    output_root = Path(state["output_root"])
    by_id = {job["job_id"]: job for job in jobs}
    additions = []
    for parent in stage_one:
        validate_job_identity(parent)
        if parent["status"] != "completed":
            continue
        result_path = output_root / parent["output_dir"] / "JOB_RESULT.json"
        try:
            validate_job_result(_read_json(result_path), parent, result_path.parent)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            continue
        for seed in (1, 2):
            job_id = parent["job_id"].replace("seed0", f"seed{seed}")
            if job_id in by_id:
                validate_job_identity(by_id[job_id], parent_job=parent)
                continue
            derived = deepcopy(parent)
            derived.update(
                {
                    "job_id": job_id,
                    "seed": seed,
                    "stage": "stage2",
                    "status": "pending",
                    "output_dir": parent["output_dir"].replace(
                        "seed0", f"seed{seed}"
                    ),
                    "parent_seed_zero": parent["job_id"],
                    "attempt_count": 0,
                    "attempt_history": [],
                    "pid": None,
                }
            )
            validate_job_identity(derived, parent_job=parent)
            additions.append(derived)
            by_id[job_id] = derived
    if additions:
        state["jobs"].extend(additions)
        state["jobs"].sort(key=lambda job: (job["seed"], job["job_id"]))
        _atomic_write_json(state_path, state)
    return state


def queue_status(state_path):
    """Return durable status counts plus any live registered child."""
    state = _read_json(state_path)
    counts = Counter(job["status"] for job in state["jobs"])
    return {
        "counts": dict(sorted(counts.items())),
        "live_matrix_children": live_matrix_children(state, state["output_root"]),
    }


def _manifest_dry_run(seed):
    manifest = build_manifest(seed=seed)
    jobs = []
    fingerprint_memo = {}
    for job in manifest["jobs"]:
        rendered = deepcopy(job)
        rendered["argv"] = build_command(job, job["output_dir"])
        rendered["environment"] = environment_overrides(job)
        rendered["source_requirements"] = source_requirements(job)
        try:
            rendered["fingerprints"] = validate_sources(job, fingerprint_memo)
        except ValueError as error:
            rendered["fingerprints"] = {}
            rendered["source_validation_error"] = str(error)
            jobs.append(rendered)
            manifest["jobs"] = jobs
            manifest["source_validation_error"] = str(error)
            return manifest
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
    manifest_parser.add_argument("--output-root", type=Path)
    for name in ("preflight", "smoke"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--manifest", type=Path, required=True)
    queue_parser = subparsers.add_parser("queue")
    queue_parser.add_argument("--manifest", type=Path, required=True)
    queue_parser.add_argument("--state", type=Path, required=True)
    expand_parser = subparsers.add_parser("expand-seeds")
    expand_parser.add_argument("--state", type=Path, required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--state", type=Path, required=True)
    run_parser = subparsers.add_parser("run-job")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "manifest":
            if args.dry_run:
                payload = _manifest_dry_run(args.seed)
                print(json.dumps(payload, indent=2, sort_keys=True))
                return int("source_validation_error" in payload)
            payload = build_manifest(args.seed)
            if args.output_root is None:
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 0
            output_root = args.output_root.resolve()
            payload["output_root"] = str(output_root)
            validate_manifest(payload)
            output_root.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(output_root / "manifest.json", payload)
            print(output_root / "manifest.json")
            return 0
        if args.command == "preflight":
            print(json.dumps(preflight_manifest(args.manifest), indent=2, sort_keys=True))
            return 0
        if args.command == "smoke":
            results = smoke_manifest(args.manifest)
            print(json.dumps(results, indent=2, sort_keys=True))
            return int(any(result["status"] != "completed" for result in results))
        if args.command == "queue":
            state = run_queue(args.manifest, args.state)
            print(json.dumps(dict(Counter(job["status"] for job in state["jobs"])), sort_keys=True))
            return 0
        if args.command == "expand-seeds":
            state = expand_seeds(args.state)
            print(json.dumps(dict(Counter(job["status"] for job in state["jobs"])), sort_keys=True))
            return 0
        if args.command == "status":
            print(json.dumps(queue_status(args.state), indent=2, sort_keys=True))
            return 0
        if args.command == "run-job":
            manifest = validate_manifest(_read_json(args.manifest))
            registered = {job["job_id"]: job for job in manifest["jobs"]}
            if args.job_id not in registered:
                raise ValueError(f"unregistered job id: {args.job_id}")
            job = registered[args.job_id]
            output_root = Path(
                manifest.get("output_root", args.manifest.parent)
            ).resolve()
            admission_state = {"jobs": deepcopy(manifest["jobs"])}
            admit_launch(admission_state, output_root)
            result = run_job(job, output_root / job["output_dir"])
            print(json.dumps(result, indent=2, sort_keys=True))
            return int(result["status"] != "completed")
        raise AssertionError(f"unsupported command: {args.command}")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        print(f"{args.command}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
