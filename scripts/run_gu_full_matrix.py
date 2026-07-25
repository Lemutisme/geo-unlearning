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
    "HF_HUB_CACHE": "/dev/shm/ungu-hf-hub-wmdp",
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
TOFU_DATASET_CACHE_ROOT = Path("/root/.cache/huggingface/datasets/locuslab___tofu")
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
        "root": Path("/dev/shm/ungu-hf-datasets-muse/muse-bench___muse-news"),
        "revision": "506bd5b150b92814d45e4404a82f120ab2d748bf",
        "manifests": {
            "raw": "e06cff3f2f6e6539bdc93bff1e9c02ce741a6f5a1336a0d377abae557082e1c4",
            "knowmem": "fb4c8e02ab1d1ef2536213721fb6d401f6ec9a6fb6d0aba39daf9085563d4f0f",
            "verbmem": "4bcebf1eea8341165e7b86d141bc5e0d3e27a466bbab7653254b34d265ffd6ea",
            "privleak": "33063ca090ed95c289f31b813071295ff382cf67fcb65588b3876ff737d57525",
        },
    },
    "Books": {
        "root": Path(
            "/root/.cache/huggingface/datasets/muse-bench___muse-books"
        ),
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
    "mmlu": "ccdd490e6042440417b380770294c4cab6d28977d522f8f68a6dd527f03ba40f",
    "wikitext": "3b162501be71321b5c3a05ecab71bb452376295224b5060ce0108cb05a52d5ce",
}
HUB_SNAPSHOTS = {
    "tofu_target_tokenizer": {
        "source_name": "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
        "revision": "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e",
        "path": (
            "/dev/shm/gu-hf/hub/models--open-unlearning--"
            "tofu_Llama-3.1-8B-Instruct_full/snapshots/"
            "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e"
        ),
        "content_manifest_sha256": "f9879ff7688d9233da6ff1c060ad3d465aaaf026d0c9e20372add28dc2c44fd3",
    },
    "muse_news_target": {
        "source_name": "muse-bench/MUSE-News_target",
        "revision": "a2f39769e9a0b98ec1cdd12f65e9962502208935",
        "path": (
            "/dev/shm/ungu-hf-hub-muse/models--muse-bench--MUSE-News_target/"
            "snapshots/a2f39769e9a0b98ec1cdd12f65e9962502208935"
        ),
        "content_manifest_sha256": "7cb73ebda17432c1f48092f243eecdc7ce0165be644d4c506ba3dee86ae6011c",
    },
    "muse_books_target": {
        "source_name": "muse-bench/MUSE-Books_target",
        "revision": "c8dd3fb23a726762ec66d277933c7cff6767f3c2",
        "path": (
            "/root/.cache/huggingface/hub/models--muse-bench--MUSE-Books_target/"
            "snapshots/c8dd3fb23a726762ec66d277933c7cff6767f3c2"
        ),
        "content_manifest_sha256": None,
    },
    "muse_tokenizer": {
        "source_name": "NousResearch/Llama-2-7b-hf",
        "revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
        "path": (
            "/dev/shm/ungu-hf-hub-muse/models--NousResearch--Llama-2-7b-hf/"
            "snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
        ),
        "content_manifest_sha256": "8f8ec553756a42d2fa5e3b630aba8cf49c8a8afa90664527d4d295c07c1000f2",
    },
    "muse_news_reference": {
        "source_name": "muse-bench/MUSE-News_retrain",
        "revision": "324ef49ee0a038078aba7d8de831edf57235c9b3",
        "path": (
            "/dev/shm/ungu-hf-hub-muse/models--muse-bench--MUSE-News_retrain/"
            "snapshots/324ef49ee0a038078aba7d8de831edf57235c9b3"
        ),
        "content_manifest_sha256": "f7638ac028811c84e246514b838eb7cd40e39c41fea3ea518e449252186c5868",
    },
    "muse_books_reference": {
        "source_name": "muse-bench/MUSE-Books_retrain",
        "revision": "1d67430e4e8bdf2a65823740e909792519175ac2",
        "path": (
            "/root/.cache/huggingface/hub/models--muse-bench--MUSE-Books_retrain/"
            "snapshots/1d67430e4e8bdf2a65823740e909792519175ac2"
        ),
        "content_manifest_sha256": "6ff6a81660877991ae44d4118441ad2815c95ff8bc2d54f2256728ba7a1ea434",
    },
    "gibberish_classifier": {
        "source_name": "madhurjindal/autonlp-Gibberish-Detector-492513457",
        "revision": "76672dd7d3575f68ab980705bcec975cc62de71c",
        "path": (
            "/dev/shm/ungu-hf-hub-muse/models--madhurjindal--"
            "autonlp-Gibberish-Detector-492513457/snapshots/"
            "76672dd7d3575f68ab980705bcec975cc62de71c"
        ),
        "content_manifest_sha256": "3d0e600d1f642bd55c16195eb8238fbb86ad17dab0a49b08bc18f4dcdd05b87c",
    },
    "wmdp_model_tokenizer": {
        "source_name": "HuggingFaceH4/zephyr-7b-beta",
        "revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c",
        "path": (
            "/dev/shm/ungu-hf-hub-wmdp/models--HuggingFaceH4--zephyr-7b-beta/"
            "snapshots/892b3d7a7b1cf10c7a701c60881cd93df615734c"
        ),
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
        if family in RMU_MODEL_OVERRIDES:
            command.extend(RMU_MODEL_OVERRIDES[family])
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


def _dataset_cache_requirement(source_name, path, content_manifest_sha256):
    return {
        "kind": "dataset_cache",
        "source_name": source_name,
        "path": str(path),
        "content_manifest_sha256": content_manifest_sha256,
        "required_suffixes": [".arrow", ".json"],
    }


def _hub_snapshot_requirement(key):
    requirement = deepcopy(HUB_SNAPSHOTS[key])
    requirement["kind"] = "hub_snapshot"
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
            _hub_snapshot_requirement("tofu_target_tokenizer"),
            _hub_snapshot_requirement("gibberish_classifier"),
        ]
        requirements.extend(
            _dataset_cache_requirement(
                f"locuslab/TOFU:{name}",
                TOFU_DATASET_CACHE_ROOT / name / "0.0.0" / TOFU_DATASET_REVISION,
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
            _hub_snapshot_requirement(target_key),
            _hub_snapshot_requirement("muse_tokenizer"),
            _hub_snapshot_requirement(reference_key),
            _hub_snapshot_requirement("gibberish_classifier"),
        ]
        requirements.extend(
            _dataset_cache_requirement(
                f"muse-bench/MUSE-{split}:{name}",
                cache["root"] / name / "0.0.0" / cache["revision"],
                manifest,
            )
            for name, manifest in cache["manifests"].items()
        )
        return requirements
    requirements = [
        _hub_snapshot_requirement("wmdp_model_tokenizer"),
        _dataset_cache_requirement(
            "cais/wmdp:wmdp_cyber",
            Path(WMDP_EVALUATION_CACHE)
            / "cais___wmdp/wmdp-cyber/0.0.0"
            / "7125571f22f032c56415e7980f48d877dd830ff8",
            WMDP_DATASET_MANIFESTS["wmdp_cyber"],
        ),
        _dataset_cache_requirement(
            "hails/mmlu_no_train",
            Path(WMDP_EVALUATION_CACHE) / "hails___mmlu_no_train",
            WMDP_DATASET_MANIFESTS["mmlu"],
        ),
    ]
    if job["method"] == "RMU":
        requirements.append(
            _dataset_cache_requirement(
                "wikitext:wikitext-2-raw-v1",
                Path(WMDP_EVALUATION_CACHE)
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


def canonical_directory_fingerprint(path):
    """Fingerprint relative paths, sizes, and file SHA256 values canonically."""
    root = Path(path)
    entries = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
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
        suffixes = set(requirement.get("required_suffixes", ()))
        observed_suffixes = {
            candidate.suffix for candidate in path.rglob("*") if candidate.is_file()
        }
        if not suffixes.issubset(observed_suffixes):
            raise ValueError(f"{source_name}: Arrow or metadata files are missing")
        observed = canonical_directory_fingerprint(path)
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
    args = parser.parse_args(argv)

    if args.command == "manifest":
        payload = _manifest_dry_run(args.seed) if args.dry_run else build_manifest(args.seed)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return int("source_validation_error" in payload)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
