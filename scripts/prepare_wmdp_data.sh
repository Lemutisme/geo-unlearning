#!/usr/bin/env bash

set -euo pipefail

family=${1:-all}
if [[ "${family}" != cyber && "${family}" != bio && "${family}" != all ]]; then
    echo "Usage: $0 [cyber|bio|all]" >&2
    exit 2
fi

dataset_id=cais/wmdp-corpora
revision=daf89fa9b618b63a624228061a9cebacca88009c
shared_root=${ORTHOGRAD_SHARED_ROOT:-$(dirname "$(git rev-parse --git-common-dir)")}
output_root=${WMDP_CORPUS_ROOT:-${shared_root}/data/wmdp/wmdp-corpora}
provenance_path="${output_root}/provenance.json"
mkdir -p "${output_root}"

download_public_config() {
    local config_name=$1
    local output_path=$2
    if [[ -s "${output_path}" ]]; then
        return
    fi
    python - "${dataset_id}" "${revision}" "${config_name}" "${output_path}" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

from datasets import load_dataset

dataset_id, revision, config_name, output_name = sys.argv[1:]
output = Path(output_name)
dataset = load_dataset(
    dataset_id,
    config_name,
    split="train",
    revision=revision,
)
output.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
)
os.close(descriptor)
try:
    with open(temporary_name, "w", encoding="utf-8") as handle:
        for row in dataset:
            text = row.get("text")
            if not isinstance(text, str) or not text:
                raise RuntimeError(f"Invalid text row in {dataset_id}/{config_name}")
            json.dump({"text": text}, handle, ensure_ascii=False)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, output)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
PY
}

install_authorized_bio_forget() {
    local output_path="${output_root}/bio-forget-corpus.jsonl"
    if [[ -s "${output_path}" ]]; then
        return
    fi
    if [[ -z "${WMDP_BIO_FORGET_SOURCE:-}" || -z "${WMDP_BIO_FORGET_URL:-}" ]]; then
        echo "The official bio forget corpus is access-controlled. Set both WMDP_BIO_FORGET_SOURCE and WMDP_BIO_FORGET_URL after obtaining authorized access." >&2
        return 3
    fi
    if [[ ! -f "${WMDP_BIO_FORGET_SOURCE}" ]]; then
        echo "WMDP_BIO_FORGET_SOURCE is not a file: ${WMDP_BIO_FORGET_SOURCE}" >&2
        return 3
    fi
    python - "${WMDP_BIO_FORGET_SOURCE}" "${output_path}" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
with source.open(encoding="utf-8") as handle:
    rows = 0
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value.get("text"), str) or not value["text"]:
            raise RuntimeError(f"Invalid text at {source}:{line_number}")
        rows += 1
if rows == 0:
    raise RuntimeError(f"Empty bio forget corpus: {source}")
output.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
)
os.close(descriptor)
try:
    shutil.copyfile(source, temporary_name)
    with open(temporary_name, "rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary_name, output)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
PY
}

if [[ "${family}" == cyber || "${family}" == all ]]; then
    download_public_config cyber-forget-corpus "${output_root}/cyber-forget-corpus.jsonl"
    download_public_config cyber-retain-corpus "${output_root}/cyber-retain-corpus.jsonl"
fi

if [[ "${family}" == bio || "${family}" == all ]]; then
    install_authorized_bio_forget
    download_public_config bio-retain-corpus "${output_root}/bio-retain-corpus.jsonl"
fi

python - "${output_root}" "${provenance_path}" "${dataset_id}" "${revision}" "${WMDP_BIO_FORGET_URL:-}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1])
output = Path(sys.argv[2])
dataset_id = sys.argv[3]
revision = sys.argv[4]
bio_url = sys.argv[5]
previous = {}
if output.is_file():
    previous = json.loads(output.read_text())
files = {}
for path in sorted(root.glob("*-corpus.jsonl")):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    name = path.name.removesuffix("-corpus.jsonl")
    if name == "bio-forget":
        source_url = bio_url or previous.get("files", {}).get(name, {}).get("url")
        if not source_url:
            raise RuntimeError("Missing WMDP_BIO_FORGET_URL provenance")
    else:
        config = f"{name}-corpus"
        source_url = f"hf://datasets/{dataset_id}@{revision}/{config}"
    files[name] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "url": source_url,
    }
record = {
    "dataset": dataset_id,
    "revision": revision,
    "files": files,
}
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, output)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
PY

echo "Prepared ${family} WMDP corpora with SHA-256 provenance: ${provenance_path}"
