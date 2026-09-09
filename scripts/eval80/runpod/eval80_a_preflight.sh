#!/usr/bin/env bash
set -euo pipefail
set -x

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/conda-envs/venus-phase1
cd /workspace/project

pwd
git rev-parse HEAD
git status --short --untracked-files=no
df -h /workspace
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

python - <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
import transformers

from src.eval80.core import (
    sha256_file,
    sha256_text_file,
    validate_freeze_record,
    validate_inference_manifest,
)

project = Path("/workspace/project")
model = Path("/workspace/models/Venus-Q-Stage1")
manifest_path = Path("/workspace/data/emoset_eval80/freeze/inference_manifest.json")
freeze_path = Path("/workspace/data/emoset_eval80/freeze/freeze_record.json")
images = Path("/workspace/data/emoset_eval80/images_blind")
config_path = project / "configs/phase1/eval80.json"
handbook_path = project / "Docs/evaluation/EVAL80_SCORING_HANDBOOK.md"
protocol_path = project / "Docs/EMOTION_AWARE_EXPERIMENT_PROTOCOL.md"
model_source_path = model / "VENUS_MODEL_SOURCE.json"
output_path = Path("/workspace/results/eval80/A/attempt-001")

print("PRECHECK_RUNTIME")
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={str(torch.cuda.is_available()).lower()}")
print(f"bf16_supported={str(torch.cuda.is_bf16_supported()).lower()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")

required = {
    "model": model,
    "model_source": model_source_path,
    "manifest": manifest_path,
    "freeze_record": freeze_path,
    "images": images,
    "config": config_path,
    "handbook": handbook_path,
    "protocol": protocol_path,
}
for name, path in required.items():
    print(f"{name}_exists={str(path.exists()).lower()} path={path}")
    if not path.exists():
        raise SystemExit(f"PRECHECK_FAIL: missing {name}: {path}")

config = json.loads(config_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
model_source = json.loads(model_source_path.read_text(encoding="utf-8"))

manifest_errors = validate_inference_manifest(manifest, images, verify_hashes=True)
freeze_errors = validate_freeze_record(
    freeze,
    config_path=config_path,
    handbook_path=handbook_path,
    inference_manifest_path=manifest_path,
)
print(f"manifest_validation_errors={len(manifest_errors)}")
print(f"freeze_validation_errors={len(freeze_errors)}")
if manifest_errors or freeze_errors:
    for error in manifest_errors + freeze_errors:
        print(f"PRECHECK_ERROR={error}")
    raise SystemExit("PRECHECK_FAIL: frozen package validation failed")

print("PRECHECK_HASHES")
print(f"config_sha256={sha256_text_file(config_path)}")
print(f"handbook_sha256={sha256_text_file(handbook_path)}")
print(f"protocol_sha256={sha256_text_file(protocol_path)}")
print(f"manifest_sha256={sha256_file(manifest_path)}")
print(f"freeze_record_sha256={sha256_file(freeze_path)}")

print("PRECHECK_MODEL")
for key in ("repo_id", "revision", "weight_format"):
    actual = model_source.get(key)
    expected = config["model"][key]
    print(f"model_{key}={actual}")
    if actual != expected:
        raise SystemExit(f"PRECHECK_FAIL: model {key} mismatch")
print(f"model_provenance_status={model_source.get('provenance_status', 'original')}")
print(f"model_source_recovered_at_utc={model_source.get('recovered_at_utc', 'not_applicable')}")

index_path = model / "model.safetensors.index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
shards = sorted(set(index["weight_map"].values()))
missing_shards = [name for name in shards if not (model / name).is_file()]
empty_shards = [name for name in shards if (model / name).is_file() and (model / name).stat().st_size == 0]
print(f"model_weight_shard_count={len(shards)}")
print(f"model_missing_weight_shards={len(missing_shards)}")
print(f"model_empty_weight_shards={len(empty_shards)}")
print(f"model_total_weight_bytes={sum((model / name).stat().st_size for name in shards if (model / name).is_file())}")
if missing_shards or empty_shards:
    raise SystemExit("PRECHECK_FAIL: model weight shard missing or empty")

print("PRECHECK_CONDITION_A")
condition = config["conditions"]["A"]
print(f"condition_name={condition['name']}")
print(f"prompt_version={condition['prompt_version']}")
print(f"prompt={condition['prompt']}")
print(f"seed={config['seed']}")
print(f"precision={config['precision']}")
print(f"batch_size={config['batch_size']}")
print(f"manifest_size={len(manifest['records'])}")
print(f"output_path_exists={str(output_path.exists()).lower()}")
if output_path.exists():
    raise SystemExit("PRECHECK_FAIL: attempt-001 output path already exists")

for key in (
    "chat_format",
    "do_sample",
    "eos_token_id",
    "max_new_tokens",
    "max_window_size",
    "pad_token_id",
    "top_k",
    "top_p",
    "transformers_version",
):
    print(f"generation_{key}={config['generation'][key]}")

print("PRECHECK_PASS=true")
PY

set +x
echo "PRECHECK_SCRIPT_COMPLETE=true"
