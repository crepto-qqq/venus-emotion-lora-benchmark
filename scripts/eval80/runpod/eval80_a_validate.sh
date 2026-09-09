#!/usr/bin/env bash
set -euo pipefail

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/conda-envs/venus-phase1
cd /workspace/project

echo "=== EVAL80 CONDITION A FINAL VALIDATION ==="
echo "command=python -m src.eval80.cli validate-output --condition A --run-dir /workspace/results/eval80/A/attempt-001 --manifest /workspace/data/emoset_eval80/freeze/inference_manifest.json --config /workspace/project/configs/phase1/eval80.json --report /workspace/results/eval80/A/attempt-001/validation.json"

set +e
PYTHONUNBUFFERED=1 python -u -m src.eval80.cli validate-output \
  --condition A \
  --run-dir /workspace/results/eval80/A/attempt-001 \
  --manifest /workspace/data/emoset_eval80/freeze/inference_manifest.json \
  --config /workspace/project/configs/phase1/eval80.json \
  --report /workspace/results/eval80/A/attempt-001/validation.json \
  2>&1 | tee /workspace/logs/eval80/A_attempt-001_validation.log
validation_status=${PIPESTATUS[0]}
set -e

echo "validation_exit_code=$validation_status"
echo "validated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$validation_status"
