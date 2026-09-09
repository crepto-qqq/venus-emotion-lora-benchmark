#!/usr/bin/env bash
set -euo pipefail

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/conda-envs/venus-phase1
cd /workspace/project

echo "=== FORMAL EVAL80 CONDITION A START ==="
echo "command=python -m src.eval80.cli run --condition A --model-path /workspace/models/Venus-Q-Stage1 --manifest /workspace/data/emoset_eval80/freeze/inference_manifest.json --image-dir /workspace/data/emoset_eval80/images_blind --freeze-record /workspace/data/emoset_eval80/freeze/freeze_record.json --handbook /workspace/project/Docs/evaluation/EVAL80_SCORING_HANDBOOK.md --config /workspace/project/configs/phase1/eval80.json --output-dir /workspace/results/eval80/A/attempt-001"
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
PYTHONUNBUFFERED=1 python -u -m src.eval80.cli run \
  --condition A \
  --model-path /workspace/models/Venus-Q-Stage1 \
  --manifest /workspace/data/emoset_eval80/freeze/inference_manifest.json \
  --image-dir /workspace/data/emoset_eval80/images_blind \
  --freeze-record /workspace/data/emoset_eval80/freeze/freeze_record.json \
  --handbook /workspace/project/Docs/evaluation/EVAL80_SCORING_HANDBOOK.md \
  --config /workspace/project/configs/phase1/eval80.json \
  --output-dir /workspace/results/eval80/A/attempt-001 \
  2>&1 | tee /workspace/logs/eval80/A_attempt-001_inference.log
run_status=${PIPESTATUS[0]}
set -e

echo "formal_inference_exit_code=$run_status"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$run_status"
