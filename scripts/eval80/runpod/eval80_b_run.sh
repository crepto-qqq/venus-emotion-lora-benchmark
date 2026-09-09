#!/usr/bin/env bash
set -euo pipefail

condition="${1:-}"
if [[ "$condition" != "B0" && "$condition" != "B1" ]]; then
  echo "usage: $0 B0|B1" >&2
  exit 2
fi

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/conda-envs/venus-phase1
cd /workspace/project

output_dir="/workspace/results/eval80/${condition}/attempt-001"
log_path="/workspace/logs/eval80/${condition}_attempt-001_inference.log"

echo "=== FORMAL EVAL80 CONDITION ${condition} START ==="
echo "command=python -m src.eval80.cli run --condition ${condition} --model-path /workspace/models/Venus-Q-Stage1 --manifest /workspace/data/emoset_eval80/freeze/inference_manifest.json --image-dir /workspace/data/emoset_eval80/images_blind --freeze-record /workspace/data/emoset_eval80/freeze/freeze_record.json --handbook /workspace/project/Docs/evaluation/EVAL80_SCORING_HANDBOOK.md --config /workspace/project/configs/phase1/eval80.json --output-dir ${output_dir}"
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
PYTHONUNBUFFERED=1 python -u -m src.eval80.cli run \
  --condition "$condition" \
  --model-path /workspace/models/Venus-Q-Stage1 \
  --manifest /workspace/data/emoset_eval80/freeze/inference_manifest.json \
  --image-dir /workspace/data/emoset_eval80/images_blind \
  --freeze-record /workspace/data/emoset_eval80/freeze/freeze_record.json \
  --handbook /workspace/project/Docs/evaluation/EVAL80_SCORING_HANDBOOK.md \
  --config /workspace/project/configs/phase1/eval80.json \
  --output-dir "$output_dir" \
  2>&1 | tee "$log_path"
run_status=${PIPESTATUS[0]}
set -e

echo "formal_inference_exit_code=$run_status"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$run_status"
