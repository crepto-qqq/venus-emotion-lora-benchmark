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

run_dir="/workspace/results/eval80/${condition}/attempt-001"
report_path="${run_dir}/validation.json"
log_path="/workspace/logs/eval80/${condition}_attempt-001_validation.log"

echo "=== EVAL80 CONDITION ${condition} FINAL VALIDATION ==="
echo "command=python -m src.eval80.cli validate-output --condition ${condition} --run-dir ${run_dir} --manifest /workspace/data/emoset_eval80/freeze/inference_manifest.json --config /workspace/project/configs/phase1/eval80.json --report ${report_path}"

set +e
PYTHONUNBUFFERED=1 python -u -m src.eval80.cli validate-output \
  --condition "$condition" \
  --run-dir "$run_dir" \
  --manifest /workspace/data/emoset_eval80/freeze/inference_manifest.json \
  --config /workspace/project/configs/phase1/eval80.json \
  --report "$report_path" \
  2>&1 | tee "$log_path"
validation_status=${PIPESTATUS[0]}
set -e

echo "validation_exit_code=$validation_status"
echo "validated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$validation_status"
