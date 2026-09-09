#!/usr/bin/env bash
set -euo pipefail

clear
echo "=== EVAL80 CONDITION A FINAL ACCEPTANCE ==="
grep -F '[EVIDENCE MILESTONE]' /workspace/logs/eval80/A_attempt-001_inference.log

python - <<'PY'
from pathlib import Path
import hashlib
import json

run_dir = Path("/workspace/results/eval80/A/attempt-001")
run = json.loads((run_dir / "run_config.json").read_text())
environment = json.loads((run_dir / "environment.json").read_text())
summary = json.loads((run_dir / "summary.json").read_text())
validation = json.loads((run_dir / "validation.json").read_text())
records_path = run_dir / "records.jsonl"
records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
records_hash = hashlib.sha256(records_path.read_bytes()).hexdigest()

expected_generation = run["generation_expected"]
actual_generation = run["generation_actual"]
generation_matches = all(actual_generation.get(key) == value for key, value in expected_generation.items())
success_count = sum(record.get("status") == "success" for record in records)
error_count = sum(record.get("status") == "error" for record in records)
empty_count = sum(not str(record.get("response", "")).strip() for record in records)
unique_ids = len({record.get("blind_id") for record in records})
git_head = environment["git_head"].get("stdout")

print(f"git_head={git_head}")
print(f"condition={run['condition']}")
print(f"prompt_version={run['prompt_version']}")
print(f"prompt={run['prompt']}")
print(f"model_revision={run['model_source']['revision']}")
print(f"model_provenance_status={run['model_source'].get('provenance_status', 'original')}")
print(f"parameter_devices={','.join(run['parameter_devices'])}")
print(f"generation_expected_matches_actual={str(generation_matches).lower()}")
print(f"model_load_seconds={run['model_load_seconds']}")
print(f"model_load_peak_gpu_memory_mb={run['model_load_peak_gpu_memory_mb']}")
print(f"record_count={len(records)}")
print(f"success_count={success_count}")
print(f"error_count={error_count}")
print(f"empty_response_count={empty_count}")
print(f"unique_blind_id_count={unique_ids}")
print(f"total_run_seconds={summary['total_run_seconds']}")
print(f"mean_inference_seconds={summary['mean_inference_seconds']}")
print(f"maximum_peak_gpu_memory_mb={summary['maximum_peak_gpu_memory_mb']}")
print(f"records_sha256={records_hash}")
print(f"summary_hash_matches={str(summary['records_sha256'] == records_hash).lower()}")
print(f"validation_hash_matches={str(validation['records_sha256'] == records_hash).lower()}")
print(f"validation_valid={str(validation['valid']).lower()}")
print(f"validation_errors={len(validation['errors'])}")
print(f"failure_file_exists={str((run_dir / 'failure.json').exists()).lower()}")

accepted = all(
    (
        validation["valid"],
        len(records) == 80,
        success_count == 80,
        error_count == 0,
        empty_count == 0,
        unique_ids == 80,
        generation_matches,
        run["parameter_devices"] == ["cuda:0"],
        summary["records_sha256"] == records_hash,
        validation["records_sha256"] == records_hash,
        not (run_dir / "failure.json").exists(),
    )
)
print(f"formal_acceptance={str(accepted).lower()}")
PY

echo "=== CONDITION A COMPLETE; B NOT STARTED ==="
