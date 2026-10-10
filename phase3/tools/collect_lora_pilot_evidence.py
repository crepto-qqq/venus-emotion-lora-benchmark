#!/usr/bin/env python3
"""Collect Git-safe pilot evidence from explicit runtime observations."""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from member3_common import canonical_sha256, read_json, sha256_file, write_json
from common import utc_now

def evidence_failures(config, attempt, evaluation, facts):
    failures=[]
    for key in ("finite_loss", "checkpoint_saved", "adapter_saved", "reload_passed", "clean_process_inference_passed"):
        if facts.get(key) is not True: failures.append(f"{key} must be true")
    for key in ("zero_supervised_tokens_observed", "unexpected_trainable_parameters"):
        if facts.get(key) is not False: failures.append(f"{key} must be false")
    if attempt.get("execution_state") != "completed" or attempt.get("returncode") != 0: failures.append("attempt did not complete successfully")
    if attempt.get("config_sha256") != canonical_sha256(config, omit_key="config_sha256"): failures.append("attempt/config mismatch")
    for record, field in ((attempt, "attempt_record_sha256"), (evaluation, "report_sha256")):
        if record.get(field) != canonical_sha256(record, omit_key=field): failures.append(f"invalid {field}")
    if evaluation.get("result") != "passed" or evaluation.get("missing_ids") != []: failures.append("evaluation incomplete")
    if not re.fullmatch(r"[0-9a-f]{64}", str(facts.get("adapter_sha256", ""))): failures.append("invalid adapter hash")
    for key in ("peak_vram_bytes", "throughput_samples_per_second", "elapsed_seconds", "displayed_hourly_price_usd", "calculated_cost_usd"):
        value=facts.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0: failures.append(f"invalid {key}")
    limits=config.get("resource_limits") or {}
    for key, limit in (("elapsed_seconds", limits.get("max_wall_clock_minutes",0)*60), ("calculated_cost_usd", limits.get("total_budget_usd",0)), ("displayed_hourly_price_usd", limits.get("max_hourly_usd",0))):
        if isinstance(facts.get(key),(int,float)) and facts[key] > limit: failures.append(f"{key} exceeds limit")
    return failures


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--attempt-record",type=Path,required=True); ap.add_argument("--evaluation",type=Path,required=True); ap.add_argument("--runtime-facts",type=Path,required=True,help="operator-produced JSON with cost/VRAM/throughput/reload/gradient facts"); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    c=read_json(a.config); attempt=read_json(a.attempt_record); ev=read_json(a.evaluation); facts=read_json(a.runtime_facts)
    required=["finite_loss","zero_supervised_tokens_observed","unexpected_trainable_parameters","peak_vram_bytes","throughput_samples_per_second","elapsed_seconds","displayed_hourly_price_usd","calculated_cost_usd","checkpoint_saved","adapter_saved","reload_passed","clean_process_inference_passed","adapter_sha256"]
    missing=[k for k in required if k not in facts]
    failures=evidence_failures(c,attempt,ev,facts)
    evidence={"schema_version":1,"kind":"lora_pilot_evidence","generated_at_utc":utc_now(),"candidate_id":c.get("candidate_id"),"config_sha256":canonical_sha256(c,omit_key="config_sha256"),"attempt_record_sha256":sha256_file(a.attempt_record),"evaluation_sha256":sha256_file(a.evaluation),"evaluation":ev,"runtime_facts":facts,"missing_runtime_facts":missing,"failures":failures,"result":"passed" if not missing and not failures else "failed"}
    evidence["report_sha256"]=canonical_sha256(evidence,omit_key="report_sha256"); write_json(a.output,evidence); print(json.dumps(evidence,indent=2)); raise SystemExit(0 if evidence["result"]=="passed" else 2)
if __name__=="__main__": main()
