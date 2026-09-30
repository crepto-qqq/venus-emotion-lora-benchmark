#!/usr/bin/env python3
"""Collect Git-safe pilot evidence from explicit runtime observations."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from member3_common import canonical_sha256, read_json, sha256_file, write_json
from common import utc_now

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--attempt-record",type=Path,required=True); ap.add_argument("--evaluation",type=Path,required=True); ap.add_argument("--runtime-facts",type=Path,required=True,help="operator-produced JSON with cost/VRAM/throughput/reload/gradient facts"); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    c=read_json(a.config); attempt=read_json(a.attempt_record); ev=read_json(a.evaluation); facts=read_json(a.runtime_facts)
    required=["finite_loss","zero_supervised_tokens_observed","unexpected_trainable_parameters","peak_vram_bytes","throughput_samples_per_second","elapsed_seconds","displayed_hourly_price_usd","calculated_cost_usd","checkpoint_saved","adapter_saved","reload_passed","clean_process_inference_passed","adapter_sha256"]
    missing=[k for k in required if k not in facts]
    evidence={"schema_version":1,"kind":"lora_pilot_evidence","generated_at_utc":utc_now(),"candidate_id":c.get("candidate_id"),"config_sha256":canonical_sha256(c,omit_key="config_sha256"),"attempt_record_sha256":sha256_file(a.attempt_record),"evaluation_sha256":sha256_file(a.evaluation),"evaluation":ev,"runtime_facts":facts,"missing_runtime_facts":missing,"result":"passed" if not missing and facts.get("finite_loss") and facts.get("reload_passed") and facts.get("clean_process_inference_passed") and not facts.get("zero_supervised_tokens_observed") and not facts.get("unexpected_trainable_parameters") else "failed"}
    evidence["report_sha256"]=canonical_sha256(evidence,omit_key="report_sha256"); write_json(a.output,evidence); print(json.dumps(evidence,indent=2)); raise SystemExit(0 if evidence["result"]=="passed" else 2)
if __name__=="__main__": main()
