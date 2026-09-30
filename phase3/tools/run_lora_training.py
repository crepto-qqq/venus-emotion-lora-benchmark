#!/usr/bin/env python3
"""Validated launcher for pilot or formal LoRA training. Dry-run unless --execute is explicit."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from member3_common import ROOT, build_finetune_command, canonical_sha256, command_text, read_json, write_json
from common import utc_now

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["pilot","formal"],required=True); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--attempt-id",required=True); ap.add_argument("--candidate-id"); ap.add_argument("--inventory",type=Path,required=True); ap.add_argument("--execute",action="store_true"); ap.add_argument("--attempt-record",type=Path,required=True); a=ap.parse_args()
    c=read_json(a.config); interface=read_json(ROOT/"phase3/configs/qwen-training-interface-v1.json")
    validator=[sys.executable,str(ROOT/"phase3/tools/validate_lora_config.py"),"--mode",a.mode,"--config",str(a.config),"--inventory",str(a.inventory)]
    validated=subprocess.run(validator,cwd=ROOT)
    if validated.returncode != 0: raise SystemExit("training launch blocked by validator")
    run_dir=c["output"]["run_dir"].replace("{attempt_id}",a.attempt_id).replace("{candidate_id}",a.candidate_id or c.get("candidate_id","formal")); c=json.loads(json.dumps(c)); c["output"]["run_dir"]=run_dir
    cmd=build_finetune_command(c,interface)
    rec={"schema_version":1,"kind":"lora_training_attempt","mode":a.mode,"attempt_id":a.attempt_id,"candidate_id":a.candidate_id,"created_at_utc":utc_now(),"config_sha256":canonical_sha256(c,omit_key="config_sha256"),"command":cmd,"command_text":command_text(cmd),"working_directory":c["runtime"]["working_directory"],"resource_limits":c.get("resource_limits"),"execution_state":"planned"}
    rec["attempt_record_sha256"]=canonical_sha256(rec,omit_key="attempt_record_sha256"); write_json(a.attempt_record,rec)
    if not a.execute: print(rec["command_text"]); return
    os.makedirs(run_dir,exist_ok=True); rec["execution_state"]="started"; rec["started_at_utc"]=utc_now(); write_json(a.attempt_record,rec)
    p=subprocess.run(cmd,cwd=c["runtime"]["working_directory"])
    rec["ended_at_utc"]=utc_now(); rec["returncode"]=p.returncode; rec["execution_state"]="completed" if p.returncode==0 else "failed"; rec["attempt_record_sha256"]=canonical_sha256(rec,omit_key="attempt_record_sha256"); write_json(a.attempt_record,rec); raise SystemExit(p.returncode)
if __name__=="__main__": main()
