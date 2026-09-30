#!/usr/bin/env python3
"""Validate Member 3B post-training acceptance before handoff to Member 4."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from member3_common import canonical_sha256, read_json, write_json
HEX64=re.compile(r'^[0-9a-f]{64}$')
def validate(r):
    f=[]
    if r.get('schema_version')!=1: f.append('schema_version must be 1')
    if r.get('kind')!='training_acceptance': f.append('kind must be training_acceptance')
    if r.get('reviewer_role')!='Member 3B': f.append('reviewer_role must be Member 3B')
    if r.get('decision') not in {'accepted','rejected'}: f.append('decision must be accepted or rejected')
    if r.get('decision')=='accepted':
        if not HEX64.fullmatch(str(r.get('formal_config_sha256',''))): f.append('formal_config_sha256 invalid')
        if not HEX64.fullmatch(str(r.get('adapter_sha256',''))): f.append('adapter_sha256 invalid')
        if r.get('reload_passed') is not True: f.append('accepted adapter requires reload_passed=true')
        if r.get('clean_process_inference_passed') is not True: f.append('accepted adapter requires clean_process_inference_passed=true')
        if r.get('selection_basis')!='BridgeTrain-v1 validation only': f.append('selection_basis must be BridgeTrain-v1 validation only')
        if not isinstance(r.get('selected_checkpoint'),dict) or not r['selected_checkpoint'].get('id'): f.append('selected_checkpoint identity required')
    return f
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('record',type=Path); ap.add_argument('--report',type=Path); a=ap.parse_args(); r=read_json(a.record); f=validate(r)
    out={'schema_version':1,'kind':'training_acceptance_validation','record_sha256':canonical_sha256(r),'result':'passed' if not f else 'failed','failures':f}; out['report_sha256']=canonical_sha256(out,omit_key='report_sha256')
    if a.report: write_json(a.report,out)
    print(json.dumps(out,indent=2)); raise SystemExit(0 if not f else 2)
if __name__=='__main__': main()
