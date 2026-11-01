#!/usr/bin/env python3
"""Validate Member 3 LoRA configs at static, runtime-inventory, pilot, or formal gates."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from member3_common import ROOT, TOOL_VERSION, build_finetune_command, canonical_sha256, git_commit, read_json, sha256_file, validate_resource_limits, verify_release_hashes, write_json
from common import utc_now

def fail_if(cond,msg,failures):
    if cond: failures.append(msg)

def static_checks(c):
    f=[]
    fail_if(c.get("schema_version")!=1,"schema_version must be 1",f)
    fail_if(c.get("lora",{}).get("mode") not in {"lora"},"only BF16 LoRA is permitted; QLoRA/full training is forbidden",f)
    t=c.get("training",{}); l=c.get("lora",{})
    fail_if(t.get("mixed_precision")!="bf16","mixed_precision must be bf16",f)
    fail_if(t.get("distributed_policy")!="single_gpu","primary Member 3 route is single_gpu",f)
    fail_if(t.get("deepspeed") is not None,"DeepSpeed must be explicitly disabled",f)
    fail_if(t.get("fix_vit") is not True,"fix_vit must be true",f)
    fail_if(l.get("modules_to_save") is not None,"modules_to_save must remain null with unchanged tokenizer",f)
    fail_if(l.get("rank")==2,"rank-2 Member 1 fixture is not a training recommendation",f)
    fail_if(t.get("epochs",99)>3,"primary experiment may use at most 3 epochs",f)
    fail_if(t.get("max_sequence_length",0)<=0,"max_sequence_length must be explicit",f)
    lock=read_json(ROOT/"phase3/configs/source-lock.json")
    r=c.get("runtime",{})
    fail_if(r.get("expected_model_revision")!=lock["model"]["revision"],"pinned model revision mismatch",f)
    fail_if(r.get("expected_upstream_commit")!=lock["upstream"]["commit"],"pinned upstream commit mismatch",f)
    fail_if(r.get("expected_patch_sha256")!=lock["upstream"]["patch_sha256"],"reviewed patch hash mismatch",f)
    f += verify_release_hashes()
    return f

def runtime_checks(c, inventory_path):
    f=[]
    fail_if(not inventory_path or not inventory_path.is_file(),"runtime inventory report is required",f)
    if f: return f
    inv=read_json(inventory_path)
    fail_if(inv.get("result")!="passed","runtime inventory did not pass",f)
    fail_if(c.get("inventory_state")!="inventory_verified","config inventory_state must be inventory_verified",f)
    l=c["lora"]
    fail_if(not isinstance(l.get("target_modules"),list) or not l["target_modules"],"target_modules must be concrete after inventory",f)
    fail_if(l.get("target_source")!="runtime_inventory","target_source must be runtime_inventory",f)
    fail_if(l.get("inventory_sha256")!=inv.get("report_sha256"),"inventory hash mismatch",f)
    fail_if(l.get("matched_module_count")!=inv.get("matched_module_count"),"matched module count mismatch",f)
    fail_if(inv.get("visual_modules_targeted") is not False,"visual modules must not be targeted",f)
    return f

def pilot_checks(c, interface):
    f=[]
    f += validate_resource_limits(c.get("resource_limits"))
    fail_if(interface.get("source_inventory_state")!="verified","pinned Qwen parser inventory is not verified",f)
    if interface.get("source_inventory_state")=="verified":
        parser_names={x.get("name") for x in interface.get("parser_inventory",[]) if isinstance(x,dict)}
        missing=set(interface.get("allowed_argument_names",[]))-parser_names
        fail_if(bool(missing),f"allowlisted arguments absent from parser inventory: {sorted(missing)}",f)
    try: build_finetune_command(c,interface)
    except Exception as e: f.append(f"command/config mapping failed: {e}")
    out=c.get("output",{}).get("run_dir","")
    fail_if(not str(out).startswith("/workspace/phase3/member3/"),"pilot output must be isolated below /workspace/phase3/member3",f)
    return f

def formal_checks(c, interface, review_path):
    f=pilot_checks(c,interface)
    if not review_path or not review_path.is_file(): f.append("Member 3B pilot-review-v1.json is required")
    else:
        r=read_json(review_path)
        fail_if(r.get("decision")!="approved","pilot review decision is not approved",f)
        fail_if(r.get("formal_config_sha256")!=canonical_sha256(c,omit_key="config_sha256"),"pilot review does not bind this formal config",f)
        if r.get('kind') == 'delegated_formal_decision':
            f += delegated_review_checks(c, r)
        else:
            fail_if(r.get('reviewer_role') != 'Member 3B', 'legacy formal review requires Member 3B identity', f)
    return f


def delegated_review_checks(c, record):
    """Explicit human decision under reported delegation, not independent 3B review."""
    f = []
    for key, expected in (('schema_version', 1), ('reviewer_role', 'delegated_user'),
                          ('decision_maker', 'Zhiyuan Zhang'), ('human_confirmed', True),
                          ('independent_member3b_review', False)):
        mismatch = record.get(key) is not expected if isinstance(expected, bool) else record.get(key) != expected
        fail_if(mismatch, 'delegated decision invalid: ' + key, f)
    delegation = record.get('delegation', {})
    fail_if(delegation.get('source') != 'user_reported_Jackie_delegation' or not delegation.get('statement'),
            'delegation source and statement required', f)
    fail_if(record.get('formal_resource_limits') != c.get('resource_limits'),
            'delegated decision resource limits mismatch', f)
    fail_if(record.get('release_hashes') != c['data'].get('release_hashes'),
            'delegated decision release hashes mismatch', f)
    fail_if(record.get('checkpoint_selection') != c.get('planning_metadata', {}).get('checkpoint_selection'),
            'delegated decision checkpoint rule mismatch', f)
    fail_if(not record.get('known_limitations'), 'known quality/evidence limitations required', f)
    evidence = record.get('reviewed_evidence', [])
    fail_if(not evidence, 'reviewed evidence files required', f)
    for item in evidence:
        path = Path(item.get('path', ''))
        if not path.is_absolute(): path = ROOT / path
        fail_if(not path.is_file() or sha256_file(path) != item.get('sha256'),
                'delegated decision evidence file missing or changed', f)
    return f

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["static","runtime-inventory","pilot","formal"],required=True); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--inventory",type=Path); ap.add_argument("--interface",type=Path,default=ROOT/"phase3/configs/qwen-training-interface-v1.json"); ap.add_argument("--pilot-review",type=Path,default=ROOT/"phase3/reviews/member3/pilot-review-v1.json"); ap.add_argument("--report",type=Path); a=ap.parse_args()
    c=read_json(a.config); interface=read_json(a.interface); failures=static_checks(c)
    checks={"static":"passed" if not failures else "failed","runtime_inventory":"not_run","pilot":"not_run","formal":"not_run"}
    if a.mode in {"runtime-inventory","pilot","formal"}:
        more=runtime_checks(c,a.inventory); failures += more; checks["runtime_inventory"]="passed" if not more else "failed"
    if a.mode in {"pilot","formal"}:
        more=pilot_checks(c,interface); failures += more; checks["pilot"]="passed" if not more else "failed"
    if a.mode=="formal":
        more=formal_checks(c,interface,a.pilot_review); failures += [x for x in more if x not in failures]; checks["formal"]="passed" if not more else "failed"
    report={"schema_version":1,"kind":"lora_validation_report","tool_version":TOOL_VERSION,"mode":a.mode,"generated_at_utc":utc_now(),"project_commit":git_commit(),"config_path":str(a.config),"config_sha256":canonical_sha256(c,omit_key="config_sha256"),"checks":checks,"result":"passed" if not failures else "failed","failures":failures}
    report["report_sha256"]=canonical_sha256(report,omit_key="report_sha256")
    if a.report: write_json(a.report,report)
    print(json.dumps(report,indent=2,ensure_ascii=False)); raise SystemExit(0 if not failures else 2)
if __name__=="__main__": main()
