#!/usr/bin/env python3
"""Score Member 3 validation predictions without touching Eval20/Eval80."""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from member3_common import ROOT, canonical_sha256, read_json, task_type, write_json
ALLOWED={"amusement","anger","awe","contentment","disgust","excitement","fear","sadness"}

def parse_guidance(text):
    req=["Emotion:","Visual evidence:","Aesthetic relationship:","Guidance:","Final emotion:"]
    ok=all(x in text for x in req)
    m1=re.search(r"(?m)^Emotion:\s*(\w+)\s*$",text); m2=re.search(r"(?m)^Final emotion:\s*(\w+)\s*$",text)
    label=m1.group(1).lower() if m1 else None; final=m2.group(1).lower() if m2 else None
    return ok and label in ALLOWED and label==final, label

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",type=Path,required=True,help="JSONL: id,prediction[,loss]"); ap.add_argument("--validation",type=Path,default=ROOT/"phase3/releases/bridge-train-v1/bridge-train-v1.validation.json"); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    gold={x["id"]:x for x in read_json(a.validation)}; pred={}
    for line in a.predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r=json.loads(line); pred[r["id"]]=r
    cls_total=cls_ok=guide_total=guide_valid=0; losses={"classification":[],"joint_guidance":[]}; missing=[]
    for sid,s in gold.items():
        p=pred.get(sid)
        if not p: missing.append(sid); continue
        t=task_type(sid); text=str(p.get("prediction","")).strip()
        if isinstance(p.get("loss"),(int,float)) and math.isfinite(p["loss"]): losses[t].append(float(p["loss"]))
        if t=="classification": cls_total+=1; cls_ok += int(text.lower()==s["conversations"][1]["value"].strip().lower())
        else:
            guide_total+=1; valid,_=parse_guidance(text); guide_valid += int(valid)
    report={"schema_version":1,"kind":"lora_pilot_evaluation","validation_set":"BridgeTrain-v1 validation only","prediction_count":len(pred),"missing_ids":missing,"classification_exact_match":cls_ok/cls_total if cls_total else None,"guidance_format_valid_rate":guide_valid/guide_total if guide_total else None,"validation_loss":{"classification":sum(losses['classification'])/len(losses['classification']) if losses['classification'] else None,"joint_guidance":sum(losses['joint_guidance'])/len(losses['joint_guidance']) if losses['joint_guidance'] else None},"result":"passed" if not missing else "incomplete"}
    report["report_sha256"]=canonical_sha256(report,omit_key="report_sha256"); write_json(a.output,report); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
