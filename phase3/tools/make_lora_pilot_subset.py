#!/usr/bin/env python3
"""Create a deterministic, task/class balanced pilot subset without ad-hoc shell slicing."""
from __future__ import annotations
import argparse, hashlib, random
from collections import defaultdict
from pathlib import Path
from member3_common import ROOT, canonical_sha256, read_json, task_type, write_json

def emotion(sample): return sample["id"].split("_",1)[0]
def choose(records, per_class_task, seed):
    buckets=defaultdict(list)
    for r in records: buckets[(emotion(r),task_type(r["id"]))].append(r)
    rng=random.Random(seed); selected=[]
    for key in sorted(buckets):
        rows=sorted(buckets[key],key=lambda x:x["id"]); rng.shuffle(rows)
        n=per_class_task.get(key[1],0); selected += rows[:n]
    return sorted(selected,key=lambda x:x["id"])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,default=ROOT/"phase3/configs/lora-pilot-v1.json"); ap.add_argument("--output-dir",type=Path,default=ROOT/"phase3/generated/lora-pilot-v1"); a=ap.parse_args()
    c=read_json(a.config); seed=c["pilot_subset"]["seed"]
    train=read_json(ROOT/c["data"]["train_repo_path"]); val=read_json(ROOT/c["data"]["validation_repo_path"])
    tr=choose(train,c["pilot_subset"]["train_per_class_task"],seed); va=choose(val,c["pilot_subset"]["validation_per_class_task"],seed+"-validation")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    train_path=a.output_dir/"pilot-train.json"; val_path=a.output_dir/"pilot-validation.json"; manifest_path=a.output_dir/"pilot-subset-manifest.json"
    write_json(train_path,tr); write_json(val_path,va)
    manifest={"schema_version":1,"kind":"lora_pilot_subset","seed":seed,"train_ids":[x["id"] for x in tr],"validation_ids":[x["id"] for x in va],"train_count":len(tr),"validation_count":len(va),"source_release_sha256":c["data"]["release_hashes"]}
    manifest["manifest_sha256"]=canonical_sha256(manifest,omit_key="manifest_sha256"); write_json(manifest_path,manifest)
    print(manifest_path)
if __name__=="__main__": main()
