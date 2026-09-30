#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from member3_common import ROOT, canonical_sha256, dotted_leaf_paths, merge_patch, read_json, write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base",type=Path,default=ROOT/"phase3/configs/lora-pilot-v1.json"); ap.add_argument("--matrix",type=Path,default=ROOT/"phase3/configs/lora-experiment-matrix-v1.json"); ap.add_argument("--output-dir",type=Path,default=ROOT/"phase3/generated/lora-pilot-v1"); a=ap.parse_args()
    base=read_json(a.base); matrix=read_json(a.matrix); candidates=matrix["candidates"]
    if len(candidates)>3: raise SystemExit("matrix exceeds three optimizer candidates")
    allowed=set(matrix["allowed_override_fields"]); a.output_dir.mkdir(parents=True,exist_ok=True)
    for cand in candidates:
        changed=dotted_leaf_paths(cand["merge_patch"])
        if not changed<=allowed: raise SystemExit(f"{cand['id']}: forbidden override(s): {sorted(changed-allowed)}")
        cfg=merge_patch(base,cand["merge_patch"]); cfg["candidate_id"]=cand["id"]; cfg["matrix_sha256"]=canonical_sha256(matrix); cfg["config_sha256"]=canonical_sha256(cfg,omit_key="config_sha256")
        write_json(a.output_dir/f"{cand['id']}.json",cfg)
        print(cand["id"],cfg["config_sha256"])
if __name__=="__main__": main()
