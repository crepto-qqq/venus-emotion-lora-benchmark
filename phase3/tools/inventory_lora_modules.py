#!/usr/bin/env python3
"""Read-only live inventory of candidate LoRA modules. No optimizer/backward is created."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from member3_common import ROOT, TOOL_VERSION, canonical_sha256, git_commit, read_json, write_json
from common import utc_now

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,default=ROOT/"phase3/configs/lora-pilot-v1.json"); ap.add_argument("--suffix",action="append",default=["c_attn","attn.c_proj","w1","w2"]); ap.add_argument("--report",type=Path,required=True); a=ap.parse_args()
    c=read_json(a.config)
    import torch
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(c["runtime"]["model_path"],trust_remote_code=True,torch_dtype=torch.bfloat16,device_map="cpu")
    matched=[]; visual=[]
    for name,module in model.named_modules():
        if any(name.endswith(s) for s in a.suffix):
            matched.append(name)
            if ".visual" in name or name.startswith("visual"): visual.append(name)
    total=sum(p.numel() for p in model.parameters())
    rank_counts={}
    for rank in (8,16,32):
        # PEFT LoRA adds A(out? actually in->r) + B(r->out) to each target Linear/Conv1D-like weight.
        extra=0
        for name,module in model.named_modules():
            if name in matched and hasattr(module,"weight") and getattr(module.weight,"ndim",0)==2:
                d0,d1=module.weight.shape; extra += rank*(int(d0)+int(d1))
        rank_counts[str(rank)]=extra
    report={"schema_version":1,"kind":"lora_module_inventory","tool_version":TOOL_VERSION,"generated_at_utc":utc_now(),"project_commit":git_commit(),"model_revision":c["runtime"]["expected_model_revision"],"upstream_commit":c["runtime"]["expected_upstream_commit"],"suffix_rules":a.suffix,"matched_module_names":matched,"matched_module_count":len(matched),"visual_modules_targeted":bool(visual),"visual_matches":visual,"total_base_parameters":total,"candidate_trainable_parameter_estimates":rank_counts,"observed_parameter_dtypes":sorted({str(p.dtype) for p in model.parameters()}),"fix_vit":c["training"]["fix_vit"],"modules_to_save":c["lora"]["modules_to_save"],"q_lora":False,"result":"passed" if matched and not visual else "failed"}
    report["report_sha256"]=canonical_sha256(report,omit_key="report_sha256"); write_json(a.report,report); print(json.dumps(report,indent=2))
    raise SystemExit(0 if report["result"]=="passed" else 2)
if __name__=="__main__": main()
