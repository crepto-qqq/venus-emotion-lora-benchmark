#!/usr/bin/env python3
"""Plan or generate BridgeTrain-only adapter validation. Never provisions resources."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from member3_common import ROOT, read_json, canonical_sha256, sha256_file, write_json
from check_lora_model_inputs import collate_training_batch


def validation_rows(path):
    rows=read_json(path)
    frozen=read_json(ROOT/'phase3/releases/bridge-train-v1/bridge-train-v1.validation.json')
    if rows != frozen:
        raise ValueError('validation must exactly match frozen BridgeTrain validation; no Eval20/Eval80')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate validation IDs')
    for row in rows:
        turns=row['conversations']
        if len(turns)!=2 or turns[0]['from']!='user' or turns[1]['from']!='assistant':
            raise ValueError('expected one user/assistant pair')
    return rows


def user_query(row):
    # Never append the assistant reference to the generation prompt.
    return row['conversations'][0]['value']


def adapter_identity(directory):
    files=[directory/'adapter_config.json']
    weights=[directory/name for name in ('adapter_model.bin','adapter_model.safetensors') if (directory/name).is_file()]
    if len(weights)!=1 or not files[0].is_file():
        raise ValueError('adapter needs config and exactly one supported weight file')
    files+=weights
    return {file.name:sha256_file(file) for file in files}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--adapter',type=Path,required=True)
    ap.add_argument('--validation',type=Path,default=ROOT/'phase3/releases/bridge-train-v1/bridge-train-v1.validation.json')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--max-new-tokens',type=int,default=512)
    ap.add_argument('--interface',type=Path,default=ROOT/'phase3/configs/qwen-training-interface-v1.json')
    ap.add_argument('--inventory',type=Path)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--pilot-review',type=Path,help='formal training decision record when validating a formal adapter')
    args=ap.parse_args()
    if args.max_new_tokens<=0: raise ValueError('max-new-tokens must be positive')
    config=read_json(args.config); rows=validation_rows(args.validation)
    plan=dict(kind='lora_validation_generation',config_sha256=canonical_sha256(config,omit_key='config_sha256'),validation_sha256=sha256_file(args.validation),sample_count=len(rows),adapter_path=str(args.adapter),max_new_tokens=args.max_new_tokens,generation=dict(do_sample=False),execution_state='planned')
    if not args.execute:
        print(json.dumps(plan,indent=2));return
    if not args.inventory: raise ValueError('runtime inventory required for execution')
    mode = 'formal' if config.get('kind') == 'lora_formal_config' else 'pilot'
    validator = [sys.executable,str(ROOT/'phase3/tools/validate_lora_config.py'),'--mode',mode,'--config',str(args.config),'--inventory',str(args.inventory),'--interface',str(args.interface)]
    if args.pilot_review:
        validator += ['--pilot-review',str(args.pilot_review)]
    subprocess.run(validator,check=True)
    if args.output_dir.exists(): raise ValueError('use a new output directory; existing evidence will not be overwritten')
    source=Path(config['runtime']['working_directory'])/'finetune.py'
    from common import utc_now
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    actual_commit=subprocess.check_output(['git','-C',str(source.parent),'rev-parse','HEAD'],text=True).strip()
    if actual_commit!=config['runtime']['expected_upstream_commit']: raise ValueError('upstream commit mismatch')
    # The interface inventory binds the reviewed, patched finetune source.
    interface=read_json(args.interface)
    if interface.get('finetune_py_sha256')!=sha256_file(source):
        raise ValueError('finetune source hash must match verified interface inventory')
    spec=importlib.util.spec_from_file_location('member3_pinned_finetune',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;sys.path.insert(0,str(source.parent));spec.loader.exec_module(module)
    for row in rows:
        images=re.findall(r'<img>(.*?)</img>',user_query(row))
        if len(images)!=1 or not Path(images[0]).is_file(): raise ValueError(f"missing image for {row['id']}")
    plan['adapter_files']=adapter_identity(args.adapter)
    plan['adapter_bundle_sha256']=canonical_sha256(plan['adapter_files'])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    write_json(args.output_dir/'run.json',plan)
    model_path=config['runtime']['model_path']
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=True)
    tokenizer.pad_token_id=tokenizer.eod_id
    base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=True,device_map={'':'cuda:0'},bf16=True)
    model=PeftModel.from_pretrained(base,str(args.adapter),is_trainable=False).eval()
    if any(parameter.device.type!='cuda' for parameter in model.parameters()): raise RuntimeError('CPU/disk offload forbidden')
    model.generation_config=GenerationConfig.from_pretrained(model_path,local_files_only=True,trust_remote_code=True)
    model.generation_config.do_sample=False
    torch.cuda.reset_peak_memory_stats()
    try:
        with (args.output_dir/'predictions.jsonl').open('x',encoding='utf-8') as output:
            for row in rows:
                batch=module.preprocess([row['conversations']],tokenizer,config['training']['max_sequence_length'])
                batch=collate_training_batch(batch,tokenizer,torch,'cuda:0')
                supervised=int((batch['labels'][:,1:]!=-100).sum().item())
                if supervised==0: raise RuntimeError('zero supervised tokens')
                with torch.inference_mode():
                    loss=float(model(**batch).loss.item())
                    response,_=model.chat(tokenizer,query=user_query(row),history=None,do_sample=False,max_new_tokens=args.max_new_tokens)
                if not math.isfinite(loss): raise RuntimeError('nonfinite validation loss')
                output.write(json.dumps(dict(id=row['id'],prediction=response,loss=loss,supervised_tokens=supervised),ensure_ascii=False)+'\n');output.flush()
        plan.update(execution_state='completed',prediction_count=len(rows),peak_vram_bytes=torch.cuda.max_memory_allocated(),completed_at_utc=utc_now())
    except BaseException as error:
        plan.update(execution_state='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        plan['report_sha256']=canonical_sha256(plan,omit_key='report_sha256');write_json(args.output_dir/'run.json',plan)

if __name__=='__main__':main()
