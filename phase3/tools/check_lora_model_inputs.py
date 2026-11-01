#!/usr/bin/env python3
"""Member 3 two-task LoRA forward/backward check, with zero optimizer steps.

Default: offline plan. --execute requires verified inventory/interface/config.
No downloads, saves, optimizer creation, or cloud resource operations.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from member3_common import ROOT, canonical_sha256, read_json, sha256_file, verify_release_hashes, verify_patched_checkout, task_type, write_json
from make_lora_pilot_subset import choose
from common import utc_now


def select_samples(config):
    failures=verify_release_hashes()
    if failures:raise ValueError('; '.join(failures))
    rows=read_json(ROOT/config['data']['train_repo_path'])
    if config.get('kind') == 'lora_formal_config':
        # This is an input smoke check, not a formal training subset.
        subset=sorted(rows,key=lambda row:row['id'])
    else:
        spec=config['pilot_subset']
        subset=choose(rows,spec['train_per_class_task'],spec['seed'])
    samples=[]
    for task in ('classification','joint_guidance'):
        candidates=[r for r in subset if task_type(r['id'])==task]
        if not candidates:raise ValueError('training samples missing task '+task)
        samples.append(candidates[0])
    return samples


def prepare_model_batch(batch,torch,device):
    """Preserve token values, using int64 indices required by CrossEntropyLoss."""
    for name in ('input_ids','labels'):
        if batch[name].dtype not in (torch.int32,torch.int64):
            raise ValueError('integer token tensor required: '+name)
    return {name:value.to(device=device,dtype=torch.long) if name in ('input_ids','labels')
            else value.to(device=device) for name,value in batch.items()}


def collate_training_batch(batch,tokenizer,torch,device):
    """Match Trainer(tokenizer=...) in pinned Transformers 4.32.0."""
    from transformers import DataCollatorWithPadding
    features=[{name:value[index] for name,value in batch.items()}
              for index in range(batch['input_ids'].shape[0])]
    collated=DataCollatorWithPadding(tokenizer)(features)
    if any(not torch.equal(value,collated[name]) for name,value in batch.items()):
        raise ValueError('collator changed padded token values or supervision mask')
    return prepare_model_batch(collated,torch,device)


def parameter_checks(model,torch,require_cuda=True):
    named=list(model.named_parameters())
    trainable=[(name,p) for name,p in named if p.requires_grad]
    unexpected=[name for name,p in trainable if 'lora_A.' not in name and 'lora_B.' not in name]
    visual=[name for name,p in trainable if '.visual.' in name or name.startswith('visual.')]
    if not trainable or unexpected or visual:raise ValueError('trainable parameters must be nonvisual LoRA A/B only')
    if any(p.grad is not None for name,p in named if not p.requires_grad):raise ValueError('frozen parameter has gradient')
    if require_cuda and any(p.device.type!='cuda' for name,p in named):raise ValueError('CPU/disk offload forbidden')
    if any(p.dtype!=torch.float32 for name,p in trainable):raise ValueError('LoRA adapter parameters must be float32; no silent dtype conversion')
    if any(p.is_floating_point() and p.dtype!=torch.bfloat16 for name,p in named if not p.requires_grad):raise ValueError('frozen base parameters must be bfloat16')
    return trainable,dict(trainable_parameter_count=sum(p.numel() for name,p in trainable),trainable_names=[name for name,p in trainable],adapter_dtype='float32',base_dtype='bfloat16',unexpected_trainable_parameters=False,visual_trainable_parameters=False)


def adapter_digest(trainable):
    digest=hashlib.sha256()
    for name,p in sorted(trainable,key=lambda pair:pair[0]):
        digest.update(name.encode());digest.update(str(tuple(p.shape)).encode())
        digest.update(p.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def check_task(model,batch,torch,require_cuda=True):
    model.zero_grad(set_to_none=True)
    supervised=int((batch['labels'][:,1:]!=-100).sum().item())
    if supervised<=0:raise ValueError('zero supervised tokens after causal shift')
    trainable,params=parameter_checks(model,torch,require_cuda)
    before=adapter_digest(trainable)
    outputs=model(**batch)
    loss=outputs.loss
    if loss.numel()!=1 or not loss.requires_grad:raise ValueError('loss must be differentiable scalar')
    value=float(loss.detach().float().cpu().item())
    if not math.isfinite(value):raise ValueError('nonfinite loss')
    loss.backward()
    nonzero=[];missing=[]
    for name,p in trainable:
        if p.grad is None:missing.append(name)
        elif not bool(torch.isfinite(p.grad).all().item()):raise ValueError('nonfinite adapter gradient: '+name)
        elif bool(p.grad.ne(0).any().item()):nonzero.append(name)
    # At zero-initialized LoRA B, A gradients may legitimately be zero.
    if not nonzero:raise ValueError('no finite nonzero LoRA gradient')
    if missing:raise ValueError('trainable adapter parameter missing gradient: '+missing[0])
    parameter_checks(model,torch,require_cuda)
    after=adapter_digest(trainable)
    if before!=after:raise ValueError('adapter weights changed without optimizer step')
    return dict(loss=value,loss_dtype=str(loss.dtype),supervised_tokens=supervised,nonzero_gradient_names=nonzero,adapter_parameters_unchanged=True,adapter_parameter_digest=after,forward_passed=True,backward_passed=True,optimizer_steps=0,parameter_checks=params)


def execute_check(args,config,samples,report):
    subprocess.run([sys.executable,str(ROOT/'phase3/tools/validate_lora_config.py'),'--mode','pilot','--config',str(args.config),'--inventory',str(args.inventory),'--interface',str(args.interface)],check=True)
    source=Path(config['runtime']['working_directory'])/'finetune.py'
    source_hash=verify_patched_checkout(source.parent,read_json(ROOT/'phase3/configs/source-lock.json'))
    interface=read_json(args.interface)
    if interface.get('finetune_py_sha256')!=source_hash:raise ValueError('interface/source hash mismatch')
    inventory=read_json(args.inventory)
    if inventory.get('report_sha256')!=canonical_sha256(inventory,omit_key='report_sha256'):raise ValueError('inventory self hash mismatch')
    for row in samples:
        paths=re.findall(r'<img>(.*?)</img>',row['conversations'][0]['value'])
        if len(paths)!=1 or not Path(paths[0]).is_file():raise ValueError('sample image unavailable: '+row['id'])
    # Imports occur only in explicitly requested execution mode.
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,get_peft_model
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA with BF16 support required')
    torch.manual_seed(config['training']['seed']);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    spec=importlib.util.spec_from_file_location('_member3_forward_source',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;sys.path.insert(0,str(source.parent));spec.loader.exec_module(module)
    tokenizer=AutoTokenizer.from_pretrained(config['runtime']['model_path'],model_max_length=config['training']['max_sequence_length'],padding_side='right',use_fast=False,local_files_only=True,trust_remote_code=True)
    tokenizer.pad_token_id=tokenizer.eod_id
    model=AutoModelForCausalLM.from_pretrained(config['runtime']['model_path'],torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},use_safetensors=True,trust_remote_code=True,local_files_only=True)
    model.config.use_cache=False
    visual=getattr(getattr(model,'transformer',None),'visual',None)
    if visual is None:raise ValueError('model visual tower not found')
    visual.requires_grad_(False)
    lora=config['lora']
    matched=[name for name,m in model.named_modules() if any(name.endswith(suffix) for suffix in lora['target_modules'])]
    if set(matched)!=set(inventory['matched_module_names']):raise ValueError('live target modules differ from inventory')
    model=get_peft_model(model,LoraConfig(r=lora['rank'],lora_alpha=lora['alpha'],target_modules=lora['target_modules'],lora_dropout=lora['dropout'],bias=lora['bias'],task_type='CAUSAL_LM',modules_to_save=None))
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable();model.enable_input_require_grads()
    model.train()
    trainable,params=parameter_checks(model,torch)
    estimate=inventory['candidate_trainable_parameter_estimates'].get(str(lora['rank']))
    if params['trainable_parameter_count']!=estimate:raise ValueError('actual LoRA parameter count differs from inventory estimate')
    report.update(parameter_checks=params,source_sha256=source_hash,inventory_sha256=inventory['report_sha256'],gpu_name=torch.cuda.get_device_name(0))
    observed=next((m for name,m in model.named_modules() if hasattr(m,'lora_A') and len(m.lora_A)>0),None)
    if observed is None:raise ValueError('injected LoRA wrapper not found')
    for row in samples:
        batch=module.preprocess([row['conversations']],tokenizer,config['training']['max_sequence_length'])
        batch=collate_training_batch(batch,tokenizer,torch,'cuda:0')
        dtypes=set()
        def hook(_module,inputs):
            if inputs and torch.is_tensor(inputs[0]) and inputs[0].is_floating_point():dtypes.add(inputs[0].dtype)
        handle=observed.register_forward_pre_hook(hook)
        try:result=check_task(model,batch,torch)
        finally:handle.remove()
        if dtypes!={torch.bfloat16}:raise ValueError('LoRA wrapper input was not exclusively BF16')
        result.update(sample_id=row['id'],task=task_type(row['id']),forward_input_dtype='bfloat16',
                      token_input_dtype=str(batch['input_ids'].dtype),labels_dtype=str(batch['labels'].dtype),
                      collator='transformers.DataCollatorWithPadding')
        report['tasks'].append(result);write_json(args.report,report)
    torch.cuda.synchronize()
    report['peak_vram_bytes']=torch.cuda.max_memory_allocated()
    report['result']='passed';report['execution_state']='completed'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--inventory',type=Path)
    ap.add_argument('--interface',type=Path,default=ROOT/'phase3/configs/qwen-training-interface-v1.json')
    ap.add_argument('--report',type=Path)
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();config=read_json(a.config);samples=select_samples(config)
    report=dict(schema_version=1,kind='member3_lora_model_input_check',config_sha256=canonical_sha256(config,omit_key='config_sha256'),created_at_utc=utc_now(),sample_ids=[row['id'] for row in samples],tasks=[],optimizer_steps=0,training_completed=False,adapter_saved=False,result='not_run',execution_state='planned')
    if not a.execute:print(json.dumps(report,indent=2));return
    if not a.inventory or not a.report:raise ValueError('execution requires inventory and new report path')
    if a.report.exists():raise ValueError('refusing to overwrite existing report')
    started=time.monotonic();report['execution_state']='started';write_json(a.report,report)
    try:execute_check(a,config,samples,report)
    except BaseException as error:
        report.update(result='failed',execution_state='failed',error_type=type(error).__name__)
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_at_utc=utc_now())
        report['report_sha256']=canonical_sha256(report,omit_key='report_sha256');write_json(a.report,report)

if __name__=='__main__':main()
