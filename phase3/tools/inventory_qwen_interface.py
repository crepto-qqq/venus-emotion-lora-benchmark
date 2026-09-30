#!/usr/bin/env python3
"""Inventory the exact HfArgumentParser dataclass surface from the pinned patched finetune.py."""
from __future__ import annotations
import argparse, ast, dataclasses, importlib.util, json, sys
from pathlib import Path
from member3_common import ROOT, canonical_sha256, read_json, sha256_file, write_json

def parser_class_names(source: str) -> list[str]:
    tree=ast.parse(source); names=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="HfArgumentParser" and node.args:
            arg=node.args[0]
            if isinstance(arg,(ast.Tuple,ast.List)):
                for e in arg.elts:
                    if isinstance(e,ast.Name): names.append(e.id)
    return names

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkout",type=Path,default=Path('/workspace/phase3/upstream/Qwen-VL-finetune')); ap.add_argument("--output",type=Path,default=ROOT/'phase3/configs/qwen-training-interface-v1.json'); a=ap.parse_args()
    src=a.checkout/'finetune.py'
    if not src.is_file(): raise SystemExit('pinned patched finetune.py not found')
    source=src.read_text(encoding='utf-8'); names=parser_class_names(source)
    spec=importlib.util.spec_from_file_location('_member3_pinned_finetune',src); mod=importlib.util.module_from_spec(spec); sys.path.insert(0,str(a.checkout)); spec.loader.exec_module(mod)
    fields=[]
    for cls_name in names:
        cls=getattr(mod,cls_name)
        for f in dataclasses.fields(cls):
            default=None if f.default is dataclasses.MISSING else f.default
            if f.default_factory is not dataclasses.MISSING: default='<factory>'
            fields.append({'owner':cls_name,'name':f.name,'type':str(f.type),'default':default,'required':f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING})
    cfg=read_json(a.output); cfg['source_inventory_state']='verified'; cfg['parser_inventory']=fields; cfg['parser_class_names']=names; cfg['finetune_py_sha256']=sha256_file(src); cfg['parser_field_set_sha256']=canonical_sha256(sorted((x['owner'],x['name']) for x in fields)); write_json(a.output,cfg); print(json.dumps({'classes':names,'field_count':len(fields),'parser_field_set_sha256':cfg['parser_field_set_sha256']},indent=2))
if __name__=='__main__': main()
