#!/usr/bin/env python3
"""Package current Phase 3 source bytes, not only committed HEAD. Offline only."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from member3_common import ROOT, canonical_sha256, sha256_file, write_json


def bundle_files(root=ROOT):
    names=subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','--','phase3','Docs/phase3/MEMBER3_START_HERE_EN.md'],text=True).splitlines()
    selected=[]
    allowed={'.py','.md','.json','.jsonl','.sh','.patch','.txt','.yml','.yaml'}
    for name in sorted(set(names)):
        if any(part in {'__pycache__','generated','envs','.git'} for part in Path(name).parts):continue
        if Path(name).suffix not in allowed:continue
        path=root/name
        if path.is_symlink():raise ValueError('symlink not allowed in bundle: '+name)
        if path.is_file():selected.append(name)
    return selected


def verify_bundle(path):
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('duplicate archive member')
        for member in members:
            if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:raise ValueError('unsafe archive member')
        manifest=json.load(archive.extractfile('bundle-manifest.json'))
        if manifest.get('bundle_identity_sha256')!=canonical_sha256(manifest,omit_key='bundle_identity_sha256'):raise ValueError('manifest identity mismatch')
        expected={item['path']:item for item in manifest['files']}
        if set(names)!=set(expected)|{'bundle-manifest.json'}:raise ValueError('bundle member set mismatch')
        for name,item in expected.items():
            data=archive.extractfile(name).read()
            if len(data)!=item['size'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('bundle file mismatch: '+name)
        return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--verify',type=Path)
    a=ap.parse_args()
    if a.verify:
        manifest=verify_bundle(a.verify);print(json.dumps(dict(result='passed',files=len(manifest['files']),bundle_identity_sha256=manifest['bundle_identity_sha256'])));return
    if not a.output:raise ValueError('--output or --verify required')
    if a.output.exists():raise ValueError('new archive path required')
    payloads={name:(ROOT/name).read_bytes() for name in bundle_files()}
    manifest=dict(schema_version=1,kind='member3_local_source_bundle',base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),contains_uncommitted_changes=True,training_executed=False,files=[dict(path=name,size=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in payloads.items()])
    manifest['bundle_identity_sha256']=canonical_sha256(manifest)
    payloads['bundle-manifest.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as zipped:
            with tarfile.open(fileobj=zipped,mode='w') as archive:
                for name,data in sorted(payloads.items()):
                    item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644
                    archive.addfile(item,io.BytesIO(data))
    verified=verify_bundle(a.output)
    write_json(a.output.with_suffix(a.output.suffix+'.verification.json'),dict(result='passed',archive_sha256=sha256_file(a.output),file_count=len(verified['files']),bundle_identity_sha256=verified['bundle_identity_sha256'],base_commit=manifest['base_commit']))
    print(a.output)

if __name__=='__main__':main()
