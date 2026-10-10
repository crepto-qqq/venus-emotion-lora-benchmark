#!/usr/bin/env python3
"""Read-only cloud preflight. No model loading, optimizer, downloads or dependency writes."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
from member3_common import ROOT, read_json, sha256_file, write_json, verify_release_hashes, verify_patched_checkout
from common import utc_now


def verify_runtime_data(directory):
    failures=[]
    release=ROOT/'phase3/releases/bridge-train-v1'
    from member3_common import EXPECTED_RELEASE_HASHES
    for name,expected in EXPECTED_RELEASE_HASHES.items():
        path=directory/name
        if not path.is_file() or sha256_file(path)!=expected:failures.append('runtime release mismatch: '+name)
    manifest=[json.loads(line) for line in (release/'bridge-train-v1.manifest.jsonl').read_text().splitlines() if line.strip()]
    # Use paths from actual prompts: manifests retain separate source identities.
    records=read_json(release/'bridge-train-v1.train.json')+read_json(release/'bridge-train-v1.validation.json')
    paths=set()
    for row in records:
        paths.update(re.findall(r'<img>(.*?)</img>',row['conversations'][0]['value']))
    image_hashes={item['runtime_image_path']:item['image_sha256'] for item in manifest}
    for path in sorted(paths):
        if not Path(path).is_file():failures.append('missing runtime image: '+path)
        elif sha256_file(Path(path))!=image_hashes.get(path):failures.append('runtime image hash mismatch: '+path)
    return failures,dict(expected_image_paths=len(paths),checked_release_files=len(EXPECTED_RELEASE_HASHES),manifest_records=len(manifest))


def verify_model(directory,full_hashes):
    snapshot=read_json(ROOT/'phase3/configs/model-snapshot-manifest.json')
    failures=[]
    for item in snapshot['files']:
        path=directory/item['path']
        if not path.is_file():failures.append('missing model file: '+item['path'])
        elif path.stat().st_size!=item['size']:failures.append('model size mismatch: '+item['path'])
        elif (full_hashes or item['storage']!='lfs') and sha256_file(path)!=item['sha256']:failures.append('model hash mismatch: '+item['path'])
    return failures,dict(file_count=snapshot['file_count'],weight_hashes_verified=full_hashes)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,default=Path('/workspace/phase3/data/bridge-train-v1'))
    ap.add_argument('--model-dir',type=Path,default=Path('/workspace/models/Venus-Q-Stage1'))
    ap.add_argument('--checkout',type=Path,default=Path('/workspace/phase3/upstream/Qwen-VL-finetune'))
    ap.add_argument('--full-model-hashes',action='store_true')
    a=ap.parse_args()
    if a.report.exists():raise ValueError('new report path required')
    failures=verify_release_hashes();data_fail,data=verify_runtime_data(a.data_dir);failures+=data_fail
    model_fail,model=verify_model(a.model_dir,a.full_model_hashes);failures+=model_fail
    lock=read_json(ROOT/'phase3/configs/source-lock.json')
    try:
        actual=subprocess.check_output(['git','-C',str(a.checkout),'rev-parse','HEAD'],text=True).strip()
        if actual!=lock['upstream']['commit']:failures.append('upstream commit mismatch')
        else:
            try:verify_patched_checkout(a.checkout,lock)
            except (ValueError,subprocess.CalledProcessError):failures.append('patched training source mismatch')
    except subprocess.CalledProcessError:
        actual=None;failures.append('cannot read upstream commit')
    versions={}
    for package,expected in [('torch','2.0.1'),('transformers','4.32.0'),('peft','0.5.0'),('jsonschema','4.19.0')]:
        try:
            versions[package]=importlib.metadata.version(package)
            if versions[package].split('+')[0]!=expected:failures.append(package+' version mismatch')
        except importlib.metadata.PackageNotFoundError:failures.append(package+' missing')
    report=dict(kind='member3_cloud_preflight',generated_at_utc=utc_now(),result='passed' if not failures else 'failed',failures=failures,data=data,model=model,upstream_commit=actual,versions=versions,model_loaded=False,training_executed=False)
    write_json(a.report,report);print(json.dumps(report,indent=2));raise SystemExit(0 if not failures else 2)

if __name__=='__main__':main()
