#!/usr/bin/env python3
"""Offline portable formal preparation bundle. Never approves or launches a run."""
import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from member3_common import ROOT, canonical_sha256, read_json, sha256_file, write_json
from prepare_member3_bundle import bundle_files, verify_bundle
from run_lora_training import formal_telemetry_contract


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n').encode()


def preparation_payloads(config_path, decision_path):
    config, decision = read_json(config_path), read_json(decision_path)
    digest = canonical_sha256(config, omit_key='config_sha256')
    formal_telemetry_contract(config)
    if config.get('config_sha256') != digest or decision.get('formal_config_sha256') != digest:
        raise ValueError('config and decision hash binding mismatch')
    if (decision.get('decision') != 'pending' or decision.get('human_confirmed') is not False
            or config.get('resource_limits') is not None):
        raise ValueError('this packager accepts pending unbudgeted drafts only')
    payloads = {'formal-preparation/formal-config-DRAFT.json': config_path.read_bytes()}
    portable = json.loads(json.dumps(decision))
    if not portable.get('reviewed_evidence'):
        raise ValueError('review evidence required')
    for index, item in enumerate(portable['reviewed_evidence']):
        source = Path(item['path'])
        if source.is_symlink() or sha256_file(source) != item['sha256']:
            raise ValueError('review evidence changed or is a symlink')
        name = f'formal-preparation/reviewed-evidence/{index+1:02d}-{source.name}'
        payloads[name] = source.read_bytes()
        item['path'] = name
    portable['path_policy'] = 'bundle_relative; materialize with prepare_formal_bundle.py before validation'
    payloads['formal-preparation/delegated-decision-PORTABLE-PENDING.json'] = encoded(portable)
    session=dict(state='draft_not_authorized',pod_id=None,attempt_id='formal1-DRAFT',
        started_at_utc=None,deadline=None,max_session_usd=None,actual_hourly_usd=None,
        training_minutes=None,resource_ledger='/workspace/phase3/member3/formal/resource-ledger.json',
        config_sha256=digest)
    session['session_sha256']=canonical_sha256(session)
    payloads['formal-preparation/session-DRAFT.json']=encoded(session)
    payloads['formal-preparation/readiness.json'] = encoded(dict(
        state='preparation_only', executable=False, training_executed=False,
        config_sha256=digest, credentials_included=False,
        remaining=['fresh cloud preflight and module inventory; rebind config',
            'deploy and live-check whole-session executor, heartbeat relay and external Pod stop watchdog',
            'explicit exact-config decision and formal resource limits',
            'formal training, three clean-process validations and output content review']))
    return payloads


def build(config_path, decision_path, output):
    if output.exists() or output.with_suffix(output.suffix+'.verification.json').exists():
        raise ValueError('refusing to overwrite bundle or verification')
    payloads = {name:(ROOT/name).read_bytes() for name in bundle_files()}
    payloads.update(preparation_payloads(config_path, decision_path))
    manifest = dict(schema_version=1, kind='member3_formal_preparation_bundle',
        base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        contains_uncommitted_changes=True, training_executed=False,
        files=[dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())
               for name,data in sorted(payloads.items())])
    manifest['bundle_identity_sha256'] = canonical_sha256(manifest)
    payloads['bundle-manifest.json'] = encoded(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as archive:
                for name,data in sorted(payloads.items()):
                    item=tarfile.TarInfo(name); item.size=len(data); item.mode=0o644
                    archive.addfile(item,io.BytesIO(data))
    verified=verify_bundle(output)
    result=dict(result='passed', archive_sha256=sha256_file(output),
        file_count=len(verified['files']), bundle_identity_sha256=verified['bundle_identity_sha256'],
        executable=False, training_executed=False)
    write_json(output.with_suffix(output.suffix+'.verification.json'),result)
    return result


def materialize(bundle_root, output):
    """Resolve evidence paths in a new pending record; never mutate source draft."""
    root=bundle_root.resolve()
    record=read_json(root/'formal-preparation/delegated-decision-PORTABLE-PENDING.json')
    if output.exists(): raise ValueError('refusing to overwrite decision')
    if record.get('decision') != 'pending' or record.get('human_confirmed') is not False:
        raise ValueError('pending decision required')
    config=read_json(root/'formal-preparation/formal-config-DRAFT.json')
    digest=canonical_sha256(config,omit_key='config_sha256')
    if config.get('config_sha256') != digest or record.get('formal_config_sha256') != digest:
        raise ValueError('config binding changed')
    for item in record['reviewed_evidence']:
        relative=Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe review path')
        source=root/relative
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError('review path escapes bundle')
        if sha256_file(source) != item['sha256']: raise ValueError('review evidence changed')
        item['path']=str(source.resolve())
    record['path_policy']='absolute paths in deployed bundle; still pending'
    write_json(output,record)
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build'); b.add_argument('--config',type=Path,required=True)
    b.add_argument('--decision',type=Path,required=True); b.add_argument('--output',type=Path,required=True)
    m=sub.add_parser('materialize'); m.add_argument('--bundle-root',type=Path,required=True)
    m.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.mode=='build': print(json.dumps(build(a.config,a.decision,a.output)))
    else:
        materialize(a.bundle_root,a.output)
        print(json.dumps(dict(state='pending',approved=False,output=str(a.output))))


if __name__=='__main__': main()
