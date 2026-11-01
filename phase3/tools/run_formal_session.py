#!/usr/bin/env python3
"""Bound one cloud session, including validation. No provisioning or credential APIs."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from member3_common import ROOT, canonical_sha256, read_json, sha256_file, write_json, validate_resource_limits
from run_lora_training import formal_telemetry_contract
from runpod_stop_watchdog import parse_deadline
from plan_formal_validation import plan


def positive(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and value>0


def check_session(session,config,now=None):
    now=time.time() if now is None else now
    if session.get('session_sha256') != canonical_sha256(session,omit_key='session_sha256'):
        raise ValueError('session hash mismatch')
    if session.get('config_sha256') != canonical_sha256(config,omit_key='config_sha256'):
        raise ValueError('session does not bind config')
    if not isinstance(session.get('pod_id'),str) or not re.fullmatch(r'[A-Za-z0-9_-]+',session['pod_id']):
        raise ValueError('actual Pod ID required')
    if not isinstance(session.get('attempt_id'),str) or not re.fullmatch(r'[A-Za-z0-9_-]+',session['attempt_id']):
        raise ValueError('safe unique attempt ID required')
    for key in ('max_session_usd','actual_hourly_usd','training_minutes'):
        if not positive(session.get(key)): raise ValueError('finite positive '+key+' required')
    start,end=parse_deadline(session['started_at_utc']),parse_deadline(session['deadline'])
    if start>now+5 or not start<=now<end: raise ValueError('session must already have started and not expired')
    minutes=(end-start)/60
    limits=config.get('resource_limits')
    errors=validate_resource_limits(limits)
    if errors: raise ValueError('; '.join(errors))
    if (minutes>limits['max_wall_clock_minutes'] or session['max_session_usd']>limits['total_budget_usd']
            or session['actual_hourly_usd']>limits['max_hourly_usd']
            or minutes*session['actual_hourly_usd']/60>session['max_session_usd']+1e-9
            or session['training_minutes']>=minutes):
        raise ValueError('whole-session limits do not fit approved config; include validation and startup')
    if Path(session['resource_ledger']).resolve()!=Path('/workspace/phase3/member3/formal/resource-ledger.json'):
        raise ValueError('shared formal optimizer ledger required')
    return end


def check_guard(path,session,now=None):
    now=time.time() if now is None else now
    report=read_json(path)
    if (report.get('kind')!='runpod_stop_watchdog' or report.get('pod_id')!=session['pod_id']
            or parse_deadline(report['deadline'])!=parse_deadline(session['deadline'])
            or report.get('execution_state')!='armed' or report.get('observed_state')!='RUNNING'):
        raise ValueError('matching armed external watchdog required')
    age=now-parse_deadline(report['heartbeat_utc'])
    if not -5<=age<=30: raise ValueError('watchdog heartbeat missing/stale or clock skew')


def commands(config,session,paths):
    formal_telemetry_contract(config)
    py=config['runtime']['python']; tools=ROOT/'phase3/tools'
    run=Path(config['output']['run_dir'].replace('{attempt_id}',session['attempt_id']))
    if any(x in str(run) for x in ('{','}')): raise ValueError('unresolved output directory')
    common=['--config',str(paths['config']),'--inventory',str(paths['inventory']),
            '--interface',str(paths['interface']),'--pilot-review',str(paths['review'])]
    out=paths['output']; attempt=out/'training-attempt.json'
    jobs=[('preflight',[py,str(tools/'member3_cloud_preflight.py'),
        '--report',str(out/'preflight.json'),'--data-dir',str(paths['data_dir']),
        '--model-dir',config['runtime']['model_path'],'--checkout',config['runtime']['working_directory'],
        '--full-model-hashes']),
        ('formal_gate',[py,str(tools/'validate_lora_config.py'),'--mode','formal',*common,
         '--report',str(out/'formal-gate.json')]),
        ('model_inputs',[py,str(tools/'check_lora_model_inputs.py'),'--config',str(paths['config']),
         '--inventory',str(paths['inventory']),'--interface',str(paths['interface']),
         '--report',str(out/'model-inputs.json'),'--execute']),
        ('training',[py,str(tools/'run_lora_training.py'),'--mode','formal',*common,
         '--attempt-id',session['attempt_id'],'--attempt-record',str(attempt),
         '--resource-ledger',session['resource_ledger'],'--attempt-minutes',str(session['training_minutes']),
         '--telemetry','--execute'])]
    val=plan(config,paths['config'],run,out/'validation',paths['interface'],paths['inventory'],paths['review'])
    jobs += [(f'validation_{j["step"]}',j['command']) for j in val['jobs']]
    jobs.append(('collect',[py,str(tools/'plan_formal_validation.py'),'--mode','collect',
        '--config',str(paths['config']),'--training-dir',str(run),'--validation-dir',str(out/'validation'),
        '--attempt-record',str(attempt),'--output',str(out/'checkpoint-comparison.json')]))
    return run,jobs


def terminate(process):
    if process.poll() is not None: return
    os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL); process.wait()


def bounded(command,log,deadline,monotonic_end,guard):
    """Poll heartbeat and both clocks while child runs; kill its entire process group."""
    with log.open('xb') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                if time.time()>=deadline or time.monotonic()>=monotonic_end:
                    raise TimeoutError('whole-session deadline reached')
                guard()
                code=process.poll()
                if code is not None: return code
                time.sleep(1)
        finally: terminate(process)


def execute(config,session,paths,runner=bounded):
    deadline=check_session(session,config)
    check_guard(paths['guard'],session)
    inventory=read_json(paths['inventory'])
    age=time.time()-parse_deadline(inventory['generated_at_utc'])
    if (inventory.get('report_sha256')!=canonical_sha256(inventory,omit_key='report_sha256')
            or not -5<=age<=86400):
        raise ValueError('fresh (within 24h) hash-verified module inventory required')
    run,jobs=commands(config,session,paths)
    if run.exists() or paths['output'].exists(): raise ValueError('fresh training and evidence directories required')
    paths['output'].mkdir(parents=True,exist_ok=False)
    status=paths['output']/'session-status.json'
    bindings={k:sha256_file(paths[k]) for k in ('config','inventory','interface','review')}
    record=dict(kind='formal_session',pod_id=session['pod_id'],deadline=session['deadline'],
        session_sha256=session['session_sha256'],file_bindings=bindings,state='running',jobs=[],
        stop_requested=False,pod_stop_confirmed=False,formal_Eval80_completed=False,final_acceptance=False)
    mono=time.monotonic()+max(0,deadline-time.time())
    def guard():
        check_guard(paths['guard'],session)
        if any(sha256_file(paths[k])!=v for k,v in bindings.items()):
            raise ValueError('bound input changed during session')
    try:
        for name,command in jobs:
            guard()
            record['active_job']=name; write_json(status,record)
            job_end=min(deadline,time.time()+session['training_minutes']*60) if name=='training' else deadline
            code=runner(command,paths['output']/(name+'.log'),job_end,
                        min(mono,time.monotonic()+max(0,job_end-time.time())),guard)
            record['jobs'].append(dict(name=name,returncode=code))
            if code!=0: raise RuntimeError('job failed: '+name)
            write_json(status,record)
        record['state']='training_and_checkpoint_validation_completed'
        record['guidance_content_review']='pending'
    except BaseException as error:
        record['state']='failed'; record['failure_type']=type(error).__name__
        # Keep full diagnostics in job logs; never record arbitrary credentials.
        raise
    finally:
        record['stop_requested']=True
        record['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        write_json(status,record)
    return record


def main():
    def interrupted(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('config','session','inventory','interface','review','guard','data-dir','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args(); config=read_json(a.config); session=read_json(a.session)
    paths={k:getattr(a,k).resolve() for k in ('config','inventory','interface','review','guard','data_dir','output')}
    if not a.execute:
        _,jobs=commands(config,session,paths)
        print(json.dumps(dict(state='planned_not_executed',jobs=jobs,pod_operations=False),indent=2));return
    execute(config,session,paths)


if __name__=='__main__': main()
