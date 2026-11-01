#!/usr/bin/env python3
"""Deadline stop-only watchdog. Default is offline planning. No create/start/delete API."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import time
import urllib.request
from member3_common import write_json

BASE='https://rest.runpod.io/v1/pods'


def parse_deadline(value):
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None: raise ValueError('deadline must include timezone')
    return dt.timestamp()


def request(method,path,key):
    req=urllib.request.Request(BASE+path,headers={'Authorization':'Bearer '+key,
        'User-Agent':'Mozilla/5.0'},method=method)
    with urllib.request.urlopen(req,timeout=15) as response:
        data=response.read()
        return json.loads(data) if data.strip() else None


def pod_state(pod_id,key,transport=request):
    pods=transport('GET','',key)
    if not isinstance(pods,list): raise ValueError('unexpected Pod list response')
    matches=[p for p in pods if p.get('id')==pod_id]
    if len(matches)!=1: raise ValueError('Pod missing or ambiguous; cannot claim stopped')
    return matches[0].get('desiredStatus')


def stop_and_confirm(pod_id,key,transport=request,sleep=time.sleep,attempts=6):
    for _ in range(attempts):
        try:
            if pod_state(pod_id,key,transport)=='EXITED': return True
            transport('POST','/'+pod_id+'/stop',key)
            if pod_state(pod_id,key,transport)=='EXITED': return True
        except Exception:
            # Do not log raw exceptions: URLs/headers may contain credentials.
            pass
        sleep(5)
    return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pod-id',required=True)
    ap.add_argument('--deadline',required=True,help='absolute ISO8601 time with timezone, covers whole paid session')
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--key-file',type=Path,help='plain API key in owner-only file, not S3 key')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--stop-request',type=Path,help='local relay creates this file to request early stop')
    ap.add_argument('--wait-for-start',action='store_true',help='arm before a stopped Pod starts; deadline is never extended')
    a=ap.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',a.pod_id): raise ValueError('invalid Pod ID')
    deadline=parse_deadline(a.deadline)
    record=dict(kind='runpod_stop_watchdog',pod_id=a.pod_id,deadline=a.deadline,execution_state='planned',pid=os.getpid())
    if not a.execute: print(json.dumps(record,indent=2));return
    if not a.key_file or a.key_file.is_symlink() or not a.key_file.is_file(): raise ValueError('owner-only API key file required')
    info=a.key_file.stat()
    if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077: raise ValueError('key file must be owned by current user with no group/other access')
    key=a.key_file.read_text().strip()
    if not key: raise ValueError('empty API key')
    state=pod_state(a.pod_id,key)
    monotonic_end=time.monotonic()+max(0,deadline-time.time())
    while a.wait_for_start and state=='EXITED' and time.time()<deadline and time.monotonic()<monotonic_end:
        record.update(execution_state='waiting_for_start',observed_state=state,
            heartbeat_utc=datetime.now(timezone.utc).isoformat())
        write_json(a.report,record)
        time.sleep(min(5,max(0,deadline-time.time()),max(0,monotonic_end-time.monotonic())))
        state=pod_state(a.pod_id,key)
    if state=='EXITED':
        record.update(execution_state='stop_state_observed',observed_desired_status=state)
        write_json(a.report,record);return
    if state!='RUNNING': raise ValueError('unexpected Pod state; watchdog not armed')
    # Monotonic bound prevents a backward wall-clock adjustment extending the session.
    # Preserve the same monotonic deadline, including the wait before startup.
    record.update(execution_state='armed',observed_state=state)
    while time.time()<deadline and time.monotonic()<monotonic_end and not (a.stop_request and a.stop_request.exists()):
        record['heartbeat_utc']=datetime.now(timezone.utc).isoformat();write_json(a.report,record)
        time.sleep(min(5,max(0,deadline-time.time()),max(0,monotonic_end-time.monotonic())))
    record['execution_state']='stop_requested';write_json(a.report,record)
    success=stop_and_confirm(a.pod_id,key)
    record.update(execution_state='stop_state_observed' if success else 'stop_unconfirmed',finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(a.report,record)
    raise SystemExit(0 if success else 2)

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print('watchdog failed: '+type(error).__name__+'; verify Pod state manually',file=__import__('sys').stderr)
        raise SystemExit(2)
