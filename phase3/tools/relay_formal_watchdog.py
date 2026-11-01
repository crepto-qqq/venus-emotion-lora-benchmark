#!/usr/bin/env python3
"""Mac-side heartbeat relay and early-stop request. API credentials stay with watchdog."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from member3_common import read_json, write_json
from runpod_stop_watchdog import parse_deadline
from run_formal_session import check_guard


def ssh_command(target,port,identity):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*@[A-Za-z0-9_][A-Za-z0-9_.-]*',target):
        raise ValueError('explicit user@host required')
    if not 1<=port<=65535: raise ValueError('invalid SSH port')
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
            '-p',str(port),'-i',str(identity),target]


def remote_path(path):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts or not p.is_relative_to('/workspace/phase3/member3/formal'):
        raise ValueError('isolated absolute formal path required')
    return shlex.quote(str(p))


def exchange(ssh,report,guard_path,status_path):
    guard=remote_path(guard_path); temporary=remote_path(str(guard_path)+'.relay-tmp')
    # Atomic replacement prevents cloud reader observing partially written JSON.
    subprocess.run(ssh+['umask 077; cat > '+temporary+' && mv '+temporary+' '+guard],
                   input=json.dumps(report).encode(),check=True,timeout=15,capture_output=True)
    path=remote_path(status_path)
    result=subprocess.run(ssh+['if test -f '+path+'; then cat '+path+'; else echo null; fi'],
                          check=True,timeout=15,capture_output=True)
    return json.loads(result.stdout)


def relay(session,ssh,report_path,guard_path,status_path,stop_request,relay_report,
          transfer=exchange,sleep=time.sleep):
    deadline=parse_deadline(session['deadline'])
    mono=time.monotonic()+max(0,deadline-time.time())
    if stop_request.exists(): raise ValueError('use fresh local stop-request path')
    state=dict(kind='formal_watchdog_relay',pod_id=session['pod_id'],state='running',pod_stop_confirmed=False)
    try:
        while time.time()<deadline and time.monotonic()<mono:
            report=read_json(report_path); check_guard(report_path,session)
            remote=transfer(ssh,report,guard_path,status_path)
            if remote is not None:
                if remote.get('pod_id')!=session['pod_id'] or remote.get('session_sha256')!=session['session_sha256']:
                    raise ValueError('cloud status identity mismatch')
                if remote.get('stop_requested') is True:
                    state['state']='early_stop_requested'; break
            state['heartbeat_utc']=datetime.now(timezone.utc).isoformat(); write_json(relay_report,state)
            sleep(5)
        else: state['state']='deadline_stop_requested'
    except BaseException as error:
        state.update(state='relay_failed_stop_requested',failure_type=type(error).__name__)
        raise
    finally:
        stop_request.parent.mkdir(parents=True,exist_ok=True)
        stop_request.touch(exist_ok=False)
        write_json(relay_report,state)
    return state


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('session','watchdog-report','identity','stop-request','relay-report'):
        ap.add_argument('--'+key,type=Path,required=True)
    for key in ('target','remote-guard','remote-status'): ap.add_argument('--'+key,required=True)
    ap.add_argument('--port',type=int,required=True); ap.add_argument('--execute',action='store_true')
    a=ap.parse_args(); session=read_json(a.session)
    ssh=ssh_command(a.target,a.port,a.identity)
    remote_path(a.remote_guard); remote_path(a.remote_status)
    if not a.execute:
        print(json.dumps(dict(state='planned_not_executed',ssh=ssh,api_key_uploaded=False)));return
    relay(session,ssh,a.watchdog_report,a.remote_guard,a.remote_status,a.stop_request,a.relay_report)


if __name__=='__main__': main()
