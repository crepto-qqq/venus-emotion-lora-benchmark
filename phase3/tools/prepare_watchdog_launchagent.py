#!/usr/bin/env python3
"""Write a macOS LaunchAgent plan only; never loads it or contacts RunPod."""
import argparse
from pathlib import Path
import plistlib
import re
from runpod_stop_watchdog import parse_deadline


def make_plist(python,watchdog,pod_id,deadline,key_file,report,log,stop_request=None):
    if not re.fullmatch('[A-Za-z0-9_-]+',pod_id): raise ValueError('invalid Pod ID')
    parse_deadline(deadline)
    paths=[python,watchdog,key_file,report,log]
    if any(not Path(p).is_absolute() for p in paths): raise ValueError('absolute paths required')
    result=dict(Label='edu.soft3888.member3.stop.'+pod_id,
                ProgramArguments=['/usr/bin/caffeinate','-is',str(python),str(watchdog),'--pod-id',pod_id,'--deadline',deadline,'--key-file',str(key_file),'--report',str(report),'--execute'],
                RunAtLoad=True,KeepAlive={'SuccessfulExit':False},ThrottleInterval=10,
                StandardOutPath=str(log),StandardErrorPath=str(log))
    if stop_request:
        if not Path(stop_request).is_absolute(): raise ValueError('absolute stop request path required')
        result['ProgramArguments']+=['--stop-request',str(stop_request)]
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('python','pod-id','deadline','key-file','report','log','output'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--stop-request')
    a=ap.parse_args()
    plan=make_plist(a.python,Path(__file__).with_name('runpod_stop_watchdog.py').resolve(),a.pod_id,a.deadline,a.key_file,a.report,a.log,a.stop_request)
    # Caller must separately authorize installing/loading this file with launchctl.
    with Path(a.output).open('xb') as output:plistlib.dump(plan,output)

if __name__=='__main__':main()
