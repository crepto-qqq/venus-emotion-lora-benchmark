import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from member3_common import canonical_sha256, write_json, read_json
from run_formal_session import check_session, check_guard, commands, bounded, execute
from relay_formal_watchdog import relay, remote_path, ssh_command
from prepare_watchdog_launchagent import make_plist


def iso(epoch): return datetime.fromtimestamp(epoch,timezone.utc).isoformat()


class FormalSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        base=Path(__file__).resolve().parents[1]/'configs/lora-pilot-v1.json'
        self.config=read_json(base)
        self.config['kind']='lora_formal_config'
        self.config['training'].update(epochs=3,max_steps=180,save_strategy='steps',save_steps=60,save_total_limit=3)
        self.config['resource_limits']=dict(total_budget_usd=10,max_hourly_usd=4,
            max_wall_clock_minutes=60,max_attempts=1,allowed_gpu='fixture',oom_policy='stop_and_request')
        self.config['output']['run_dir']=str(self.root/'train'/'{attempt_id}')
        self.config['config_sha256']=canonical_sha256(self.config,omit_key='config_sha256')
        now=time.time()
        self.session=dict(pod_id='fixture',attempt_id='test',started_at_utc=iso(now-10),
            deadline=iso(now+600),max_session_usd=1,actual_hourly_usd=3,
            training_minutes=2,resource_ledger='/workspace/phase3/member3/formal/resource-ledger.json',
            config_sha256=self.config['config_sha256'])
        self.bind()
        self.paths={k:self.root/(k+'.json') for k in ('config','inventory','interface','review','guard')}
        self.paths.update(output=self.root/'evidence',data_dir=self.root/'data')
        write_json(self.paths['config'],self.config)
        inventory=dict(generated_at_utc=iso(now)); inventory['report_sha256']=canonical_sha256(inventory)
        write_json(self.paths['inventory'],inventory)
        for key in ('interface','review'): write_json(self.paths[key],{})
        self.guard=dict(kind='runpod_stop_watchdog',pod_id='fixture',deadline=self.session['deadline'],
            execution_state='armed',observed_state='RUNNING',heartbeat_utc=iso(now))
        write_json(self.paths['guard'],self.guard)

    def bind(self): self.session['session_sha256']=canonical_sha256(self.session,omit_key='session_sha256')
    def tearDown(self): self.temp.cleanup()

    def test_budget_bound_and_config_binding(self):
        check_session(self.session,self.config)
        self.session['max_session_usd']=0.01; self.bind()
        with self.assertRaisesRegex(ValueError,'whole-session'): check_session(self.session,self.config)

    def test_draft_resources_blocked(self):
        self.config['resource_limits']=None
        self.session['config_sha256']=canonical_sha256(self.config,omit_key='config_sha256'); self.bind()
        with self.assertRaises(ValueError): check_session(self.session,self.config)

    def test_missing_actual_pod_blocked_clearly(self):
        self.session['pod_id']=None; self.bind()
        with self.assertRaisesRegex(ValueError,'actual Pod ID'): check_session(self.session,self.config)

    def test_guard_stale_or_wrong_pod(self):
        check_guard(self.paths['guard'],self.session)
        self.guard['heartbeat_utc']=iso(time.time()-31); write_json(self.paths['guard'],self.guard)
        with self.assertRaisesRegex(ValueError,'stale'): check_guard(self.paths['guard'],self.session)
        self.guard['pod_id']='other'; write_json(self.paths['guard'],self.guard)
        with self.assertRaisesRegex(ValueError,'matching'): check_guard(self.paths['guard'],self.session)

    def test_only_three_checkpoints_and_fresh_initialization(self):
        _,jobs=commands(self.config,self.session,self.paths)
        self.assertEqual([n for n,c in jobs],['preflight','formal_gate','model_inputs','training',
            'validation_60','validation_120','validation_180','collect'])
        training=dict(jobs)['training']
        self.assertIn('--telemetry',training); self.assertNotIn('--resume_from_checkpoint',training)
        self.assertTrue(all('--pilot-review' in dict(jobs)['validation_'+str(step)] for step in (60,120,180)))

    def test_fail_stops_remaining_jobs_and_requests_pod_stop(self):
        calls=[]
        def runner(command,*args): calls.append(command); return 2
        with self.assertRaisesRegex(RuntimeError,'preflight'): execute(self.config,self.session,self.paths,runner)
        self.assertEqual(len(calls),1)
        record=read_json(self.paths['output']/'session-status.json')
        self.assertEqual(record['state'],'failed'); self.assertTrue(record['stop_requested'])
        self.assertFalse(record['pod_stop_confirmed'])

    def test_complete_fixture_does_not_claim_final_evaluation(self):
        result=execute(self.config,self.session,self.paths,lambda *a:0)
        self.assertEqual(len(result['jobs']),8); self.assertTrue(result['stop_requested'])
        self.assertFalse(result['formal_Eval80_completed']); self.assertFalse(result['final_acceptance'])
        with self.assertRaisesRegex(ValueError,'fresh'): execute(self.config,self.session,self.paths,lambda *a:0)

    def test_input_tamper_stops_next_job(self):
        def runner(*args): self.paths['review'].write_text('{} '); return 0
        with self.assertRaisesRegex(ValueError,'changed'): execute(self.config,self.session,self.paths,runner)
        self.assertEqual(len(read_json(self.paths['output']/'session-status.json')['jobs']),1)

    def test_real_subprocess_timeout(self):
        start=time.monotonic()
        with self.assertRaises(TimeoutError):
            bounded([sys.executable,'-c','import time; time.sleep(30)'],self.root/'sleep.log',
                time.time()+0.2,time.monotonic()+0.2,lambda:None)
        self.assertLess(time.monotonic()-start,4)

    def test_actual_nested_launcher_gets_sigterm(self):
        # Real separately grouped child must disappear when outer launcher is terminated.
        marker=self.root/'child.pid'; tools=str(Path(__file__).resolve().parents[1]/'tools')
        child='import os,time; from pathlib import Path; Path('+repr(str(marker))+').write_text(str(os.getpid())); time.sleep(30)'
        script='import sys; sys.path.insert(0,'+repr(tools)+'); from run_lora_training import run_bounded; run_bounded([sys.executable,"-c",'+repr(child)+'],".",30)'
        process=subprocess.Popen([sys.executable,'-c',script])
        try:
            end=time.monotonic()+5
            while not marker.exists() and time.monotonic()<end: time.sleep(.05)
            self.assertTrue(marker.exists())
            pid=int(marker.read_text()); process.terminate(); process.wait(timeout=8)
            with self.assertRaises(ProcessLookupError): os.kill(pid,0)
        finally:
            if process.poll() is None: process.kill(); process.wait()

    def test_relay_completion_requests_local_stop(self):
        request=self.root/'stop'; status=dict(pod_id='fixture',session_sha256=self.session['session_sha256'],stop_requested=True)
        result=relay(self.session,[],self.paths['guard'],'/workspace/phase3/member3/formal/guard.json',
            '/workspace/phase3/member3/formal/status.json',request,self.root/'relay.json',
            transfer=lambda *a:status,sleep=lambda _:None)
        self.assertTrue(request.exists()); self.assertEqual(result['state'],'early_stop_requested')

    def test_relay_transport_failure_requests_stop(self):
        def fail(*args): raise OSError('fixture disconnected')
        request=self.root/'stop'
        with self.assertRaises(OSError): relay(self.session,[],self.paths['guard'],
            '/workspace/phase3/member3/formal/guard.json','/workspace/phase3/member3/formal/status.json',
            request,self.root/'relay.json',transfer=fail)
        self.assertTrue(request.exists())

    def test_relay_paths_and_launchagent_stop_file(self):
        with self.assertRaises(ValueError): remote_path('/workspace/phase3/envs/erase')
        with self.assertRaises(ValueError): ssh_command('host; injected',22,'key')
        result=make_plist('/python','/watchdog','fixture',self.session['deadline'],'/key','/report','/log','/stop')
        self.assertEqual(result['ProgramArguments'][-2:],['--stop-request','/stop'])

    def test_watchdog_early_request_calls_stop_and_confirms(self):
        import runpod_stop_watchdog as watchdog
        key=self.root/'key'; key.write_text('fixture-not-a-real-key'); key.chmod(0o600)
        stop=self.root/'stop'; stop.touch(); report=self.root/'watchdog.json'
        argv=['watchdog','--pod-id','fixture','--deadline',self.session['deadline'],
              '--key-file',str(key),'--report',str(report),'--stop-request',str(stop),'--execute']
        with patch.object(sys,'argv',argv), patch.object(watchdog,'pod_state',return_value='RUNNING'), \
                patch.object(watchdog,'stop_and_confirm',return_value=True) as stopping:
            with self.assertRaises(SystemExit) as result: watchdog.main()
        self.assertEqual(result.exception.code,0); stopping.assert_called_once()
        self.assertEqual(read_json(report)['execution_state'],'stop_state_observed')

    def test_watchdog_unconfirmed_stop_is_not_success(self):
        import runpod_stop_watchdog as watchdog
        key=self.root/'key'; key.write_text('fixture'); key.chmod(0o600)
        stop=self.root/'stop'; stop.touch(); report=self.root/'watchdog.json'
        argv=['watchdog','--pod-id','fixture','--deadline',self.session['deadline'],
              '--key-file',str(key),'--report',str(report),'--stop-request',str(stop),'--execute']
        with patch.object(sys,'argv',argv), patch.object(watchdog,'pod_state',return_value='RUNNING'), \
                patch.object(watchdog,'stop_and_confirm',return_value=False):
            with self.assertRaises(SystemExit) as result: watchdog.main()
        self.assertEqual(result.exception.code,2)
        self.assertEqual(read_json(report)['execution_state'],'stop_unconfirmed')


if __name__=='__main__': unittest.main()
