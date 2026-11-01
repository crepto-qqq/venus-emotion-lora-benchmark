import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from member3_budget import reserve
from runpod_stop_watchdog import parse_deadline, stop_and_confirm

class BudgetWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.limits=dict(allowed_gpu='test-gpu',max_hourly_usd=6,total_budget_usd=1,max_wall_clock_minutes=10,max_attempts=2,oom_policy='stop_and_request')
    def test_retries_share_budget_and_attempt_limit(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'ledger.json'
            reserve(p,'pilot',self.limits,'first','a'*64,4)
            reserve(p,'pilot',self.limits,'second','b'*64,4)
            with self.assertRaisesRegex(ValueError,'attempts'):reserve(p,'pilot',self.limits,'third','c'*64,1)
    def test_total_budget_and_time_cannot_reset_with_config_change(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'ledger.json'
            limits=dict(self.limits,max_attempts=3)
            reserve(p,'pilot',limits,'first','a'*64,6)
            with self.assertRaisesRegex(ValueError,'budget'):reserve(p,'pilot',limits,'second','b'*64,5)
            with self.assertRaisesRegex(ValueError,'scope/limits'):reserve(p,'formal',limits,'second','b'*64,1)
            with self.assertRaisesRegex(ValueError,'scope/limits'):reserve(p,'pilot',dict(limits,total_budget_usd=2),'second','b'*64,1)
    def test_duplicate_attempt_and_corrupt_ledger_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'ledger.json';reserve(p,'pilot',self.limits,'first','a'*64,1)
            with self.assertRaisesRegex(ValueError,'already'):reserve(p,'pilot',self.limits,'first','a'*64,1)
            data=json.loads(p.read_text());data['attempts']=[];p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'hash'):reserve(p,'pilot',self.limits,'second','a'*64,1)
    def test_stop_requires_verified_state_and_never_deletes(self):
        state={'value':'RUNNING'};calls=[]
        def transport(method,path,key):
            calls.append((method,path))
            if method=='POST':state['value']='EXITED';return None
            return [{'id':'test-pod','desiredStatus':state['value']}]
        self.assertTrue(stop_and_confirm('test-pod','fake',transport=transport,sleep=lambda _:None))
        self.assertEqual(calls,[('GET',''),('POST','/test-pod/stop'),('GET','')])
    def test_missing_pod_or_failed_api_not_reported_as_stopped(self):
        self.assertFalse(stop_and_confirm('missing','fake',transport=lambda *args:[],sleep=lambda _:None,attempts=2))
        def fail(*args):raise OSError('offline')
        self.assertFalse(stop_and_confirm('test','fake',transport=fail,sleep=lambda _:None,attempts=2))
    def test_deadline_requires_explicit_timezone(self):
        with self.assertRaises(ValueError):parse_deadline('2026-10-05T10:00:00')
        self.assertEqual(parse_deadline('2026-10-05T10:00:00Z'),parse_deadline('2026-10-05T18:00:00+08:00'))
    def test_prearmed_watchdog_wait_does_not_extend_deadline(self):
        from unittest.mock import patch
        import runpod_stop_watchdog as watchdog
        now=[0.0]; reports=[]
        with tempfile.TemporaryDirectory() as td:
            key=Path(td)/'key';key.write_text('fake');key.chmod(0o600)
            args=['watchdog','--pod-id','fixture','--deadline','1970-01-01T00:00:10Z',
                '--key-file',str(key),'--report',str(Path(td)/'report'), '--wait-for-start','--execute']
            with patch.object(sys,'argv',args), patch.object(watchdog,'pod_state',side_effect=['EXITED','RUNNING']), \
                 patch.object(watchdog.time,'time',side_effect=lambda:now[0]), \
                 patch.object(watchdog.time,'monotonic',side_effect=lambda:now[0]), \
                 patch.object(watchdog.time,'sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)), \
                 patch.object(watchdog,'write_json',side_effect=lambda path,data:reports.append(copy.deepcopy(data))), \
                 patch.object(watchdog,'stop_and_confirm',return_value=True) as stop:
                with self.assertRaises(SystemExit) as exit:
                    watchdog.main()
                self.assertEqual(exit.exception.code,0)
                stop.assert_called_once()
            self.assertEqual(now[0],10)
            self.assertEqual(reports[0]['execution_state'],'waiting_for_start')
            self.assertEqual(reports[-1]['execution_state'],'stop_state_observed')
if __name__=='__main__':unittest.main()

class LaunchAgentPlanTests(unittest.TestCase):
    def test_plan_preserves_paths_with_spaces_and_retries_only_failure(self):
        from prepare_watchdog_launchagent import make_plist
        p=make_plist('/tmp/python','/tmp/watch dog.py','fake-pod','2026-10-05T10:00:00Z','/tmp/key file','/tmp/report','/tmp/log')
        self.assertIn('/tmp/key file',p['ProgramArguments'])
        self.assertEqual(p['ProgramArguments'][:2],['/usr/bin/caffeinate','-is'])
        self.assertEqual(p['KeepAlive'],{'SuccessfulExit':False})
        self.assertNotIn('start',p['ProgramArguments'])
        self.assertNotIn('delete',p['ProgramArguments'])
