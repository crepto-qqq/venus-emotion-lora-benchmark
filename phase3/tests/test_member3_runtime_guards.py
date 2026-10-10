import copy
import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from member3_common import ROOT, read_json, write_json, canonical_sha256, validate_resource_limits
from run_lora_training import check_data, run_bounded
from make_lora_pilot_subset import choose
from collect_lora_pilot_evidence import evidence_failures

class RuntimeGuards(unittest.TestCase):
    def test_timeout_stops_real_process(self):
        start=time.monotonic()
        code, timed_out=run_bounded([sys.executable,'-c','import time; time.sleep(30)'], ROOT, .1)
        self.assertEqual((code,timed_out),(124,True))
        self.assertLess(time.monotonic()-start, 10)

    def test_subset_required_and_tamper_rejected(self):
        c=read_json(ROOT/'phase3/configs/lora-pilot-v1.json')
        with tempfile.TemporaryDirectory() as directory:
            for split in ('train','validation'):
                rows=read_json(ROOT/c['data'][split+'_repo_path'])
                selected=choose(rows,c['pilot_subset'][split+'_per_class_task'],c['pilot_subset']['seed']+('-validation' if split=='validation' else ''))
                self.assertEqual(len(selected),24 if split=='train' else 96)
                path=Path(directory)/(split+'.json');write_json(path,selected);c['data'][split+'_path']=str(path)
            check_data(c,'pilot')
            write_json(Path(c['data']['train_path']),read_json(ROOT/c['data']['train_repo_path']))
            with self.assertRaises(ValueError): check_data(c,'pilot')

    def test_invalid_numeric_limits(self):
        limits=dict(allowed_gpu='H100', max_hourly_usd=1,total_budget_usd=2,max_wall_clock_minutes=3,max_attempts=1,oom_policy='stop_and_request')
        self.assertEqual(validate_resource_limits(limits),[])
        for value in (True,'2',float('nan'),float('inf'),-1):
            self.assertTrue(validate_resource_limits(dict(limits,total_budget_usd=value)))

    def test_evidence_rejects_failed_save_incomplete_eval_and_wrong_config(self):
        config={'resource_limits':dict(max_wall_clock_minutes=10,total_budget_usd=2,max_hourly_usd=2)}
        attempt=dict(execution_state='completed',returncode=0,config_sha256=canonical_sha256(config))
        attempt['attempt_record_sha256']=canonical_sha256(attempt)
        ev=dict(result='passed',missing_ids=[]);ev['report_sha256']=canonical_sha256(ev)
        facts=dict(finite_loss=True,checkpoint_saved=True,adapter_saved=True,reload_passed=True,clean_process_inference_passed=True,zero_supervised_tokens_observed=False,unexpected_trainable_parameters=False,adapter_sha256='a'*64,peak_vram_bytes=100,throughput_samples_per_second=1,elapsed_seconds=60,displayed_hourly_price_usd=1,calculated_cost_usd=.1)
        self.assertEqual(evidence_failures(config,attempt,ev,facts),[])
        for key in ('checkpoint_saved','adapter_saved'):
            self.assertTrue(evidence_failures(config,attempt,ev,dict(facts,**{key:False})))
        self.assertTrue(evidence_failures(config,attempt,dict(ev,result='incomplete'),facts))
        self.assertTrue(evidence_failures(config,dict(attempt,config_sha256='b'*64),ev,facts))

if __name__=='__main__': unittest.main()
