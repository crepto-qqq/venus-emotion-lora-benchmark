"""Offline formal completion evidence tests; no GPU or cloud success is claimed."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_training_telemetry_launcher as fixtures
from member3_common import write_json, sha256_file, ROOT
import run_lora_training as launcher

class FormalEvidenceTests(unittest.TestCase):
    plan = fixtures.LauncherTelemetryTests.plan
    report = fixtures.LauncherTelemetryTests.report
    save_report = fixtures.LauncherTelemetryTests.save_report
    tearDown = fixtures.LauncherTelemetryTests.tearDown
    def setUp(self):
        fixtures.LauncherTelemetryTests.setUp(self)
        self.config['kind'] = 'lora_formal_config'
        self.config['training'].update(epochs=3, max_steps=180, save_steps=60, save_total_limit=3)

    def formal_fixture(self):
        _, identity = self.plan()
        directory, report = self.report(identity)
        report.update(dataset_rows_checked=480, microbatches_checked=1440,
                      samples_observed=1440, optimizer_steps_observed=180)
        write_json(directory / 'trainable-parameters.json', {'parameters':[{'name':'language.lora_B.default.weight'}]})
        write_json(directory / 'parameter-updates.json', {'changed_parameter_tensors':1,
            'parameters':[{'name':'language.lora_B.default.weight','before_sha256':'0'*64,'after_sha256':'1'*64}]})
        report['parameter_updates_sha256'] = sha256_file(directory / 'parameter-updates.json')
        for step in [60,120,180]:
            p = self.root / 'run' / f'checkpoint-{step}'
            write_json(p / 'trainer_state.json', {'global_step':step})
            write_json(p / 'adapter_config.json', {'peft_type':'LORA','r':8})
            (p / 'adapter_model.bin').write_bytes(b'fixture weights; no model loaded')
        self.save_report(directory, report)
        return directory, report, identity

    def verify(self, directory, identity):
        return launcher.verify_training_telemetry(directory, self.config, identity, self.inventory_path)

    def test_formal_complete_fixture_binds_all_checkpoints(self):
        d, r, i = self.formal_fixture()
        result = self.verify(d,i)
        self.assertEqual([x['step'] for x in result['formal']['checkpoints']], [60,120,180])
        self.assertEqual(result['formal']['changed_parameter_tensors'], 1)

    def test_formal_wrong_counts_rejected(self):
        d,r,i = self.formal_fixture()
        for key,value in [('dataset_rows_checked',24),('microbatches_checked',1439),('samples_observed',1439),('optimizer_steps_observed',179)]:
            broken=dict(r); broken[key]=value; self.save_report(d,broken)
            with self.assertRaises(ValueError): self.verify(d,i)

    def test_formal_missing_checkpoint_rejected(self):
        d,r,i = self.formal_fixture()
        (self.root/'run/checkpoint-120/adapter_model.bin').unlink()
        with self.assertRaisesRegex(ValueError,'adapter'): self.verify(d,i)

    def test_formal_wrong_checkpoint_step_rejected(self):
        d,r,i = self.formal_fixture()
        write_json(self.root/'run/checkpoint-60/trainer_state.json',{'global_step':59})
        with self.assertRaisesRegex(ValueError,'step'): self.verify(d,i)

    def test_formal_missing_or_tampered_updates_rejected(self):
        d,r,i = self.formal_fixture()
        (d/'parameter-updates.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash'): self.verify(d,i)

    def test_formal_unchanged_adapter_rejected(self):
        d,r,i = self.formal_fixture()
        write_json(d/'parameter-updates.json',{'changed_parameter_tensors':0,'parameters':[
            {'name':'language.lora_B.default.weight','before_sha256':'0'*64,'after_sha256':'0'*64}]})
        r['parameter_updates_sha256']=sha256_file(d/'parameter-updates.json'); self.save_report(d,r)
        with self.assertRaisesRegex(ValueError,'changes'): self.verify(d,i)

    def test_formal_contract_rejects_partial_epoch_and_checkpoint_rotation(self):
        self.assertEqual(launcher.formal_telemetry_contract(self.config)['checkpoint_steps'],[60,120,180])
        for key,value in [('epochs',1.5),('max_steps',30),('save_steps',10),('save_total_limit',2),('micro_batch_size',2)]:
            broken=copy.deepcopy(self.config);broken['training'][key]=value
            with self.assertRaises(ValueError):launcher.formal_telemetry_contract(broken)

    def test_formal_requires_telemetry_before_validator_or_launch(self):
        write_json(self.config_path,self.config)
        args=['run_lora_training.py','--mode','formal','--config',str(self.config_path),
              '--inventory',str(self.inventory_path),'--interface',str(self.interface_path),
              '--attempt-id','formal-fixture','--attempt-record',str(self.root/'formal-attempt.json')]
        with patch.object(sys,'argv',args),patch.object(launcher.subprocess,'run') as validator:
            with self.assertRaisesRegex(SystemExit,'requires --telemetry'):launcher.main()
            validator.assert_not_called()
        self.assertFalse((self.root/'formal-attempt.json').exists())

    def test_formal_dry_run_does_not_execute_or_reserve(self):
        write_json(self.config_path,self.config)
        args=['run_lora_training.py','--mode','formal','--config',str(self.config_path),
              '--inventory',str(self.inventory_path),'--interface',str(self.interface_path),
              '--attempt-id','formal-fixture','--attempt-record',str(self.root/'formal-attempt.json'),'--telemetry']
        with patch.object(sys,'argv',args), patch.object(launcher.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as validator, \
             patch.object(launcher,'check_data') as data, patch('member3_budget.reserve') as reserve, \
             patch.object(launcher,'run_bounded') as run:
            launcher.main()
            self.assertIn('formal',validator.call_args.args[0])
            data.assert_called_once_with(self.config,'formal')
            reserve.assert_not_called();run.assert_not_called()
        self.assertFalse((self.root/'run').exists())
        record=json.loads((self.root/'formal-attempt.json').read_text())
        self.assertEqual(record['execution_state'],'planned')
        self.assertEqual(record['finetune_command'][record['finetune_command'].index('--max_steps')+1],'180')

if __name__=='__main__':unittest.main()
