"""Offline launcher integration and fail-closed tests. No cloud/model access."""
import copy
import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_lora_training as launcher
from member3_common import ROOT, build_finetune_command, canonical_sha256, read_json, write_json


class LauncherTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = read_json(ROOT / 'phase3/configs/lora-pilot-v1.json')
        self.config['candidate_id'] = 'lr-5e-5'
        self.config['output']['run_dir'] = str(self.root / 'run')
        self.config['training']['learning_rate'] = 5e-5
        self.config['lora'].update(target_modules=['c_attn'], inventory_sha256='a'*64)
        self.config['resource_limits'] = dict(allowed_gpu='fixture', max_hourly_usd=1,
            total_budget_usd=1, max_wall_clock_minutes=10, max_attempts=3, oom_policy='stop_and_request')
        self.interface = read_json(ROOT / 'phase3/configs/qwen-training-interface-v1.json')
        self.interface.update(source_inventory_state='verified', finetune_py_sha256='b'*64)
        self.interface_path = self.root / 'interface.json'
        self.inventory_path = self.root / 'inventory.json'
        self.config_path = self.root / 'config.json'
        write_json(self.interface_path, self.interface)
        write_json(self.inventory_path, {'report_sha256':'a'*64})
        write_json(self.config_path, self.config)

    def tearDown(self):
        self.temp.cleanup()

    def plan(self):
        return launcher.instrumented_command(self.config, self.interface, self.inventory_path, self.interface_path)

    def report(self, identity):
        directory = Path(identity['directory'])
        directory.mkdir(parents=True)
        for name in ('events.jsonl', 'trainable-parameters.json'):
            (directory / name).write_text('{}\n')
        report = dict(identity=dict(config_sha256=canonical_sha256(self.config, omit_key='config_sha256'),
            inventory_sha256='a'*64, source_sha256='b'*64,
            interface_sha256=identity['interface_sha256'], wrapper_sha256=identity['wrapper_sha256'],
            telemetry_sha256=identity['telemetry_sha256']), state='source_completed',
            finite_loss=True, finite_gradients=True, zero_supervised_tokens_observed=False,
            unexpected_trainable_parameters=False, optimizer_steps_observed=30,
            dataset_rows_checked=24, microbatches_checked=240, samples_observed=240,
            training_peak_allocated_bytes=1234, training_peak_reserved_bytes=5678,
            training_elapsed_seconds=2, throughput_samples_per_second=120)
        return directory, report

    def save_report(self, directory, report):
        report['report_sha256'] = canonical_sha256(report, omit_key='report_sha256')
        write_json(directory / 'summary.json', report)

    def test_wrapper_forwards_exact_original_training_arguments(self):
        command, identity = self.plan()
        self.assertEqual(command[command.index('--')+1:], build_finetune_command(self.config, self.interface)[2:])
        self.assertEqual(identity['directory'], str(self.root / 'run/telemetry'))
        self.assertFalse((self.root / 'run').exists())

    def test_pending_interface_or_unresolved_output_rejected(self):
        with self.assertRaises(ValueError):
            launcher.instrumented_command(self.config, dict(self.interface, source_inventory_state='pending'), self.inventory_path, self.interface_path)
        self.config['output']['run_dir'] += '/{attempt_id}'
        with self.assertRaises(ValueError): self.plan()

    def test_report_identity_must_match_and_partial_fails(self):
        _, identity = self.plan()
        directory, report = self.report(identity)
        self.save_report(directory, report)
        self.assertEqual(launcher.verify_training_telemetry(directory, self.config, identity, self.inventory_path)['state'], 'source_completed')
        for key, value in [('state','training'), ('finite_gradients',None),
                           ('optimizer_steps_observed',29), ('training_peak_allocated_bytes',None)]:
            broken = copy.deepcopy(report); broken[key] = value
            self.save_report(directory, broken)
            with self.assertRaises(ValueError):
                launcher.verify_training_telemetry(directory, self.config, identity, self.inventory_path)
        report['identity']['config_sha256'] = 'c'*64
        self.save_report(directory, report)
        with self.assertRaisesRegex(ValueError,'identity'):
            launcher.verify_training_telemetry(directory, self.config, identity, self.inventory_path)

    def test_tampered_report_hash_rejected(self):
        _, identity = self.plan(); directory, report = self.report(identity)
        self.save_report(directory, report)
        report['finite_loss'] = False
        write_json(directory / 'summary.json', report)
        with self.assertRaisesRegex(ValueError,'hash'):
            launcher.verify_training_telemetry(directory, self.config, identity, self.inventory_path)

    def invoke(self, execute=False, child=None):
        args = ['run_lora_training.py','--mode','pilot','--config',str(self.config_path),
            '--interface',str(self.interface_path),'--inventory',str(self.inventory_path),
            '--candidate-id','lr-5e-5','--attempt-id','fixture-attempt',
            '--attempt-record',str(self.root / 'attempt.json'),'--telemetry']
        if execute:
            args += ['--execute','--resource-ledger','/workspace/phase3/member3/pilot/resource-ledger.json','--attempt-minutes','2']
        with patch.object(sys,'argv',args), patch.object(launcher.subprocess,'run',return_value=SimpleNamespace(returncode=0)), \
             patch.object(launcher,'check_data'), patch('member3_budget.reserve',return_value={'seconds':120}) as reserve, \
             patch.object(launcher,'run_bounded',side_effect=child) as bounded:
            try: launcher.main()
            except SystemExit as error: code = error.code
            else: code = 0
            return code, reserve, bounded

    def test_dry_run_never_reserves_or_launches_or_creates_training_directory(self):
        code, reserve, bounded = self.invoke()
        self.assertEqual(code, 0); reserve.assert_not_called(); bounded.assert_not_called()
        self.assertFalse((self.root / 'run').exists())
        self.assertEqual(read_json(self.root / 'attempt.json')['execution_state'], 'planned')

    def test_missing_telemetry_turns_zero_child_return_into_failed_attempt(self):
        code, reserve, bounded = self.invoke(True, lambda *args: (0,False))
        self.assertEqual(code, 2); reserve.assert_called_once(); bounded.assert_called_once()
        self.assertEqual(read_json(self.root / 'attempt.json')['execution_state'], 'failed')

    def test_execute_resolves_config_before_child_and_records_verified_summary(self):
        def child(command, cwd, seconds):
            config = read_json(self.root / 'run/telemetry-config.json')
            self.assertEqual(config, self.config)
            _, identity = self.plan()
            directory, report = self.report(identity); self.save_report(directory, report)
            return 0, False
        code, reserve, bounded = self.invoke(True, child)
        self.assertEqual(code, 0); reserve.assert_called_once(); bounded.assert_called_once()
        attempt = read_json(self.root / 'attempt.json')
        self.assertEqual(attempt['telemetry_verification']['state'], 'source_completed')
        self.assertEqual(attempt['config_sha256'], canonical_sha256(self.config, omit_key='config_sha256'))

    def runtime_command_fixture(self):
        self.config['runtime']['working_directory'] = str(self.root)
        (self.root / 'finetune.py').write_text("raise RuntimeError('must never execute fixture source')\n")
        from member3_common import sha256_file
        self.interface['finetune_py_sha256'] = sha256_file(self.root / 'finetune.py')
        write_json(self.interface_path, self.interface)
        inventory = {'matched_module_names':['c_attn']}
        inventory['report_sha256'] = canonical_sha256(inventory)
        write_json(self.inventory_path, inventory)
        self.config['lora']['inventory_sha256'] = inventory['report_sha256']
        command, identity = self.plan()
        Path(identity['resolved_config_path']).parent.mkdir(parents=True)
        write_json(Path(identity['resolved_config_path']), self.config)
        command[0] = sys.executable
        return command, identity

    def test_real_wrapper_process_rejects_training_flag_drift_before_importing_model(self):
        command, identity = self.runtime_command_fixture()
        command[command.index('--learning_rate')+1] = '0.5'
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('training arguments differ', result.stderr)
        self.assertFalse(Path(identity['directory']).exists())

    def test_real_wrapper_process_rejects_tool_hash_drift(self):
        command, identity = self.runtime_command_fixture()
        command[command.index('--expected-wrapper-sha256')+1] = '0'*64
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('execution identity changed', result.stderr)
        self.assertFalse(Path(identity['directory']).exists())


if __name__ == '__main__': unittest.main()
