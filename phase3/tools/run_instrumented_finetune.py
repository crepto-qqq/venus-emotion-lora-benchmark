#!/usr/bin/env python3
"""Explicit opt-in wrapper. Does not authorize paid work or replace bounded launcher."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import runpy
import sys
from pathlib import Path
from member3_common import build_finetune_command, canonical_sha256, read_json, sha256_file
from training_telemetry import Recorder, instrument_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--interface', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--expected-wrapper-sha256', required=True)
    parser.add_argument('--expected-telemetry-sha256', required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--expected-interface-sha256', required=True)
    parser.add_argument('--telemetry-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('training_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.execute:
        parser.error('requires --execute; invoke only through validated bounded launcher')
    config, inventory, interface = read_json(args.config), read_json(args.inventory), read_json(args.interface)
    if config.get('kind') == 'lora_formal_config':
        from run_lora_training import formal_telemetry_contract
        formal_telemetry_contract(config)
    bindings = [(args.expected_wrapper_sha256, sha256_file(Path(__file__))),
        (args.expected_telemetry_sha256, sha256_file(Path(__file__).with_name('training_telemetry.py'))),
        (args.expected_config_sha256, canonical_sha256(config, omit_key='config_sha256')),
        (args.expected_interface_sha256, sha256_file(args.interface))]
    if any(expected != actual for expected, actual in bindings):
        parser.error('execution identity changed after launcher planning')
    tail = args.training_arguments
    if tail and tail[0] == '--':
        tail = tail[1:]
    expected = build_finetune_command(config, interface)
    if tail != expected[2:]:
        parser.error('training arguments differ from resolved config/interface command')
    if '{' in config['output']['run_dir'] or '}' in config['output']['run_dir']:
        parser.error('wrapper requires a resolved output path')
    source = (Path(config['runtime']['working_directory']) / interface['entry_point']).resolve()
    if args.source.resolve() != source:
        parser.error('source path differs from config/interface')
    if interface.get('source_inventory_state') != 'verified' or interface.get('finetune_py_sha256') != args.expected_source_sha256:
        parser.error('source hash differs from verified interface')
    if sha256_file(args.source) != args.expected_source_sha256:
        parser.error('upstream source hash mismatch')
    if inventory.get('report_sha256') != canonical_sha256(inventory, omit_key='report_sha256'):
        parser.error('inventory hash mismatch')
    if config['lora']['inventory_sha256'] != inventory['report_sha256']:
        parser.error('config/inventory binding mismatch')
    if config['lora']['mode'] != 'lora' or config['lora']['modules_to_save'] is not None or config['lora']['bias'] != 'none':
        parser.error('only existing LoRA/no-bias/no-modules-to-save contract is supported')
    recorder = Recorder(args.telemetry_dir, dict(config_sha256=canonical_sha256(config, omit_key='config_sha256'),
        inventory_sha256=inventory['report_sha256'], source_sha256=args.expected_source_sha256,
        interface_sha256=sha256_file(args.interface),
        wrapper_sha256=sha256_file(Path(__file__)),
        telemetry_sha256=sha256_file(Path(__file__).with_name('training_telemetry.py'))))
    original_argv, original_path = sys.argv[:], sys.path[:]
    original_trainer = None
    try:
        import torch
        import transformers
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            recorder.fail('runtime wrapper requires exactly one visible CUDA device')
        original_trainer = transformers.Trainer
        recorder.summary['identity'].update(torch_version=torch.__version__, transformers_version=transformers.__version__,
            trainer_source_sha256=hashlib.sha256(inspect.getsource(original_trainer).encode()).hexdigest())
        transformers.Trainer = instrument_trainer(original_trainer, transformers.TrainerCallback,
            torch, recorder, inventory, config['lora']['rank'],
            track_updates=config.get('kind') == 'lora_formal_config')
        sys.argv = [str(args.source), *tail]
        sys.path.insert(0, str(args.source.resolve().parent))
        runpy.run_path(str(args.source.resolve()), run_name='__main__')
        if recorder.summary['state'] != 'training_completed':
            recorder.fail('source did not complete instrumented Trainer.train')
        recorder.summary['state'] = 'source_completed'
        recorder.save()
    except BaseException as error:
        recorder.summary.update(state='failed', failure_type=type(error).__name__)
        recorder.save()
        raise
    finally:
        if original_trainer is not None:
            transformers.Trainer = original_trainer
        sys.argv, sys.path[:] = original_argv, original_path
        recorder.close()


if __name__ == '__main__':
    main()
