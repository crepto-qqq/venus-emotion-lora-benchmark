#!/usr/bin/env python3
"""Offline checkpoint job planner and strict collector; never launches GPU jobs."""
import argparse
import math
from pathlib import Path
from member3_common import ROOT, canonical_sha256, read_json, sha256_file, write_json, task_type
from run_lora_training import formal_telemetry_contract
from generate_lora_validation import adapter_identity, validation_rows
from evaluate_lora_pilot import ALLOWED, parse_guidance


def plan(config, config_path, training_dir, output_dir, interface, inventory, review):
    contract = formal_telemetry_contract(config)
    if any(x in str(training_dir) for x in ('{', '}')):
        raise ValueError('training directory must be resolved')
    jobs = []
    for step in contract['checkpoint_steps']:
        directory = output_dir / f'checkpoint-{step}'
        command = [config['runtime']['python'], str(ROOT / 'phase3/tools/generate_lora_validation.py'),
            '--config', str(config_path), '--adapter', str(training_dir / f'checkpoint-{step}'),
            '--validation', str(ROOT / config['data']['validation_repo_path']),
            '--output-dir', str(directory), '--max-new-tokens', '512', '--interface', str(interface),
            '--inventory', str(inventory), '--pilot-review', str(review), '--execute']
        jobs.append(dict(step=step, command=command, output_dir=str(directory)))
    return dict(kind='formal_checkpoint_validation_plan', execution_state='planned_not_executed',
        config_sha256=canonical_sha256(config, omit_key='config_sha256'), jobs=jobs,
        sequential_clean_processes=True, provisioning=False,
        execution_requirements=['completed verified formal training', 'approved exact-config decision',
            'fresh runtime preflight', 'whole-session budget and deadline with active external Pod watchdog'],
        warning='Commands are plans only. This tool has no GPU execution mode or Pod stop capability.')


def checked_record(path):
    record = read_json(path)
    key = 'attempt_record_sha256' if 'attempt_record_sha256' in record else 'report_sha256'
    if record.get(key) != canonical_sha256(record, omit_key=key):
        raise ValueError('record hash mismatch: ' + str(path))
    return record


def collect(config, training_dir, validation_dir, attempt_path):
    contract = formal_telemetry_contract(config)
    attempt = checked_record(attempt_path)
    config_hash = canonical_sha256(config, omit_key='config_sha256')
    if (attempt.get('mode') != 'formal' or attempt.get('execution_state') != 'completed'
            or attempt.get('returncode') != 0 or attempt.get('config_sha256') != config_hash):
        raise ValueError('completed matching formal training attempt required')
    verified = attempt.get('telemetry_verification', {})
    if verified.get('state') != 'source_completed':
        raise ValueError('verified formal telemetry required')
    telemetry = training_dir / 'telemetry'
    for name,key in [('summary.json','summary_sha256'),('events.jsonl','events_sha256'),
                     ('trainable-parameters.json','trainable_parameters_sha256')]:
        if sha256_file(telemetry/name) != verified.get(key):
            raise ValueError('training telemetry file differs from completed attempt')
    summary = checked_record(telemetry/'summary.json')
    if (summary.get('state') != 'source_completed'
            or summary.get('identity', {}).get('config_sha256') != attempt.get('resolved_config_sha256')
            or any(summary.get(k) != contract[k] for k in ('dataset_rows_checked','microbatches_checked','samples_observed','optimizer_steps_observed'))):
        raise ValueError('training telemetry identity or completion mismatch')
    if sha256_file(telemetry/'parameter-updates.json') != verified.get('formal', {}).get('parameter_updates_sha256'):
        raise ValueError('training parameter update evidence changed')
    bound = {x['step']: x for x in verified.get('formal', {}).get('checkpoints', [])}
    if sorted(bound) != contract['checkpoint_steps']:
        raise ValueError('training attempt checkpoint coverage mismatch')
    validation_path = ROOT / config['data']['validation_repo_path']
    gold_rows = validation_rows(validation_path)
    gold = {x['id']: x for x in gold_rows}
    candidates = []
    for step in contract['checkpoint_steps']:
        adapter = training_dir / f'checkpoint-{step}'
        identities = adapter_identity(adapter)
        weight_hashes = [h for n,h in identities.items() if n != 'adapter_config.json']
        if (weight_hashes != [bound[step]['weights_sha256']]
                or identities['adapter_config.json'] != bound[step]['config_sha256']
                or sha256_file(adapter/'trainer_state.json') != bound[step]['trainer_state_sha256']):
            raise ValueError('checkpoint differs from verified training attempt')
        directory = validation_dir / f'checkpoint-{step}'
        run = checked_record(directory / 'run.json')
        if (run.get('execution_state') != 'completed' or run.get('config_sha256') != config_hash
                or run.get('validation_sha256') != sha256_file(validation_path)
                or run.get('adapter_files') != identities
                or run.get('adapter_bundle_sha256') != canonical_sha256(identities)
                or run.get('generation') != {'do_sample': False} or run.get('max_new_tokens') != 512
                or run.get('sample_count') != 96 or run.get('prediction_count') != 96):
            raise ValueError('validation execution identity/coverage mismatch')
        predictions = [__import__('json').loads(l) for l in (directory/'predictions.jsonl').read_text().splitlines() if l.strip()]
        if len(predictions) != 96 or {x['id'] for x in predictions} != set(gold):
            raise ValueError('validation missing, duplicate or unexpected IDs')
        weighted = tokens = correct = guide_valid = 0
        classes = sorted(ALLOWED)
        confusion = {label:{p:0 for p in classes + ['invalid']} for label in classes}
        for row in predictions:
            loss, count = row.get('loss'), row.get('supervised_tokens')
            if (isinstance(loss, bool) or not isinstance(loss, (int,float)) or not math.isfinite(loss)
                    or isinstance(count, bool) or not isinstance(count, int) or count <= 0):
                raise ValueError('every prediction requires finite loss and positive token count')
            weighted += loss * count; tokens += count
            text = row.get('prediction')
            if not isinstance(text, str): raise ValueError('prediction must be text')
            if task_type(row['id']) == 'classification':
                label = gold[row['id']]['conversations'][1]['value'].strip().lower()
                prediction = text.strip().lower()
                confusion[label][prediction if prediction in ALLOWED else 'invalid'] += 1
                correct += prediction == label
            else:
                guide_valid += parse_guidance(text)[0]
        score = weighted / tokens
        if not math.isfinite(score): raise ValueError('aggregate validation loss is nonfinite')
        candidates.append(dict(step=step, validation_token_weighted_loss=score,
            classification_correct=correct, classification_total=80, classification_exact_match=correct/80,
            guidance_format_valid=guide_valid, guidance_total=16, guidance_format_valid_rate=guide_valid/16,
            confusion_matrix=confusion, per_class={label:dict(correct=confusion[label][label],total=sum(confusion[label].values())) for label in classes},
            adapter_files=identities, run_sha256=sha256_file(directory/'run.json'),
            predictions_sha256=sha256_file(directory/'predictions.jsonl')))
    rule = config.get('planning_metadata', {}).get('checkpoint_selection', {})
    if (rule.get('primary') != 'lowest_validation_token_weighted_loss'
            or rule.get('tie_breakers') != ['highest_classification_exact_match','highest_guidance_format_valid_rate','earliest_checkpoint_step']):
        raise ValueError('checkpoint selection rule missing or changed')
    ranked = sorted(candidates, key=lambda r:(r['validation_token_weighted_loss'],
        -r['classification_exact_match'], -r['guidance_format_valid_rate'], r['step']))
    report = dict(kind='formal_checkpoint_validation_comparison', state='validation_comparison_completed',
        config_sha256=config_hash, training_attempt_sha256=sha256_file(attempt_path),
        validation_sha256=sha256_file(validation_path), selection_rule=rule,
        provisional_selected_step=ranked[0]['step'], ranked_steps=[x['step'] for x in ranked],
        candidates=candidates, guidance_content_review='pending', final_acceptance=False,
        formal_Eval80_completed=False, frozen_comparison=False)
    report['report_sha256'] = canonical_sha256(report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['plan','collect'], required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--training-dir', type=Path, required=True)
    ap.add_argument('--validation-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--interface', type=Path)
    ap.add_argument('--inventory', type=Path)
    ap.add_argument('--pilot-review', type=Path)
    ap.add_argument('--attempt-record', type=Path)
    args = ap.parse_args()
    if args.output.exists(): raise ValueError('refusing to overwrite existing plan/evidence')
    config = read_json(args.config)
    if args.mode == 'plan':
        if not all((args.interface,args.inventory,args.pilot_review)):
            ap.error('plan requires interface, inventory and pilot-review paths')
        result = plan(config,args.config,args.training_dir,args.validation_dir,args.interface,args.inventory,args.pilot_review)
    else:
        if not args.attempt_record: ap.error('collect requires completed training attempt')
        result = collect(config,args.training_dir,args.validation_dir,args.attempt_record)
    write_json(args.output,result)


if __name__ == '__main__': main()
