#!/usr/bin/env python3
"""Validated launcher for pilot or formal LoRA training. Dry-run unless --execute is explicit."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, signal, re
from pathlib import Path
from member3_common import ROOT, build_finetune_command, canonical_sha256, command_text, read_json, write_json, sha256_file
from common import utc_now

def run_bounded(command, cwd, seconds):
    """Bound the process group, not Pod billing. No cloud resource operations."""
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return _run_bounded(command, cwd, seconds)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _run_bounded(command, cwd, seconds):
    # Forward outer session termination to the separately grouped Trainer process.
    with subprocess.Popen(command, cwd=cwd, start_new_session=True) as process:
        try:
            return process.wait(timeout=seconds), False
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124, True


def check_data(config, mode):
    from make_lora_pilot_subset import choose
    for split, key in (("train", "train"), ("validation", "validation")):
        source = read_json(ROOT / config["data"][key + "_repo_path"])
        expected = source
        if mode == "pilot":
            spec = config["pilot_subset"]
            expected = choose(source, spec[split + "_per_class_task"], spec["seed"] + ("-validation" if split == "validation" else ""))
        actual = read_json(Path(config["data"][key + "_path"]))
        if actual != expected:
            raise ValueError(f"{split} runtime data does not match frozen {'pilot subset' if mode == 'pilot' else 'release'}")


def formal_telemetry_contract(config):
    """Bounded, full-release, whole-epoch route only; no resource/review authorization."""
    t = config['training']
    if config['data']['train_repo_path'] != 'phase3/releases/bridge-train-v1/bridge-train-v1.train.json':
        raise ValueError('formal telemetry requires frozen BridgeTrain-v1 train path')
    rows = len(read_json(ROOT / config['data']['train_repo_path']))
    epochs = t['epochs']
    if config.get('kind') != 'lora_formal_config' or rows != 480:
        raise ValueError('formal telemetry requires the 480-row formal release')
    if isinstance(epochs, bool) or epochs not in (1, 2, 3):
        raise ValueError('formal telemetry requires 1 to 3 whole epochs')
    if t['micro_batch_size'] != 1 or t['gradient_accumulation_steps'] != 8:
        raise ValueError('formal telemetry requires micro batch 1 and accumulation 8')
    steps = rows // 8 * int(epochs)
    if t['max_steps'] != steps or t['save_strategy'] != 'steps' or t['save_steps'] != rows // 8:
        raise ValueError('formal steps/checkpoints must match whole release epochs')
    if t['save_total_limit'] < epochs:
        raise ValueError('formal checkpoints must all be retained')
    return dict(dataset_rows_checked=rows, microbatches_checked=rows * int(epochs),
        samples_observed=rows * int(epochs), optimizer_steps_observed=steps,
        checkpoint_steps=list(range(rows // 8, steps + 1, rows // 8)))


def instrumented_command(config, interface, inventory_path, interface_path):
    """Pure command plan. Does not reserve resources or create output files."""
    source_hash = interface.get('finetune_py_sha256', '')
    if interface.get('source_inventory_state') != 'verified' or not re.fullmatch(r'[0-9a-f]{64}', source_hash):
        raise ValueError('telemetry requires verified interface and exact finetune source hash')
    run_dir = Path(config['output']['run_dir'])
    if any(x in str(run_dir) for x in ('{', '}')):
        raise ValueError('telemetry requires resolved output directory')
    plain = build_finetune_command(config, interface)
    wrapper = ROOT / 'phase3/tools/run_instrumented_finetune.py'
    telemetry = ROOT / 'phase3/tools/training_telemetry.py'
    resolved_path = run_dir / 'telemetry-config.json'
    telemetry_dir = run_dir / 'telemetry'
    wrapper_hash, telemetry_hash = sha256_file(wrapper), sha256_file(telemetry)
    cmd = [plain[0], str(wrapper), '--config', str(resolved_path),
        '--inventory', str(Path(inventory_path).resolve()), '--interface', str(Path(interface_path).resolve()),
        '--source', str(Path(config['runtime']['working_directory']) / interface['entry_point']),
        '--expected-source-sha256', source_hash,
        '--expected-wrapper-sha256', wrapper_hash, '--expected-telemetry-sha256', telemetry_hash,
        '--expected-config-sha256', canonical_sha256(config, omit_key='config_sha256'),
        '--expected-interface-sha256', sha256_file(interface_path),
        '--telemetry-dir', str(telemetry_dir), '--execute', '--', *plain[2:]]
    identity = dict(enabled=True, resolved_config_path=str(resolved_path), directory=str(telemetry_dir),
        wrapper_sha256=wrapper_hash, telemetry_sha256=telemetry_hash,
        interface_sha256=sha256_file(interface_path), expected_source_sha256=source_hash)
    return cmd, identity


def verify_training_telemetry(path, resolved_config, identity, inventory_path):
    """Completed training is required; missing/partial/mismatched output fails closed."""
    report = read_json(Path(path) / 'summary.json')
    if report.get('report_sha256') != canonical_sha256(report, omit_key='report_sha256'):
        raise ValueError('telemetry report hash mismatch')
    observed = report.get('identity', {})
    expected = dict(config_sha256=canonical_sha256(resolved_config, omit_key='config_sha256'),
        inventory_sha256=read_json(Path(inventory_path))['report_sha256'],
        source_sha256=identity['expected_source_sha256'], interface_sha256=identity['interface_sha256'],
        wrapper_sha256=identity['wrapper_sha256'], telemetry_sha256=identity['telemetry_sha256'])
    if any(observed.get(key) != value for key, value in expected.items()):
        raise ValueError('telemetry execution identity mismatch')
    if report.get('state') != 'source_completed':
        raise ValueError('telemetry incomplete or failed')
    for key in ('finite_loss', 'finite_gradients'):
        if report.get(key) is not True:
            raise ValueError('telemetry did not pass ' + key)
    for key in ('zero_supervised_tokens_observed', 'unexpected_trainable_parameters'):
        if report.get(key) is not False:
            raise ValueError('telemetry did not pass ' + key)
    if report.get('optimizer_steps_observed') != resolved_config['training']['max_steps']:
        raise ValueError('telemetry optimizer step count mismatch')
    for key in ('dataset_rows_checked', 'microbatches_checked', 'samples_observed',
                'training_peak_allocated_bytes', 'training_peak_reserved_bytes',
                'training_elapsed_seconds', 'throughput_samples_per_second'):
        import math
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid telemetry ' + key)
    formal = None
    if resolved_config.get('kind') == 'lora_formal_config':
        expected = formal_telemetry_contract(resolved_config)
        for key in ('dataset_rows_checked', 'microbatches_checked', 'samples_observed', 'optimizer_steps_observed'):
            if report.get(key) != expected[key]:
                raise ValueError('formal telemetry count mismatch: ' + key)
        updates = read_json(Path(path) / 'parameter-updates.json')
        if report.get('parameter_updates_sha256') != sha256_file(Path(path) / 'parameter-updates.json'):
            raise ValueError('formal parameter update evidence hash mismatch')
        params = read_json(Path(path) / 'trainable-parameters.json')['parameters']
        names = [p['name'] for p in params]
        entries = updates.get('parameters', [])
        if sorted(x['name'] for x in entries) != sorted(names) or not names:
            raise ValueError('formal parameter update identities mismatch')
        for entry in entries:
            if any(not re.fullmatch(r'[0-9a-f]{64}', str(entry.get(k, ''))) for k in ('before_sha256', 'after_sha256')):
                raise ValueError('formal parameter update digest invalid')
        changed = sum(x['before_sha256'] != x['after_sha256'] for x in entries)
        if changed <= 0 or updates.get('changed_parameter_tensors') != changed:
            raise ValueError('formal adapter has no verified parameter changes')
        checkpoints = []
        for step in expected['checkpoint_steps']:
            directory = Path(resolved_config['output']['run_dir']) / f'checkpoint-{step}'
            state = read_json(directory / 'trainer_state.json')
            if state.get('global_step') != step:
                raise ValueError('formal checkpoint step mismatch')
            weights = [directory / n for n in ('adapter_model.bin', 'adapter_model.safetensors') if (directory / n).is_file()]
            if len(weights) != 1 or weights[0].stat().st_size <= 0:
                raise ValueError('formal checkpoint adapter missing or ambiguous')
            adapter_config = read_json(directory / 'adapter_config.json')
            if adapter_config.get('peft_type') != 'LORA' or adapter_config.get('r') != resolved_config['lora']['rank']:
                raise ValueError('formal checkpoint adapter configuration mismatch')
            checkpoints.append(dict(step=step, weights_sha256=sha256_file(weights[0]),
                config_sha256=sha256_file(directory / 'adapter_config.json'),
                trainer_state_sha256=sha256_file(directory / 'trainer_state.json')))
        formal = dict(checkpoints=checkpoints, changed_parameter_tensors=changed,
            parameter_updates_sha256=sha256_file(Path(path) / 'parameter-updates.json'))
    result = dict(summary_sha256=sha256_file(Path(path) / 'summary.json'),
        events_sha256=sha256_file(Path(path) / 'events.jsonl'),
        trainable_parameters_sha256=sha256_file(Path(path) / 'trainable-parameters.json'),
        state=report['state'])
    if formal is not None:
        result['formal'] = formal
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--interface",type=Path,default=ROOT/"phase3/configs/qwen-training-interface-v1.json"); ap.add_argument("--mode",choices=["pilot","formal"],required=True); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--attempt-id",required=True); ap.add_argument("--candidate-id"); ap.add_argument("--inventory",type=Path,required=True); ap.add_argument("--execute",action="store_true"); ap.add_argument("--attempt-record",type=Path,required=True); ap.add_argument("--resource-ledger",type=Path); ap.add_argument("--attempt-minutes",type=float)
    ap.add_argument('--telemetry', action='store_true', help='instrument actual Trainer execution; bind wrapper and report identities')
    ap.add_argument('--pilot-review', type=Path, help='explicit review/decision record; no fabricated 3B identity')
    a=ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.attempt_id): raise SystemExit("unsafe attempt id")
    if a.attempt_record.exists(): raise SystemExit("attempt record already exists; use a new attempt id")
    c=read_json(a.config); interface=read_json(a.interface)
    if a.mode == 'formal':
        if not a.telemetry:
            raise SystemExit('formal training requires --telemetry')
        formal_telemetry_contract(c)
    validator=[sys.executable,str(ROOT/"phase3/tools/validate_lora_config.py"),"--mode",a.mode,"--config",str(a.config),"--inventory",str(a.inventory),"--interface",str(a.interface)]
    if a.pilot_review:
        validator += ['--pilot-review', str(a.pilot_review)]
    validated=subprocess.run(validator,cwd=ROOT)
    if validated.returncode != 0: raise SystemExit("training launch blocked by validator")
    source_hash=canonical_sha256(c,omit_key="config_sha256")
    if a.candidate_id and a.candidate_id != c.get("candidate_id"): raise SystemExit("candidate id does not match config")
    check_data(c, a.mode)
    run_dir=c["output"]["run_dir"].replace("{attempt_id}",a.attempt_id).replace("{candidate_id}",a.candidate_id or c.get("candidate_id","formal")); c=json.loads(json.dumps(c)); c["output"]["run_dir"]=run_dir
    cmd=build_finetune_command(c,interface)
    plain_cmd=cmd[:]
    telemetry_identity=None
    if a.telemetry:
        cmd,telemetry_identity=instrumented_command(c,interface,a.inventory,a.interface)
    rec={"schema_version":1,"kind":"lora_training_attempt","mode":a.mode,"attempt_id":a.attempt_id,"candidate_id":a.candidate_id,"created_at_utc":utc_now(),"config_sha256":source_hash,"resolved_config_sha256":canonical_sha256(c,omit_key="config_sha256"),"command":cmd,"command_text":command_text(cmd),"working_directory":c["runtime"]["working_directory"],"resource_limits":c.get("resource_limits"),"execution_state":"planned"}
    if telemetry_identity:
        rec.update(telemetry=telemetry_identity,finetune_command=plain_cmd)
    rec["attempt_record_sha256"]=canonical_sha256(rec,omit_key="attempt_record_sha256"); write_json(a.attempt_record,rec)
    if not a.execute: print(rec["command_text"]); return
    if not a.resource_ledger or a.attempt_minutes is None: raise SystemExit("execute requires shared stage resource-ledger and attempt-minutes")
    if Path(run_dir).exists(): raise SystemExit("output directory already exists")
    expected_ledger=Path('/workspace/phase3/member3')/a.mode/'resource-ledger.json'
    if a.resource_ledger.resolve()!=expected_ledger: raise SystemExit(f"all {a.mode} candidates must share ledger {expected_ledger}")
    from member3_budget import reserve
    reservation=reserve(a.resource_ledger,a.mode,c["resource_limits"],a.attempt_id,source_hash,a.attempt_minutes)
    rec["budget_reservation"]=reservation
    os.makedirs(run_dir,exist_ok=False)
    if telemetry_identity:
        write_json(Path(telemetry_identity['resolved_config_path']),c)
    rec["execution_state"]="started"; rec["started_at_utc"]=utc_now(); rec["attempt_record_sha256"]=canonical_sha256(rec,omit_key="attempt_record_sha256"); write_json(a.attempt_record,rec)
    limits=c["resource_limits"]
    seconds=reservation["seconds"]
    try:
        returncode, timed_out=run_bounded(cmd,c["runtime"]["working_directory"],seconds)
        if telemetry_identity and returncode == 0:
            rec['telemetry_verification']=verify_training_telemetry(telemetry_identity['directory'],c,telemetry_identity,a.inventory)
    except Exception as error:
        rec['failure_type']=type(error).__name__
        rec['failure']=str(error)
        returncode,timed_out=2,False
    rec["ended_at_utc"]=utc_now(); rec["returncode"]=returncode; rec["execution_state"]="timed_out" if timed_out else ("completed" if returncode==0 else "failed"); rec["attempt_record_sha256"]=canonical_sha256(rec,omit_key="attempt_record_sha256"); write_json(a.attempt_record,rec); raise SystemExit(returncode)
if __name__=="__main__": main()
