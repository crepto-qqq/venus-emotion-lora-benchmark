"""Opt-in, single-GPU Trainer instrumentation; never changes weights or source files."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path


class TelemetryError(RuntimeError):
    pass


class Recorder:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events = (self.directory / 'events.jsonl').open('x', encoding='utf-8')
        self.summary = dict(schema_version=1, identity=identity, state='initialized',
                            dataset_rows_checked=0, microbatches_checked=0,
                            optimizer_steps_observed=0, samples_observed=0,
                            zero_supervised_tokens_observed=None,
                            unexpected_trainable_parameters=None,
                            finite_loss=None, finite_gradients=None,
                            memory_scope='Trainer.train only, excluding initial model loading and validation')
        self.started = None
        self.save()

    def save(self):
        body = dict(self.summary)
        body['report_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True,
            separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        temp = self.directory / 'summary.json.tmp'
        temp.write_text(json.dumps(body, indent=2, allow_nan=False) + '\n')
        temp.replace(self.directory / 'summary.json')

    def event(self, kind, **fields):
        self.events.write(json.dumps(dict(kind=kind, **fields), allow_nan=False) + '\n')
        self.events.flush()

    def fail(self, message, **fields):
        self.summary.update(state='failed', failure=message)
        self.event('failure', message=message, **fields)
        self.save()
        raise TelemetryError(message)

    def check_labels(self, torch, labels, kind, **identity):
        if labels is None or labels.ndim not in (1, 2):
            self.fail('missing or invalid labels', **identity)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        # CausalLM predicts labels[..., 1:], including any supervised control tokens.
        counts = (labels[..., 1:] != -100).sum(dim=-1).detach().cpu().tolist()
        self.event(kind, supervised_tokens_per_row=counts, **identity)
        if not counts or min(counts) <= 0:
            self.summary['zero_supervised_tokens_observed'] = True
            self.fail('zero shifted supervised tokens', **identity)
        return len(counts)

    def check_parameters(self, model, inventory, rank):
        expected = set(inventory['matched_module_names'])
        actual = set()
        rows, bad = [], []
        visual_total = visual_trainable = 0
        for name, parameter in model.named_parameters():
            is_visual = 'visual' in name.split('.')
            if is_visual:
                visual_total += parameter.numel()
                if parameter.requires_grad:
                    visual_trainable += parameter.numel()
            if not parameter.requires_grad:
                continue
            rows.append(dict(name=name, shape=list(parameter.shape), numel=parameter.numel(),
                             dtype=str(parameter.dtype), device=str(parameter.device)))
            normalized = name.removeprefix('base_model.model.')
            matched = [module for module in expected if normalized in
                (module + '.lora_A.default.weight', module + '.lora_B.default.weight')]
            if len(matched) != 1 or str(parameter.dtype) != 'torch.float32' or is_visual:
                bad.append(name)
            else:
                actual.add(normalized)
        required = {module + suffix for module in expected for suffix in
                    ('.lora_A.default.weight', '.lora_B.default.weight')}
        count = sum(row['numel'] for row in rows)
        report = dict(parameters=rows, trainable_numel=count,
                      visual_numel=visual_total, visual_trainable_numel=visual_trainable,
                      unexpected_names=bad, missing_names=sorted(required - actual))
        (self.directory / 'trainable-parameters.json').write_text(json.dumps(report, indent=2) + '\n')
        estimate = inventory['candidate_trainable_parameter_estimates'][str(rank)]
        if bad or actual != required or count != estimate or not visual_total or visual_trainable:
            self.summary['unexpected_trainable_parameters'] = True
            self.fail('actual trainable parameter inventory mismatch')
        self.summary['unexpected_trainable_parameters'] = False
        self.save()

    def check_gradients(self, torch, model):
        squared, present = [], 0
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if not parameter.requires_grad:
                self.fail('gradient on frozen parameter', parameter=name)
            if not bool(torch.isfinite(gradient).all().item()):
                self.summary['finite_gradients'] = False
                self.fail('nonfinite gradient', parameter=name)
            squared.append(gradient.detach().float().norm(2).double().square())
            present += 1
        if not present:
            self.fail('no gradients observed after backward')
        norm = math.sqrt(sum(value.item() for value in squared))
        if not math.isfinite(norm):
            self.summary['finite_gradients'] = False
            self.fail('nonfinite gradient norm')
        self.event('gradient', microbatch=self.summary['microbatches_checked'] + 1,
                   parameters_with_grad=present, accumulated_pre_clip_l2_norm=norm)

    def close(self):
        self.events.close()

    def parameter_digests(self, model):
        # LoRA is FP32 by contract. Hash CPU bytes one tensor at a time; no retained
        # copies of the model and no writes to weights or the upstream checkout.
        return {name: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for name, p in model.named_parameters() if p.requires_grad}

    def record_parameter_updates(self, model, before):
        after = self.parameter_digests(model)
        if before.keys() != after.keys():
            self.fail('trainable parameter identities changed during training')
        changed = sum(before[n] != after[n] for n in before)
        evidence = dict(changed_parameter_tensors=changed, parameters=[dict(name=n,
            before_sha256=before[n], after_sha256=after[n]) for n in sorted(before)])
        path = self.directory / 'parameter-updates.json'
        path.write_text(json.dumps(evidence, indent=2) + '\n')
        self.summary['parameter_updates_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not changed:
            self.fail('no LoRA parameter changes observed')


def instrument_trainer(base, callback_base, torch, recorder, inventory, rank, track_updates=False):
    class StepObserver(callback_base):
        def on_step_end(self, args, state, control, **kwargs):
            recorder.summary['optimizer_steps_observed'] += 1
            recorder.event('optimizer_step_end', global_step=int(state.global_step))
            recorder.save()

    class ObservedTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.args.world_size != 1 or self.args.n_gpu > 1 or self.args.fp16 or self.args.deepspeed:
                recorder.fail('telemetry supports single-device non-FP16 non-DeepSpeed training only')
            self.add_callback(StepObserver())

        def compute_loss(self, model, inputs, *args, **kwargs):
            loss = super().compute_loss(model, inputs, *args, **kwargs)
            value = loss[0] if isinstance(loss, tuple) else loss
            if not bool(torch.isfinite(value).all().item()):
                recorder.summary['finite_loss'] = False
                recorder.fail('nonfinite loss before backward')
            recorder.event('loss', value=float(value.detach().item()),
                           global_step=int(self.state.global_step))
            return loss

        def training_step(self, model, inputs, *args, **kwargs):
            samples = recorder.check_labels(torch, inputs.get('labels'), 'batch',
                microbatch=recorder.summary['microbatches_checked'] + 1,
                global_step=int(self.state.global_step))
            result = super().training_step(model, inputs, *args, **kwargs)
            recorder.check_gradients(torch, model)
            recorder.summary['microbatches_checked'] += 1
            recorder.summary['samples_observed'] += samples
            recorder.save()
            return result

        def train(self, *args, **kwargs):
            recorder.check_parameters(self.model, inventory, rank)
            before = recorder.parameter_digests(self.model) if track_updates else None
            for index in range(len(self.train_dataset)):
                recorder.check_labels(torch, self.train_dataset[index].get('labels'),
                                      'dataset_row', row=index)
                recorder.summary['dataset_rows_checked'] += 1
            if not recorder.summary['dataset_rows_checked']:
                recorder.fail('empty training dataset')
            cuda = torch.cuda.is_available()
            device = self.args.device
            if cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            recorder.started = time.monotonic()
            recorder.summary.update(state='training', device=str(device))
            recorder.save()
            succeeded = False
            try:
                result = super().train(*args, **kwargs)
                if not recorder.summary['microbatches_checked'] or not recorder.summary['optimizer_steps_observed']:
                    recorder.fail('no training microbatches or optimizer steps observed')
                if track_updates:
                    recorder.record_parameter_updates(self.model, before)
                recorder.summary.update(finite_loss=True, finite_gradients=True,
                                        zero_supervised_tokens_observed=False)
                succeeded = True
                return result
            finally:
                if cuda:
                    torch.cuda.synchronize(device)
                    recorder.summary['training_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                    recorder.summary['training_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
                else:
                    recorder.summary['training_peak_allocated_bytes'] = None
                    recorder.summary['training_peak_reserved_bytes'] = None
                elapsed = time.monotonic() - recorder.started
                recorder.summary.update(training_elapsed_seconds=elapsed,
                    throughput_samples_per_second=recorder.summary['samples_observed'] / elapsed)
                recorder.summary['state'] = 'training_completed' if succeeded else 'failed'
                recorder.save()
    return ObservedTrainer
