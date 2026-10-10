"""CPU tensor/optimizer tests, not a claim of CUDA or full Venus execution."""
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import torch
from training_telemetry import Recorder, TelemetryError, instrument_trainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(2, 2)
        self.visual.requires_grad_(False)
        self.language = torch.nn.Module()
        self.language.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(2, 1, bias=False)})
        self.language.lora_B = torch.nn.ModuleDict({'default': torch.nn.Linear(1, 2, bias=False)})

    def forward(self):
        return self.language.lora_B['default'](self.language.lora_A['default'](torch.ones(1, 2))).square().mean()


INVENTORY = dict(matched_module_names=['language'], candidate_trainable_parameter_estimates={'1': 4})


class SmallTrainingLoop:
    """Explicit test double for Trainer control flow; uses real autograd and SGD."""
    def __init__(self, model, rows, nonfinite=False, poison_gradient=False):
        self.model, self.train_dataset = model, rows
        self.args = SimpleNamespace(world_size=1, n_gpu=0, fp16=False, deepspeed=None, device=torch.device('cpu'))
        self.state = SimpleNamespace(global_step=0)
        self.nonfinite, self.poison_gradient = nonfinite, poison_gradient
        self.callbacks = []
        self.optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def compute_loss(self, model, inputs):
        loss = model()
        return loss * float('nan') if self.nonfinite else loss

    def training_step(self, model, inputs):
        loss = self.compute_loss(model, inputs)
        (loss / 2).backward()
        if self.poison_gradient:
            next(p for p in model.parameters() if p.requires_grad).grad.fill_(float('inf'))
        return loss.detach() / 2

    def train(self):
        for index, row in enumerate(self.train_dataset):
            self.training_step(self.model, {'labels': row['labels'].unsqueeze(0)})
            if (index + 1) % 2 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1
                for callback in self.callbacks:
                    callback.on_step_end(self.args, self.state, None)
        return self.state.global_step


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.recorder = Recorder(Path(self.temp.name) / 'telemetry', {'test': True})

    def tearDown(self):
        self.recorder.close()
        self.temp.cleanup()

    def trainer(self, **kwargs):
        cls = instrument_trainer(SmallTrainingLoop, object, torch, self.recorder, INVENTORY, 1)
        rows = [{'labels': torch.tensor([-100, 1, 2])} for _ in range(4)]
        return cls(TinyModel(), rows, **kwargs)

    def test_success_real_backward_accumulation_and_optimizer(self):
        trainer = self.trainer()
        before = trainer.model.language.lora_B['default'].weight.detach().clone()
        self.assertEqual(trainer.train(), 2)
        self.assertFalse(torch.equal(before, trainer.model.language.lora_B['default'].weight))
        facts = self.recorder.summary
        self.assertEqual((facts['dataset_rows_checked'], facts['microbatches_checked'], facts['optimizer_steps_observed']), (4, 4, 2))
        self.assertTrue(facts['finite_loss'] and facts['finite_gradients'])
        self.assertIs(facts['zero_supervised_tokens_observed'], False)
        self.assertIs(facts['unexpected_trainable_parameters'], False)
        self.assertIsNone(facts['training_peak_allocated_bytes'])
        self.assertGreater(facts['throughput_samples_per_second'], 0)
        saved = json.loads((self.recorder.directory / 'summary.json').read_text())
        self.assertEqual(saved['state'], 'training_completed')

    def test_only_first_token_supervised_is_rejected(self):
        with self.assertRaises(TelemetryError):
            self.recorder.check_labels(torch, torch.tensor([1, -100, -100]), 'row')
        self.assertIs(self.recorder.summary['zero_supervised_tokens_observed'], True)

    def test_formal_records_actual_cpu_optimizer_parameter_changes(self):
        cls = instrument_trainer(SmallTrainingLoop, object, torch, self.recorder,
                                 INVENTORY, 1, track_updates=True)
        rows = [{'labels': torch.tensor([-100, 1, 2])} for _ in range(4)]
        trainer = cls(TinyModel(), rows)
        trainer.train()
        evidence = json.loads((self.recorder.directory / 'parameter-updates.json').read_text())
        self.assertGreater(evidence['changed_parameter_tensors'], 0)
        self.assertEqual(len(evidence['parameters']), 2)
        self.assertEqual(len(self.recorder.summary['parameter_updates_sha256']), 64)

    def test_formal_rejects_no_parameter_change(self):
        model = TinyModel()
        before = self.recorder.parameter_digests(model)
        with self.assertRaisesRegex(TelemetryError, 'no LoRA parameter changes'):
            self.recorder.record_parameter_updates(model, before)

    def test_one_bad_row_in_batch_is_rejected(self):
        with self.assertRaises(TelemetryError):
            self.recorder.check_labels(torch, torch.tensor([[-100, 1], [-100, -100]]), 'batch')

    def test_unfrozen_vision_stops_before_optimizer(self):
        trainer = self.trainer()
        trainer.model.visual.requires_grad_(True)
        with self.assertRaises(TelemetryError):
            trainer.train()
        self.assertEqual(trainer.state.global_step, 0)
        self.assertTrue(self.recorder.summary['unexpected_trainable_parameters'])

    def test_count_mismatch_stops(self):
        with self.assertRaises(TelemetryError):
            self.recorder.check_parameters(TinyModel(), dict(INVENTORY, candidate_trainable_parameter_estimates={'1': 5}), 1)

    def test_nonfinite_loss_stops_before_backward_and_update(self):
        trainer = self.trainer(nonfinite=True)
        with self.assertRaises(TelemetryError):
            trainer.train()
        self.assertEqual(trainer.state.global_step, 0)
        self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))
        self.assertIs(self.recorder.summary['finite_loss'], False)

    def test_nonfinite_gradient_stops_before_update(self):
        trainer = self.trainer(poison_gradient=True)
        with self.assertRaises(TelemetryError):
            trainer.train()
        self.assertEqual(trainer.state.global_step, 0)
        self.assertIs(self.recorder.summary['finite_gradients'], False)

    def test_missing_gradient_is_not_passed(self):
        with self.assertRaises(TelemetryError):
            self.recorder.check_gradients(torch, TinyModel())
        self.assertIsNone(self.recorder.summary['finite_gradients'])

    def test_output_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):
            Recorder(self.recorder.directory, {})

    def test_missing_labels_stop(self):
        with self.assertRaises(TelemetryError):
            self.recorder.check_labels(torch, None, 'batch')

    def test_wrong_adapter_dtype_stops(self):
        model = TinyModel()
        model.language.to(torch.bfloat16)
        with self.assertRaises(TelemetryError):
            self.recorder.check_parameters(model, INVENTORY, 1)

    def test_cuda_counter_calls_and_scope_using_simulated_device(self):
        # Simulated CUDA API tests scope/order only; no measured VRAM is claimed.
        calls = []
        cuda = SimpleNamespace(is_available=lambda: True,
            synchronize=lambda device: calls.append('sync'),
            reset_peak_memory_stats=lambda device: calls.append('reset'),
            max_memory_allocated=lambda device: 1234,
            max_memory_reserved=lambda device: 5678)
        proxy = SimpleNamespace(cuda=cuda, isfinite=torch.isfinite)
        cls = instrument_trainer(SmallTrainingLoop, object, proxy, self.recorder, INVENTORY, 1)
        trainer = cls(TinyModel(), [{'labels': torch.tensor([-100, 1])} for _ in range(2)])
        trainer.train()
        self.assertEqual(calls, ['sync', 'reset', 'sync'])
        self.assertEqual(self.recorder.summary['training_peak_allocated_bytes'], 1234)
        self.assertEqual(self.recorder.summary['training_peak_reserved_bytes'], 5678)

    def test_installed_transformers_compute_loss_and_training_step(self):
        from transformers import Trainer
        class Model(TinyModel):
            def forward(self, **inputs):
                return {'loss': super().forward()}
        cls = instrument_trainer(Trainer, object, torch, self.recorder, INVENTORY, 1)
        # Bypass Trainer initialization (local accelerate is unavailable), but execute
        # the installed HF compute_loss + training_step bodies with real autograd.
        trainer = object.__new__(cls)
        trainer.args = SimpleNamespace(n_gpu=0, gradient_accumulation_steps=2, past_index=-1)
        trainer.state = SimpleNamespace(global_step=0)
        trainer.label_smoother = None
        trainer.do_grad_scaling = False
        trainer.use_apex = False
        trainer._prepare_inputs = lambda inputs: inputs
        trainer.compute_loss_context_manager = nullcontext
        trainer.accelerator = SimpleNamespace(backward=lambda loss: (loss / 2).backward())
        model = Model()
        result = trainer.training_step(model, {'labels': torch.tensor([[-100, 1, 2]])})
        self.assertTrue(torch.isfinite(result))
        self.assertEqual(self.recorder.summary['microbatches_checked'], 1)
        self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
