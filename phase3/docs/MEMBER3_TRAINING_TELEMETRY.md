# Member 3 training telemetry

This is an opt-in instrumentation component, not paid-run authorization. It
is wired into `run_lora_training.py` using explicit `--telemetry` for pilot mode.
The pilot default without this flag remains unchanged. Formal mode requires
this flag and supports only the frozen 480-row release, micro batch 1,
accumulation 8, and 1–3 whole epochs (60 steps per epoch). Do not invoke this
wrapper manually to bypass the validated bounded launcher or the Pod watchdog.

## Files and integration boundary

- `tools/training_telemetry.py`: recorder and Trainer subclass factory.
- `tools/run_instrumented_finetune.py`: source-hash-checked `runpy` entry point.
- `tests/test_training_telemetry.py`: CPU autograd/optimizer and failure tests.

The wrapper replaces `transformers.Trainer` only inside its own process before
the pinned source imports `Trainer`. It restores that reference on exit and
never writes to the shared upstream checkout. Training arguments are forwarded
unchanged and compared exactly with the resolved config/interface command.
The launcher records both the wrapper command and the underlying finetune
command, hashes the wrapper/telemetry/interface/source, and creates an isolated
resolved config only after shared-ledger reservation and run-directory creation.
Dry-run records the command without reserving resources or creating that directory.
The wrapper rejects changed tool/config/interface hashes before model loading.
Successful child return is insufficient: the launcher requires a complete,
self-hash-valid telemetry report with matching execution identity, the expected
optimizer step count, finite losses/gradients, no bad trainables or zero-token
rows, and positive CUDA peaks/timing/throughput. Missing or partial reports fail
the attempt. Successful verification binds hashes of all three telemetry files.
Existing time/ledger checks and external Pod watchdog remain required.

The component supports the existing single-device LoRA contract: no trainable
visual parameters, FP32 LoRA A/B default-adapter weights only, no bias or
modules-to-save, no FP16/DeepSpeed. The runtime wrapper requires exactly one
visible CUDA device. CPU usage is for local tests only.

## Recorded checks

Before training, record every actual trainable parameter name, shape, dtype,
device and count. Require both LoRA matrices for each inventory-bound module,
the rank-specific inventory count, and a present fully frozen visual encoder.
Reject missing/unexpected trainables or wrong dtypes before any optimizer step.

Check each training dataset row and every actual microbatch. Count labels after
causal shifting (`labels[..., 1:] != -100`); each row must have at least one
supervised token. These counts include supervised response/control tokens and
must not be described as answer-body-only counts.

Check each loss before backward and each present gradient after backward and
before Trainer clipping/optimizer update. Reject nonfinite loss/gradients,
gradient on a frozen parameter, or absence of all gradients. Record L2 norms
of the accumulated gradients before clipping (not isolated per-sample norms),
microbatch counts, and Trainer optimizer-step-end events. Zero gradient norms
are allowed when finite; optimizer step events do not alone prove quality.

Reset CUDA peak counters immediately before the parent `Trainer.train` call;
synchronize and capture allocated and reserved peaks when it returns or fails.
This is the training-window peak, including resident model memory, optimizer,
checkpoint saving inside Trainer and any evaluation inside Trainer. It excludes
initial model loading, the dataset precheck and later external validation.
Elapsed time includes instrumentation overhead; throughput is observed input
samples (including repeated epochs) divided by that elapsed time.

## Evidence and failure semantics

A new directory contains `events.jsonl`, `trainable-parameters.json` and
`summary.json`. Existing directories are refused. Events are flushed; summary
updates are atomic and have a canonical self hash. The wrapper binds config,
inventory, upstream source, wrapper, telemetry implementation and Trainer source
hashes plus package versions. These private runtime files are not automatically
Git-safe. They contain parameter names and runtime identity, but no image bytes
or conversation contents.

Unknown checks start as `null`, not passed. CPU tests write `null` for CUDA
peaks. Exceptions produce failed summaries and propagate nonzero exit status.
Hard kills cannot run cleanup: a summary still marked `training` is incomplete,
never success. `training_completed` does not establish adapter save/reload,
validation or final acceptance. `source_completed` only means the upstream
source returned after instrumented training; those other gates still require
their existing independent evidence.

This telemetry does not measure official billing, control Pod lifecycle, or
repair historical pilot omissions. Cost/billing timestamps and clean-process
reload evidence remain separate. Next-run values cannot be substituted into
the first pilot's missing fields.

## Bounded formal support (2026-10-10 local implementation)

Formal mode checks exact dataset-row, microbatch, observed sample and optimizer
step counts after training. It requires every epoch checkpoint to remain present
with matching Trainer global_step, a readable rank-matching LoRA config and one
nonempty adapter weight file. It binds hashes of these files in the attempt.
This establishes saved-file identity, not successful reload or valid tensors;
independent checkpoint loading and validation are still mandatory.

For formal runs only, the recorder hashes the actual FP32 trainable tensors on
CPU before and after Trainer.train. It records parameter names and before/after
digests in parameter-updates.json and fails if identities change or no tensor
changes. The launcher checks that file against the summary hash, the recorded
trainable identities and the change count. No model weights are modified by this
comparison. One changed tensor is evidence of a numerical update, not a claim
that every module learned or model quality improved.

The formal review validator, full-data comparison, shared resource ledger,
bounded process timeout and external Pod watchdog remain required. No approval
record or cloud compatibility result was created by this local implementation.

## Local verification boundary

Local tests use real PyTorch CPU tensors, backward, gradient accumulation and
SGD updates. They test HF Transformers 4.32.0 `compute_loss` and `training_step`
method bodies with a lightweight accelerator substitute. Full Trainer loop
tests use an explicitly named test double; local `accelerate` is unavailable.
Simulated CUDA calls test reset/capture ordering only. No local test claims a
complete Venus run, actual CUDA peak measurement or cloud compatibility pass.
Launcher tests use temporary fixtures and mocked model/budget processes;
two real wrapper subprocess tests reject argument/hash drift before model loading.
Run a bounded runtime compatibility check on the retained pinned environment
before proceeding to candidate training.
