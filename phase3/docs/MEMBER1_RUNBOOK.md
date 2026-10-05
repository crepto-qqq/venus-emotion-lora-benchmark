# Member 1 environment runbook

> Historical private-environment runbook: cloud state was recorded on
> 2026-09-25, and the private volume name/identifier are redacted. The public
> checkout also omits the reviewed Qwen patch, model, data, and fixture needed
> to execute the end-to-end commands.

## Purpose and stop line

Member 1 establishes a controlled Phase 3 runtime with pinned top-level
dependencies and a captured resolved package freeze, then proves it with one
real image in a controlled forward/backward pass. The floating base parameters
and input entering the PEFT wrapper are BF16; the trainable LoRA adapter
parameters are FP32. This evidence does not assert the dtype of every internal
LoRA matrix operation. The run stops after gradient checks, performs no
optimizer step, and saves no adapter or checkpoint. Full-data conversion,
formal LoRA settings, training, save/reload, inference, evaluation, and the
small demonstration caller belong to later work.

The project checkout must use branch `phase3member1-env`, based on `tlia0262`
commit `869a24bb42b83f8b58c6f693a2cae6a84df5ae9d`. All Member 1 additions
stay under `phase3/`.

## Preconditions

Use the target setup in [`RUNPOD_TEAM_SETUP.md`](RUNPOD_TEAM_SETUP.md): a
normal RunPod Team, one on-demand Secure Cloud Pod with at least 48 GB GPU
memory, and one 120 GB Standard Network Volume mounted at `/workspace`. This
technical target completed on 2026-09-25. A private volume in `US-NE-1` held
the shared environment, exact model snapshot, and fixture at the recorded
handoff point. H100 NVL
`attempt-001` and both Member 1 verification passes succeeded at project commit
`ea2dace2026523b6a498d634301927ddc335e015`. All temporary Pods are stopped.

Members 2-6 have not yet been invited, and no separate-account identity check
is claimed. Before starting any future Pod, recheck capacity and price, mount
the existing volume, and use an automatic stop limit. Do not create a second
Phase 3 volume.

These inputs are fixed:

```text
Venus source:       44e2971b7fe41f88a88c7e0ed8bf4c85eeb0155f
Qwen-VL fine-tune:  efa37ba284d56192b246d9b4ed5d3668c1abd163
Qwen patch SHA-256: be9e14a60f4f2189a108aa1f46a7384616bfdd527ca6c0a41a4c175a92172eba
Venus-Q-Stage1:     0f5c00c8d07ba889e9c5d12f828129dc322aae6a
Model path:         /workspace/models/Venus-Q-Stage1
```

Reject a source or model whose recorded revision differs. Do not compensate for
a failure with quantization, CPU offload, a floating branch, or changed LoRA
settings.

## Stable shared layout

```text
/workspace/phase3/
  envs/venus-phase3/  one shared pinned Python environment
  upstream/           pinned Venus and patched Qwen checkouts
  cache/              shared package and model caches
  data/               runtime-only source data
  smoke/              contentment_05000 technical fixture
  members/<id>/        member-owned work
  runs/<id>/           member-owned mutable run output
  reports/<id>/        redacted evidence owned by that member
  locks/               shared writer lock
```

The model remains at `/workspace/models/Venus-Q-Stage1`. Treat the model and
both upstream checkouts as shared read-only inputs after bootstrap. Members may
write only to their own `members/<id>`, `runs/<id>`, and `reports/<id>`
paths.

## Step 1: reuse the completed shared runtime

Initial provisioning and technical acceptance are complete. For recovery or a
later approved compute window:

1. Recheck the deployment page for a compatible GPU with actual capacity. The
   smoke requires at least 44 GiB total GPU memory and 40 GiB free immediately
   before model load; a 48 GB-or-larger GPU is the practical minimum.
2. Record the exact region, GPU type, displayed hourly price, volume price, and
   intended maximum runtime for review.
3. Confirm the existing Team still lists Member 1 as Admin. This was complete
   on 2026-09-25; do not create another Team.
4. In the authorised private environment, reuse the existing approved volume;
   do not create another volume solely from this historical runbook.
5. Deploy at most one approved Pod with that volume mounted at `/workspace` and
   configure an automatic stop limit.
6. Run bootstrap only to verify or repair the retained runtime. It downloads
   only missing files from the pinned Hugging Face revision.

## Step 2: bootstrap from a clean shell

Open a new SSH login shell. Do not activate an earlier environment or depend on
shell-local variables. From the project repository root:

```bash
bash phase3/scripts/bootstrap.sh \
  --member-id member1 \
  --workspace-root /workspace/phase3 \
  --model-path /workspace/models/Venus-Q-Stage1 \
  --download-model-if-missing
```

The script takes the shared writer lock, creates the stable directory layout,
creates or verifies `/workspace/phase3/envs/venus-phase3`, fetches the exact
Venus and Qwen commits, applies and verifies the reviewed Qwen patch, and
prepares the Stage 1 snapshot. Missing model files are downloaded only from the
locked Hugging Face revision through the persistent
`/workspace/phase3/cache/huggingface` cache. The preparation verifies all 22
files, writes the exact provenance marker, removes all write bits from the
pinned files, marker, and model root, then validates the structured report.
Bootstrap finally runs static preflight and records a full package freeze. Its
evidence is written to:

```text
/workspace/phase3/reports/member1/bootstrap/
  model-snapshot.json
  preflight-static.json
  requirements.freeze.txt
  checksums.sha256
```

Bootstrap is idempotent. An existing source checkout must have the exact pinned
origin, revision, and reviewed diff. The script refuses unexpected changes. An
incomplete environment path is left in place for inspection rather than
silently deleted. It also pre-creates caller-owned paths for all six members
with group-collaboration permissions; path ownership remains a team convention,
because Pod users with shell access are trusted collaborators.

## Step 3: prepare the one-record fixture

Place the real image at:

```text
/workspace/phase3/smoke/contentment_05000.jpg
```

Its required SHA-256 is
`f4552a57efd8ff17e0a7a9fe28e1e94e98401cc5a5ace21ee6c246d74766d082`.
It must also be exactly 126,584 bytes and decode to 720 x 480 pixels.
The acceptance command reads the exact first record from
`data/datasets/contentment/train_contentment.jsonl`, checks the source record,
annotation, and image hashes, checks available Eval20/Eval80 exclusion evidence,
and emits the one-sample upstream JSON array plus its sidecar manifest. It does
not convert the complete 480-record snapshot.

## Step 4: reproduce the GPU acceptance only when required

`attempt-001` already passed and is immutable. Any justified rerun must use a
directory that does not exist, starting with `attempt-002`:

```bash
bash phase3/scripts/member1_acceptance.sh \
  --member-id member1 \
  --image /workspace/phase3/smoke/contentment_05000.jpg \
  --report-dir /workspace/phase3/reports/member1/attempt-002
```

The command takes the shared writer lock and then runs:

1. runtime preflight for source, model, disk, CUDA, BF16, package, at least
   44 GiB total GPU memory, and at least 40 GiB immediately free GPU memory;
2. deterministic one-record fixture creation;
3. one real forward loss and backward pass using BF16 floating base parameters,
   BF16 input entering the PEFT wrapper, and FP32 technical rank 2 LoRA adapter
   parameters;
4. aggregate report and SHA-256 manifest creation.

A successful attempt contains:

```text
preflight-runtime.json
smoke-backward.json
member1-handoff.json
technical-fixture.json
technical-fixture-manifest.json
environment.freeze.txt
fixture-artifacts.sha256
checksums.json
checksums.sha256
```

The attempt becomes read-only after success. It passes only when the loss and
an intended trainable gradient are finite, at least one intended gradient is
non-zero, all pins match, exactly 128 target modules are matched, exactly
3,506,176 LoRA parameters are trainable, and the 44/40 GiB GPU-memory gates
pass. It also requires `optimizer_step_performed: false`,
`adapter_saved: false`, `full_dataset_ready: false`, and
`formal_training_authorized: false`. A passing report establishes runtime
viability only.

## Step 5: preserve completed verification and defer account onboarding

Member 1 clean-shell verification and a second verification from a replacement
Pod attached to the retained volume both passed. Their reports are stored at:

```text
/workspace/phase3/reports/member1/verification-h100-001/
/workspace/phase3/reports/member1/verification-shared-volume-001/
```

The immutable attempt records project commit `ea2dace`. A later verifier must
use a clean checkout of exactly that commit. After invitations are accepted,
Member 2 may run this through their own RunPod account and SSH key:

```bash
git clone --no-checkout "$(git remote get-url origin)" \
  /workspace/phase3/members/member2/member1-acceptance
git -C /workspace/phase3/members/member2/member1-acceptance \
  switch --detach ea2dace2026523b6a498d634301927ddc335e015
cd /workspace/phase3/members/member2/member1-acceptance
bash phase3/scripts/verify_handoff.sh \
  --member-id member2 \
  --report-dir /workspace/phase3/reports/member1/attempt-001
```

Verification checks the existing hashes, schemas, exact Phase 3 commit, exact
Qwen diff, full 22-file model manifest, captured environment freeze, fixture
JSON and sidecar semantics, fixture hashes, shared reads, and member-scoped
write paths. It rejects symlinks, nested entries, writable source artifacts, or
missing shared read bits. It snapshots Member 1's attempt before and after the
check to prove it was not changed and replaces any provisional success with a
`source_bundle_changed` failure if the bundle changes. It never reruns the GPU
smoke. By default, its timestamped result and checksum are sealed read-only
below `/workspace/phase3/reports/<caller>/`.

The script validates filesystem behavior under the supplied `--member-id`; it
does not authenticate the RunPod account or SSH key behind the shell. Calling
the handoff cross-account complete will require private Team invitation and
SSH-login evidence for Member 2. That coordination step is currently deferred,
and the machine-readable status remains `coordination_required`.

## Recovery

- Rerun bootstrap after a network interruption. It verifies completed artifacts
  before reuse.
- Preserve every failed acceptance directory and use `attempt-002`,
  `attempt-003`, and so on. A smoke failure keeps its machine-readable report;
  a handoff-verification failure records a stable `failure_code`, retains
  completed checks when possible, writes a checksum, and seals caller-owned
  evidence.
- `flock` releases the kernel lock when its process exits. The lock file may
  remain as an audit marker and must not be treated as proof that a process is
  still running.
- If a pin or checksum differs, stop and restore the exact input. Do not repair
  a shared checkout in place while another member is using it.
- If either the 44 GiB total-memory gate or 40 GiB immediate free-memory gate
  fails, retain the failure report and stop. Do not change the technical test
  into a different experiment.
- Stop the GPU Pod immediately after technical acceptance and Member 1
  verification. Do not keep an expensive GPU running while waiting for account
  onboarding. Confirm the stopped state in the RunPod console.

## Historical working-time estimate

These are active-work estimates after account actions are approved and network
access is available:

| Stage | Expected active time |
| --- | ---: |
| Capacity check, Team conversion, volume, Pod, and access setup | 30-90 minutes |
| Environment, exact upstream, pinned model download, hashing, and seal | 2-5 hours |
| Real-image GPU acceptance and one retry allowance | 1-2 hours |
| Clean-shell and cross-account verification | 30-60 minutes |
| Report review and handoff | 30-60 minutes |

This was the planning estimate for the now-completed Member 1 technical work.
No further GPU work is required for Member 1. The only deferred Member 1
coordination is inviting Members 2-6 and helping one member perform the
separate-account access check; it does not require an H100 to remain running.
Repository scripts, pins, and evidence paths are shared and reviewable, so
Members 2-6 are not blocked by Member 1's day-to-day availability after account
onboarding.
