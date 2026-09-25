# Member 1 environment runbook

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
memory, and one 120 GB Standard Network Volume mounted at `/workspace`. That
target is still pending as of 2026-09-25. The account is personal and has no
Pod or Network Volume. Every compatible 48 GB-or-larger candidate checked on
the deployment page -- L40S, A40, RTX A6000, and A100 PCIe 80 GB -- reported
`Out of capacity`. The A100 PCIe listing showed a USD 1.59/hour baseline.
Recheck capacity and price before any paid action. Do not run acceptance until
the exact model snapshot and fixture image have been prepared and verified on
the target volume.

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

## Step 1: provision the shared runtime

After the required RunPod account and billing actions are explicitly approved:

1. Recheck the deployment page for a compatible GPU with actual capacity. The
   smoke requires at least 44 GiB total GPU memory and 40 GiB free immediately
   before model load; a 48 GB-or-larger GPU is the practical minimum.
2. Record the exact region, GPU type, displayed hourly price, volume price, and
   intended maximum runtime for review.
3. Convert the personal account to a normal Team, with Member 1 as Admin.
4. Create one 120 GB Standard Network Volume in that same Secure Cloud region.
5. Deploy one Pod with the volume mounted at `/workspace`. Do not create a Pod
   or volume per member.
6. Use the locked bootstrap download described below. There is no existing Pod
   or disk to copy; the authoritative model source is the pinned Hugging Face
   revision.

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

## Step 4: run one GPU acceptance attempt

Use a directory that does not exist yet. Never reuse an attempt number:

```bash
bash phase3/scripts/member1_acceptance.sh \
  --member-id member1 \
  --image /workspace/phase3/smoke/contentment_05000.jpg \
  --report-dir /workspace/phase3/reports/member1/attempt-001
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

## Step 5: verify from clean and separate accounts

Member 1 first disconnects, opens a new SSH login shell, and runs:

```bash
bash phase3/scripts/verify_handoff.sh \
  --member-id member1 \
  --report-dir /workspace/phase3/reports/member1/attempt-001
```

After invitations are accepted, Member 2 repeats the command through their own
RunPod account and SSH key:

```bash
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
the handoff cross-account complete also requires private Team invitation and
SSH-login evidence for Member 2. The machine-readable status remains
`coordination_required` because that identity evidence must stay outside Git.

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
- Stop the GPU Pod immediately after acceptance and cross-account verification.
  Confirm the stopped state in the RunPod console.

## Working-time estimate

These are active-work estimates after account actions are approved and network
access is available:

| Stage | Expected active time |
| --- | ---: |
| Capacity check, Team conversion, volume, Pod, and access setup | 30-90 minutes |
| Environment, exact upstream, pinned model download, hashing, and seal | 2-5 hours |
| Real-image GPU acceptance and one retry allowance | 1-2 hours |
| Clean-shell and cross-account verification | 30-60 minutes |
| Report review and handoff | 30-60 minutes |

The expected active work is about 4.5-10.5 hours after approvals and deployable
capacity exist. Allow one to two working days because a roughly 19.3 GB model
download, full hashing, package downloads, invitation acceptance, or one retry
can add elapsed time. GPU capacity can add an unbounded wait before that work
starts. Repository scripts, pins, and evidence paths are shared and reviewable,
so Members 2-6 are not blocked by Member 1's day-to-day availability after the
handoff artifacts exist.
