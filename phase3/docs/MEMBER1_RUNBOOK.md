# Member 1 environment runbook

## Purpose and stop line

Member 1 establishes a controlled Phase 3 runtime with pinned top-level
dependencies and a captured resolved package freeze, then proves it with one
real image in BF16 forward/backward. The run stops after gradient checks. It performs
no optimizer step and saves no adapter or checkpoint. Full-data conversion,
formal LoRA settings, training, save/reload, inference, evaluation, and the
small demonstration caller belong to later work.

The project checkout must use branch `phase3member1-env`, based on `tlia0262`
commit `869a24bb42b83f8b58c6f693a2cae6a84df5ae9d`. All Member 1 additions
stay under `phase3/`.

## Preconditions

Use the target setup in [`RUNPOD_TEAM_SETUP.md`](RUNPOD_TEAM_SETUP.md): a
normal RunPod Team, one on-demand Secure Cloud Pod with at least 48 GB GPU
memory, and one 120 GB Standard Network Volume mounted at `/workspace`. That
target is still pending as of 2026-09-23; the existing account has two stopped
legacy A40 Pods with separate Pod Volume Disks and no Network Volume. Do not run
the acceptance command until the selected model snapshot and fixture image have
been copied and verified on the target volume.

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

## Step 1: provision and migrate the shared storage

After the required RunPod account and billing actions are explicitly approved:

1. Convert the personal account to a normal Team, with Member 1 as Admin.
2. Create one 120 GB Standard Network Volume in a Secure Cloud location.
3. Start only the old Pod that contains the verified Stage 1 snapshot and
   required fixture source; record its start time and current hourly price.
4. Copy the required immutable model snapshot and fixture source to the new
   Network Volume, then verify counts and hashes before stopping the old Pod.
5. Keep both old Pod Volume Disks until the migration evidence has been checked.
   Their deletion is a separate destructive decision.
6. Deploy the new 48 GB Pod with the Network Volume mounted at `/workspace`.

Do not claim a migration from a file count alone. Record the model revision,
ten weight-shard count, total shard bytes, index file, and the technical image
SHA-256.

## Step 2: bootstrap from a clean shell

Open a new SSH login shell. Do not activate an earlier environment or depend on
shell-local variables. From the project repository root:

```bash
bash phase3/scripts/bootstrap.sh \
  --member-id member1 \
  --workspace-root /workspace/phase3 \
  --model-path /workspace/models/Venus-Q-Stage1
```

The script takes the shared writer lock, creates the stable directory layout,
creates or verifies `/workspace/phase3/envs/venus-phase3`, fetches the exact
Venus and Qwen commits, applies and verifies the reviewed Qwen patch, runs the
static preflight, and records a full package freeze. Its evidence is written to:

```text
/workspace/phase3/reports/member1/bootstrap/
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
The acceptance command reads the exact first record from
`data/datasets/contentment/train_contentment.jsonl`, checks the source record,
annotation, and image hashes, checks available Eval20/Eval80 exclusion evidence,
and emits the one-sample upstream JSON array. It does not convert the complete
480-record snapshot.

## Step 4: run one GPU acceptance attempt

Use a directory that does not exist yet. Never reuse an attempt number:

```bash
bash phase3/scripts/member1_acceptance.sh \
  --member-id member1 \
  --image /workspace/phase3/smoke/contentment_05000.jpg \
  --report-dir /workspace/phase3/reports/member1/attempt-001
```

The command takes the shared writer lock and then runs:

1. runtime preflight for source, model, disk, CUDA, BF16, package, and GPU-memory
   requirements;
2. deterministic one-record fixture creation;
3. one real BF16 forward loss and backward pass using technical rank 2 LoRA;
4. aggregate report and SHA-256 manifest creation.

A successful attempt contains:

```text
preflight-runtime.json
smoke-backward.json
member1-handoff.json
technical-fixture-manifest.json
environment.freeze.txt
fixture-artifacts.sha256
checksums.json
checksums.sha256
```

The attempt becomes read-only after success. It passes only when the loss and
an intended trainable gradient are finite, at least one intended gradient is
non-zero, all pins match, `optimizer_step_performed` is false,
`adapter_saved` is false, `full_dataset_ready` is false, and
`formal_training_authorized` is false. A passing report establishes runtime
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
evidence chain, shared reads, and member-scoped write paths. It snapshots Member 1's
attempt before and after the check to prove it was not changed. It never reruns
the GPU smoke. By default, its timestamped result is created below
`/workspace/phase3/reports/<caller>/`.

The script validates filesystem behavior under the supplied `--member-id`; it
does not authenticate the RunPod account or SSH key behind the shell. Calling
the handoff cross-account complete also requires private Team invitation and
SSH-login evidence for Member 2. The machine-readable status remains
`coordination_required` because that identity evidence must stay outside Git.

## Recovery

- Rerun bootstrap after a network interruption. It verifies completed artifacts
  before reuse.
- Preserve every failed acceptance directory and use `attempt-002`,
  `attempt-003`, and so on.
- `flock` releases the kernel lock when its process exits. The lock file may
  remain as an audit marker and must not be treated as proof that a process is
  still running.
- If a pin or checksum differs, stop and restore the exact input. Do not repair
  a shared checkout in place while another member is using it.
- If 48 GB is insufficient, retain the failure report and stop. Do not change
  the technical test into a different experiment.
- Stop the GPU Pod immediately after acceptance and cross-account verification.
  Confirm the stopped state in the RunPod console.

## Working-time estimate

These are active-work estimates after account actions are approved and network
access is available:

| Stage | Expected active time |
| --- | ---: |
| Team, storage, access, and migration checks | 1-3 hours |
| Environment and exact upstream bootstrap | 1-2 hours |
| Real-image GPU acceptance and one retry allowance | 1-2 hours |
| Clean-shell and cross-account verification | 30-60 minutes |
| Report review and handoff | 30-60 minutes |

Allow one to two working days for Member 1, mainly because model transfer,
package downloads, GPU availability, team invitation acceptance, or a retry can
add elapsed time. None of Member 1's tasks should become a dependency on the
user's availability: repository scripts, pins, and evidence paths are shared
and reviewable by the other members.
