# Member 1 handoff

## Completion boundary

The repository-ready portion contains the branch, pins, reviewed patch,
automation, schemas, and instructions. The complete Member 1 handoff additionally
requires these runtime deliverables:

- branch `phase3member1-env`, based on `tlia0262` commit
  `869a24bb42b83f8b58c6f693a2cae6a84df5ae9d`;
- one shared environment at `/workspace/phase3/envs/venus-phase3`, built from
  pinned top-level dependencies and accompanied by its captured package freeze;
- exact Venus and Qwen source integration with a reviewed compatibility patch;
- direct preparation and read-only sealing of the immutable Venus-Q-Stage1
  snapshot at its pinned Hugging Face revision;
- one real `contentment_05000` forward/backward technical run with BF16
  floating base parameters and wrapper input plus FP32 trainable LoRA adapter
  parameters;
- redacted static, runtime, smoke, aggregate, and handoff-verification evidence;
- the RunPod Team, shared volume layout, access procedure, and one-writer rule.

All source additions live below `phase3/`. This work stops before any optimizer
step and does not include a formal training configuration, full converter,
adapter save/reload, training, inference, or evaluation. Member 1 implements no
API. Any later callable demo is inference-only. Training remains CLI-based, and
no training, data-upload, or evaluation HTTP API is in scope.

## Required revisions

| Component | Required revision |
| --- | --- |
| Venus source | `44e2971b7fe41f88a88c7e0ed8bf4c85eeb0155f` |
| Qwen-VL fine-tuning source | `efa37ba284d56192b246d9b4ed5d3668c1abd163` |
| Qwen compatibility patch | `be9e14a60f4f2189a108aa1f46a7384616bfdd527ca6c0a41a4c175a92172eba` |
| `popo28/Venus-Q-Stage1` | `0f5c00c8d07ba889e9c5d12f828129dc322aae6a` |

The model lives at `/workspace/models/Venus-Q-Stage1`. Bootstrap may download
missing files directly from the locked revision, verifies the full 22-file
manifest and exact provenance, then removes write bits from the pinned files,
provenance marker, and model root. The patched Qwen tree and clean Venus tree
live below `/workspace/phase3/upstream`. Later members must treat these as
shared read-only inputs.

## Repository state versus cloud acceptance

The checked-in scripts, pins, schemas, and documentation are repository-side
preparation. They are not evidence that RunPod provisioning or GPU acceptance
has happened. As of 2026-09-25, the RunPod Team exists, Member 1 is its Admin,
and Member 1 is currently its only member. There is no Phase 3 Pod or Network
Volume. The 120 GB Network Volume, new shared Pod, invitations for Members 2-6,
and real GPU reports remain runtime actions. Every compatible 48 GB-or-larger
candidate checked on the deployment page -- L40S, A40, RTX A6000, and A100
PCIe 80 GB -- reported `Out of capacity`. The A100 PCIe listing showed a
USD 1.59/hour baseline. No GPU availability is claimed until the deployment
page is checked again.

The handoff becomes complete only when:

1. `bootstrap.sh` succeeds from a clean login shell on the target storage;
2. `member1_acceptance.sh` produces a passing immutable attempt for the real
   `contentment_05000` image;
3. the report proves at least 44 GiB total and 40 GiB immediately free GPU
   memory, BF16 floating base parameters, BF16 wrapper input, FP32 trainable
   LoRA adapter parameters, finite loss, and a finite non-zero intended LoRA
   gradient;
4. it records zero optimizer steps, no adapter/checkpoint save, and
   `full_dataset_ready: false` plus `formal_training_authorized: false`;
5. Member 1 verifies from a fresh login and Member 2 verifies through a
   separate RunPod account and SSH key;
6. Member 1 reviews the redacted evidence and stops the GPU Pod.

## Evidence later members may rely on

Bootstrap evidence:

```text
/workspace/phase3/reports/member1/bootstrap/
  model-snapshot.json
  preflight-static.json
  requirements.freeze.txt
  checksums.sha256
```

One immutable acceptance attempt:

```text
/workspace/phase3/reports/member1/attempt-001/
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

Each handoff verification is timestamped below the caller's own report root,
for example `/workspace/phase3/reports/member2/verification-*/`. Verification
does not rewrite Member 1's attempt and does not execute the model. It checks a
flat immutable bundle of regular files, validates both the fixture JSON and
sidecar against the source locks, and compares before/after source snapshots.
Success and failure outputs are sealed with checksums. A failure carries a
stable `failure_code` and retains completed checks whenever possible.

The verifier proves that the bundle, pinned inputs, shared reads, and
caller-owned write path work under the supplied `--member-id`. That text value
does not authenticate the RunPod account or SSH key that opened the shell.
Calling the handoff cross-account complete therefore also requires private
evidence that Member 2 joined the Team and logged in with Member 2's own SSH
key. Keep that identity evidence outside Git and outside redacted reports.

The passing aggregate report establishes technical environment viability only.
Later owners must still complete the deterministic full-data converter, repair
and revalidate the released dataset, freeze formal LoRA settings, train and
save an adapter, reload it, implement the simple demo inference caller, and run
the agreed evaluation.

## Current data boundary

The pulled project snapshot contains 480 JSONL records, but it is not direct
input to the pinned Qwen loader and is not fully accepted:

```bash
python -m src.datasets.annotation_pipeline validate \
  --output-root data/datasets \
  --require-accepted
```

- `amusement` has 50 training and 10 validation records; all 60 fail
  `--require-accepted`, producing 60 review-status findings;
- the 60 `excitement` records each produce one target-structure finding and
  five missing-provenance-field findings, for 60 + 300 = 360 findings;
- `--require-accepted` therefore reports 420 findings across 120 affected
  records: 60 amusement review-status + 60 excitement target-structure + 300
  excitement missing-provenance findings.

The number 420 counts findings, not distinct bad records. These defects do not
block Member 1's isolated fixture, but they require
`full_dataset_ready: false`. The complete converter and release checks belong
to the later data owner.

A separate semantic inspection found three `excitement` exceptions that are
not additional items in that 420 breakdown:

- `excitement_05052` records `emotion: excitement`, while its target describes
  contentment;
- `excitement_05058` records `emotion: excitement`, while its target describes
  amusement;
- `excitement_05016` is stored and identified as an excitement example, but
  both its record emotion and target are contentment. This causes the observed
  validation counts of 11 contentment and 9 excitement examples.

Running `--require-accepted --require-complete` adds the resulting
`contentment:validation` count error to the 420 findings, producing 421. The
configured `initial_emotions` list contains only `sadness`, `awe`, and
`contentment`, so `--require-complete` checks expected counts only for those
three classes and cannot establish eight-class completeness by itself.

## Consumption rules

Use the member-scoped paths in
[`RUNPOD_TEAM_SETUP.md`](RUNPOD_TEAM_SETUP.md). Do not alter another member's
directory, a completed report attempt, the upstream trees, or the shared model.
Any operation that mutates shared environment/source/fixture state uses the
shared writer lock.

Only Member 1 runs the shared-state wrappers `bootstrap.sh` and
`member1_acceptance.sh`. Lower-level environment, source-fetch, and fixture
builder commands are internal implementation steps; do not invoke them directly
against shared storage outside those locked wrappers.

Only redacted reports and source belong in Git. Images, model weights, caches,
checkpoints, adapters, private logs, account links, and credentials stay
outside it.
