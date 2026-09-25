# Member 1 handoff

## Completion boundary

Member 1's technical environment handoff contains the branch, pins, reviewed
patch, automation, schemas, instructions, and these completed runtime
deliverables:

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

Member 1 technical environment acceptance is complete as of 2026-09-25:

1. the shared 120 GB Standard Network Volume `phase3-shared-120gb`
   (`bk4fycduml`) is retained in `US-NE-1`; it was mounted at `/workspace` for
   acceptance and must use that mount point on future Pods;
2. bootstrap produced the shared Python 3.10.13/PyTorch 2.0.1+cu118
   environment, exact upstream checkouts, and verified read-only model
   snapshot;
3. H100 NVL `attempt-001` at project commit
   `ea2dace2026523b6a498d634301927ddc335e015` passed the real
   `contentment_05000` forward/backward smoke, all memory and BF16 gates, and
   the finite non-zero LoRA-gradient checks;
4. the accepted report records zero optimizer steps, no adapter/checkpoint
   save, `full_dataset_ready: false`, and
   `formal_training_authorized: false`;
5. Member 1 clean-shell verification and a second verification from a
   replacement Pod attached to the retained volume both passed; and
6. the temporary Pods were stopped after the evidence was copied to the
   shared volume and checked.

The observed compute spend was approximately USD 1.39. The retained volume is
approximately USD 8.40 per month at the recorded rate. No running Pod is part
of the handoff.

Six-member account onboarding is a separate coordination gate. Invitations
for Members 2-6 and verification through another member's own RunPod account
and SSH key are deferred by the project owner. The technical environment is
ready, but cross-account/shared-access completion is not claimed.

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

The completed verification evidence and portable private package are:

```text
/workspace/phase3/reports/member1/verification-h100-001/
/workspace/phase3/reports/member1/verification-shared-volume-001/
/workspace/phase3/reports/member1/packages/
  member1-gpu-acceptance-reports.tar.gz
```

The package SHA-256 is
`244d3f3d9bd417591fde3669f69eb31a3a94b49d495c7d856e50ad8587c47791`.
The checked-in `phase3/reports/member1/attempt-001/README.md` is a redacted
index; the private bundle on the shared volume remains authoritative.

Each later handoff verification is written below the caller's own report root.
The existing attempt records project commit `ea2dace`; verify it from a clean
checkout of exactly that commit. Verification does not rewrite Member 1's
attempt or execute the model. It checks a flat immutable bundle of regular
files, validates the fixture JSON and sidecar against the source locks, and
compares before/after source snapshots. Success and failure outputs are sealed
with checksums. A failure carries a stable `failure_code` and retains completed
checks whenever possible.

The verifier proves that the bundle, pinned inputs, shared reads, and
caller-owned write path work under the supplied `--member-id`. That text value
does not authenticate the RunPod account or SSH key that opened the shell.
Calling the handoff cross-account complete will require private evidence that
Member 2 joined the Team and logged in with Member 2's own SSH key. That check
is deferred. Keep future identity evidence outside Git and outside redacted
reports.

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
