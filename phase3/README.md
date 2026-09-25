# Phase 3 Member 1 environment handoff

This directory contains the repository-side environment and upstream-integration
work owned by Member 1. The work is on branch `phase3member1-env`, based on
`tlia0262` commit `869a24bb42b83f8b58c6f693a2cae6a84df5ae9d`. All new
implementation lives under `phase3/`; inherited project files remain outside
this change.

Member 1 owns the Python environment with pinned top-level dependencies and a
captured package freeze, the reviewed upstream compatibility patch, the shared
RunPod layout, and one real-image BF16 forward/backward technical smoke. This
scope stops before an optimizer step. It does not include full-data conversion,
formal LoRA settings, training, adapter save/reload, inference, or evaluation.
Member 1 implements no API. Any later callable demo is inference-only. Training
remains CLI-based, and no training, data-upload, or evaluation HTTP API is in
scope.

## Current status

Member 1's technical environment and GPU acceptance completed on 2026-09-25:

- the shared 120 GB Standard Network Volume `phase3-shared-120gb`
  (`bk4fycduml`) is retained in `US-NE-1`; it was mounted at `/workspace` for
  acceptance and must use that mount point on future Pods;
- `/workspace/phase3/envs/venus-phase3` contains Python 3.10.13 and
  PyTorch 2.0.1+cu118, and the exact upstream sources and model snapshot were
  verified and sealed read-only;
- H100 NVL `attempt-001` completed one real-image forward/backward pass and
  passed every runtime gate, with no optimizer step and no adapter save;
- Member 1 clean-shell verification and a second verification from a
  replacement Pod attached to the retained volume both passed; and
- every temporary Pod is stopped, so only the Network Volume continues to
  incur charges.

The accepted runtime and evidence are pinned to project commit
`ea2dace2026523b6a498d634301927ddc335e015`. The observed compute spend was
approximately USD 1.39, and the retained volume is approximately USD 8.40 per
month at the recorded rate. Prices and future GPU capacity must be checked
again before another Pod is started.

Member 1's technical environment handoff is complete. Invitations for Members
2-6 and independent cross-account identity verification are deliberately
deferred, so six-account/shared-access completion is not claimed. The reports
continue to record `full_dataset_ready: false` and
`formal_training_authorized: false`.

## Pinned inputs

| Input | Exact revision |
| --- | --- |
| `PKU-ICST-MIPL/Venus_CVPR2026` | `44e2971b7fe41f88a88c7e0ed8bf4c85eeb0155f` |
| `cognitedata/Qwen-VL-finetune` | `efa37ba284d56192b246d9b4ed5d3668c1abd163` |
| Reviewed Qwen compatibility patch | SHA-256 `be9e14a60f4f2189a108aa1f46a7384616bfdd527ca6c0a41a4c175a92172eba` |
| `popo28/Venus-Q-Stage1` | `0f5c00c8d07ba889e9c5d12f828129dc322aae6a` |

The exact model snapshot lives read-only at
`/workspace/models/Venus-Q-Stage1`; it is never copied into Git. Bootstrap can
download missing files directly from the pinned Hugging Face revision, verify
all 22 locked files and their hashes, write exact provenance, and remove write
bits from the snapshot, marker, and model root.

## Reproduction and later verification entry points

The following commands are retained for recovery, reproduction, or a later
acceptance attempt. They are not instructions to create another volume or
rerun the already-passing `attempt-001`. Run them from a clean SSH login shell
at the repository root on a Pod that mounts the retained volume. The standard
handoff uses `/workspace/phase3` as its runtime root and
`/workspace/phase3/envs/venus-phase3` as its shared environment. Low-level CLI
path overrides are for isolated tests or recovery only.

```bash
bash phase3/scripts/bootstrap.sh \
  --member-id member1 \
  --workspace-root /workspace/phase3 \
  --model-path /workspace/models/Venus-Q-Stage1 \
  --download-model-if-missing
```

If a rerun is justified, preserve `attempt-001` and use a new attempt number:

```bash
bash phase3/scripts/member1_acceptance.sh \
  --member-id member1 \
  --image /workspace/phase3/smoke/contentment_05000.jpg \
  --report-dir /workspace/phase3/reports/member1/attempt-002
```

The existing immutable attempt records commit `ea2dace`. Verification of that
attempt must run from a clean checkout of exactly that commit; later handoff
maintenance commits do not replace the recorded runtime source. The verifier
reads the attempt without rerunning the smoke and writes a new report below the
caller's own report directory:

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

This command validates the bundle and filesystem access associated with the
supplied `--member-id`; it cannot authenticate which RunPod account or SSH key
opened the shell. A separate-account handoff also requires private Team and SSH
access evidence outside Git.

## Detailed documents

- [`docs/MEMBER1_RUNBOOK.md`](docs/MEMBER1_RUNBOOK.md) gives the ordered
  bootstrap, acceptance, verification, recovery, and timing procedure.
- [`docs/MEMBER1_HANDOFF.md`](docs/MEMBER1_HANDOFF.md) defines the ownership
  boundary and the evidence later members may rely on.
- [`docs/RUNPOD_TEAM_SETUP.md`](docs/RUNPOD_TEAM_SETUP.md) records the current
  cloud state and the planned multi-account provisioning.
- [`docs/TRAINING_INPUT_CONTRACT.md`](docs/TRAINING_INPUT_CONTRACT.md) records
  the upstream conversation shape and why the full dataset is not yet ready.
- [`docs/EXPERIMENT_RECORD_TEMPLATE.md`](docs/EXPERIMENT_RECORD_TEMPLATE.md)
  gives later training owners one consistent experiment record.
- [`docs/TEAM_HANDOFF_CHECKLIST.md`](docs/TEAM_HANDOFF_CHECKLIST.md) lists the
  checks every member should complete before using or handing off shared work.

The one-record fixture proves only that the controlled stack can consume a real
image and complete forward/backward with BF16 floating base parameters, BF16
input entering the PEFT wrapper, and FP32 trainable LoRA adapter parameters. It
does not assert that every internal LoRA matrix operation executes in BF16. The
fixture manifest, aggregate handoff report, and handoff verification report
deliberately retain `full_dataset_ready: false` and
`formal_training_authorized: false`.
