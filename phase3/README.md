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

The repository-side scripts, pins, schemas, and instructions are ready for
review and local static checks. This does not make the cloud/runtime handoff
complete. As observed on 2026-09-23, the RunPod account still had two stopped
legacy A40 Pods, each with its own 120 GB Pod Volume Disk, and no Network
Volume. The planned normal Team, 120 GB Standard Network Volume, shared Secure
Cloud Pod, data migration, team invitations, and real GPU acceptance remain
pending runtime actions.

## Pinned inputs

| Input | Exact revision |
| --- | --- |
| `PKU-ICST-MIPL/Venus_CVPR2026` | `44e2971b7fe41f88a88c7e0ed8bf4c85eeb0155f` |
| `cognitedata/Qwen-VL-finetune` | `efa37ba284d56192b246d9b4ed5d3668c1abd163` |
| Reviewed Qwen compatibility patch | SHA-256 `be9e14a60f4f2189a108aa1f46a7384616bfdd527ca6c0a41a4c175a92172eba` |
| `popo28/Venus-Q-Stage1` | `0f5c00c8d07ba889e9c5d12f828129dc322aae6a` |

The existing model snapshot is reused read-only at
`/workspace/models/Venus-Q-Stage1`; it is never copied into Git.

## One-command entry points

Run these commands from a clean SSH login shell at the repository root on the
selected RunPod host. The standard handoff uses `/workspace/phase3` as its
runtime root and `/workspace/phase3/envs/venus-phase3` as its shared
environment. Low-level CLI path overrides are for isolated tests or recovery
only. The formal acceptance and handoff-verification wrappers enforce the
standard runtime and model paths.

```bash
bash phase3/scripts/bootstrap.sh \
  --member-id member1 \
  --workspace-root /workspace/phase3 \
  --model-path /workspace/models/Venus-Q-Stage1
```

After the real `contentment_05000` image is present, use a new attempt number:

```bash
bash phase3/scripts/member1_acceptance.sh \
  --member-id member1 \
  --image /workspace/phase3/smoke/contentment_05000.jpg \
  --report-dir /workspace/phase3/reports/member1/attempt-001
```

Verification reads that immutable attempt, does not rerun the smoke, and writes
a new report below the caller's own report directory:

```bash
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
  cloud state and the planned multi-account migration.
- [`docs/TRAINING_INPUT_CONTRACT.md`](docs/TRAINING_INPUT_CONTRACT.md) records
  the upstream conversation shape and why the full dataset is not yet ready.

The one-record fixture proves only that the controlled stack can consume a real
image and complete BF16 forward/backward. The fixture manifest, aggregate
handoff report, and handoff verification report deliberately retain
`full_dataset_ready: false` and `formal_training_authorized: false`.
