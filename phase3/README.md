# Phase 3 handoff

> **Public-portfolio boundary:** this is a historical source snapshot of the
> team's Phase 3 engineering. The private dataset release, images, reports,
> reviewed Qwen compatibility patch, model files, and cloud workspace are not
> distributed here. The public tests remain runnable, but bootstrap, dataset
> rebuild, GPU acceptance, and end-to-end training commands require those
> separately authorised inputs and will not run unchanged from this checkout.

This directory contains the shared Phase 3 environment, dataset, and handoff
contracts. Member 1 owns the pinned runtime and upstream integration. Member 2
owns the deterministic BridgeTrain-v1 conversion, validation, and training
input release. Member 3A implements and runs the pilot and formal training;
Member 3B reviews the pilot, authorizes the formal run within Member 3, performs
the final technical acceptance, and selects the final checkpoint. Member 4
owns adapter inference and Condition C integration. Member 5 independently
runs the formal Eval80, and Member 6 owns the minimal inference API.

## Member 2 dataset status

BridgeTrain-v1 has passed machine validation against the original EmoSet
archive and both frozen evaluation sets. It contains 480 unique images and 576
Qwen-VL conversations: 480 classification samples plus 96 balanced
joint-guidance samples. In the private project, the Git-safe release artefacts
were stored under `releases/bridge-train-v1/`. Both those derived records and
the 480 source images are intentionally excluded from this public portfolio.

Member 1 approved all twelve post-audit guidance corrections on 2026-09-26.
The rebuilt release records `full_dataset_ready: true`, no pending human-review
records, and `formal_training_authorized: false`. See
`docs/MEMBER2_HANDOFF.md` for the exact rebuild, verification, and
shared-volume steps.

The dataset-level `formal_training_authorized: false` value remains unchanged:
it separates data readiness from execution. Member 3B records the internal
pilot decision at `phase3/reviews/member3/pilot-review-v1.json`; a passing
record authorizes Member 3A to run the exact reviewed formal configuration.
After formal training, Member 3B records the selected checkpoint, adapter hash,
reload result, and technical acceptance at
`phase3/reviews/member3/training-acceptance-v1.json`. No approval from Member
1, Member 4, or another team member is required for the Member 3 training loop.

Member 1 owns the Python environment with pinned top-level dependencies and a
captured package freeze, the reviewed upstream compatibility patch, the shared
RunPod layout, and one real-image BF16 forward/backward technical smoke. This
scope stops before an optimizer step. It does not include full-data conversion,
formal LoRA settings, training, adapter save/reload, inference, or evaluation.
Member 1 implements no API. Any later callable demo is inference-only. Training
remains CLI-based, and no training, data-upload, or evaluation HTTP API is in
scope.

## Current status

Member 1's technical environment and GPU acceptance completed on 2026-09-25.
The following bullets record historical state as of that date; this repository
has no access to and makes no claim about the current cloud state:

- a private 120 GB Standard Network Volume in `US-NE-1` was mounted at
  `/workspace` for acceptance;
- `/workspace/phase3/envs/venus-phase3` contains Python 3.10.13 and
  PyTorch 2.0.1+cu118, and the exact upstream sources and model snapshot were
  verified and sealed read-only;
- H100 NVL `attempt-001` completed one real-image forward/backward pass and
  passed every runtime gate, with no optimizer step and no adapter save;
- Member 1 clean-shell verification and a second verification from a
  replacement Pod attached to the retained volume both passed; and
- every temporary Pod was recorded as stopped at the handoff point.

The accepted runtime and evidence are pinned to project commit
`ea2dace2026523b6a498d634301927ddc335e015`. The private project recorded an
observed compute spend of approximately USD 1.39 and an estimated storage rate
of USD 8.40 per month at that time. These are historical observations, not a
statement of current billing or resource availability.

Before each paid pilot and formal training run, Member 3A must ask the user to
set the allowed GPU, maximum hourly price, total budget for that stage, and the
response to OOM or a required time extension. The user and the Member 3 agents
decide the training limits; the documents do not assume them. All Phase 3
non-training costs have a cumulative USD 20 ceiling, including the existing
H100 smoke cost, retained-volume charges, and future non-training work.

Member 1's technical environment handoff is complete. Invitations for Members
2-6 and independent cross-account identity verification are deliberately
deferred, so six-account/shared-access completion is not claimed. The immutable
Member 1 smoke reports continue to record `full_dataset_ready: false` because
they predate and do not authorize the full release. The current signed
BridgeTrain-v1 summary records `full_dataset_ready: true`; both the historical
smoke and current dataset release retain `formal_training_authorized: false`.

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

## Historical reproduction and verification entry points

The following commands are retained for recovery, reproduction, or a later
acceptance attempt inside the authorised private environment. They are not
self-contained public quick-start commands: the excluded Qwen patch, private
dataset fixture, model snapshot, and cloud workspace are prerequisites. They
are not instructions to create another volume or rerun the already-passing
`attempt-001`. The standard handoff used `/workspace/phase3` as its runtime root and
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
- [`docs/RUNPOD_TEAM_SETUP.md`](docs/RUNPOD_TEAM_SETUP.md) records the
  2026-09-25 historical cloud state and conditional future access policy.
- [`docs/TRAINING_INPUT_CONTRACT.md`](docs/TRAINING_INPUT_CONTRACT.md) records
  the accepted upstream conversation shape and BridgeTrain-v1 release gates.
- [`docs/EXPERIMENT_RECORD_TEMPLATE.md`](docs/EXPERIMENT_RECORD_TEMPLATE.md)
  gives later training owners one consistent experiment record.
- [`docs/TEAM_HANDOFF_CHECKLIST.md`](docs/TEAM_HANDOFF_CHECKLIST.md) lists the
  checks every member should complete before using or handing off shared work.
- [`docs/MEMBER2_HANDOFF.md`](docs/MEMBER2_HANDOFF.md) defines the complete
  BridgeTrain-v1 release, rebuild, validation, and shared-volume handoff.
- `Docs/phase3/MEMBER3_START_HERE_EN.md` was the private project's canonical
  Member 3 start document and is not distributed here.
- `Docs/phase3/PHASE3_LORA_WORKFLOW_EN.docx` was the private project's
  canonical project-level workflow and is not distributed here.

In the private repository, `Docs/phase3/` contained assessment/project workflow
documents, while `phase3/docs/` contained the technical contracts and runbooks
retained in this portfolio. References to the former are provenance markers,
not public links.

The one-record fixture proves only that the controlled stack can consume a real
image and complete forward/backward with BF16 floating base parameters, BF16
input entering the PEFT wrapper, and FP32 trainable LoRA adapter parameters. It
does not assert that every internal LoRA matrix operation executes in BF16. The
fixture manifest, aggregate handoff report, and handoff verification report
deliberately retain `full_dataset_ready: false` and
`formal_training_authorized: false`.
