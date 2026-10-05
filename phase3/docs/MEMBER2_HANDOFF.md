# Member 2 dataset handoff

## Delivered scope

Member 2 delivers the deterministic BridgeTrain-v1 dataset boundary between
the reviewed EmoSet records and the pinned Qwen-VL fine-tuning loader. The
release contains 480 unique images represented by 576 conversations: 480
label-only classification samples and 96 structured joint-guidance samples.

The Git-safe release is in `phase3/releases/bridge-train-v1/`:

- `bridge-train-v1.train.json`: 480 Qwen-VL conversations;
- `bridge-train-v1.validation.json`: 96 Qwen-VL conversations;
- `bridge-train-v1.manifest.jsonl`: one provenance row per conversation;
- `bridge-train-v1.rejections.jsonl`: records that are ineligible for the
  joint-guidance task, while remaining available for classification; and
- `bridge-train-v1.summary.json`: counts, readiness state, prompt hashes, and
  artifact hashes.

The source images are intentionally excluded from Git. The locally verified
runtime tree contains 480 files under
`tmp/bridge-train-v1-runtime/image/<emotion>/` and must be copied to the shared
volume at `/workspace/phase3/data/bridge-train-v1/image/<emotion>/` before a
GPU training run. The signed Git-safe release itself is complete; deployment
of the private image tree is a separate runtime preflight.

## Rebuild and verify

Run these commands from the repository root. Set `ARCHIVE` to the local
`EmoSet-118K.zip` path.

```bash
python -m src.datasets.bridge_train \
  --archive "$ARCHIVE" \
  --output-dir tmp/bridge-train-v1-release \
  --extract-images tmp/bridge-train-v1-runtime \
  --overwrite

python phase3/tools/verify_training_release.py \
  --release-dir tmp/bridge-train-v1-release \
  --archive "$ARCHIVE" \
  --extracted-image-root tmp/bridge-train-v1-runtime
```

The first command validates the source JSONL records, the archive, both frozen
evaluation sets, quotas, targets, and prompt locks before writing any release
file. The second command reopens the written artifacts and independently checks
their hashes, structure, counts, archive members, and extracted images.

`phase3/tools/migrate_excitement_records.py` records the deterministic legacy
excitement migration, and `phase3/tools/apply_guidance_corrections.py` records
the exact post-audit text corrections. Both tools are idempotent. Run them
without `--apply` to validate the checked-in state; use `--apply` only when
reconstructing those source changes from an older checkout.

The older annotation-wide validator still reports ten findings when invoked
with `--require-accepted`: four deliberately rejected amusement records and
three deliberately rejected excitement targets, each reported for both label
mismatch and review state. These seven source images are kept for their valid
archive-label classification samples and are excluded from joint guidance.
BridgeTrain-v1's task-aware builder and verifier are the release authority.

## Shared-volume handoff

Copy the contents of `tmp/bridge-train-v1-runtime/` into
`/workspace/phase3/data/bridge-train-v1/` on the retained RunPod network
volume. From a checkout on a Pod, verify the deployed image tree with:

```bash
python phase3/tools/verify_training_release.py \
  --release-dir phase3/releases/bridge-train-v1 \
  --extracted-image-root /workspace/phase3/data/bridge-train-v1
```

This step requires storage access but does not require a GPU. The JSON files
already point to the shared-volume path expected by the training loader.

## Current readiness state

All machine validation passes. Member 1 approved all twelve image-audited
post-review guidance corrections on 2026-09-26. The approval is recorded in
`phase3/configs/bridge-train-v1.json`, the release was rebuilt, and its summary
now records `full_dataset_ready: true`,
`human_review_signoff_pending: false`, and zero pending review records.

`formal_training_authorized` remains false in the dataset release by design.
It separates dataset readiness from training execution and is not rewritten by
the later training workflow. Member 3A owns both the pilot and formal training.
After Member 3A completes the pilot, Member 3B reviews its evidence and writes
`phase3/reviews/member3/pilot-review-v1.json`; a passing record authorizes the
exact formal configuration within Member 3. No outside team-member approval is
required. After the formal run, Member 3B selects the final checkpoint and
writes `phase3/reviews/member3/training-acceptance-v1.json` before handing the
accepted adapter to Member 4. Member 4 owns adapter inference and Condition C
integration, Member 5 independently runs the formal Eval80, and Member 6 owns
the minimal inference API.

Before each paid pilot and formal run, Member 3A must ask the user to set that
stage's GPU, maximum hourly price, total budget, and response to OOM or a
required time extension. Training has no assumed document-level cost ceiling.
The separate cumulative ceiling for all Phase 3 non-training costs is USD 20;
it includes the existing H100 smoke, retained-volume charges, and future
non-training work.

The exact completion record and repository-only starting procedure for Member
3 are in the canonical English project documents:

- [`MEMBER3_START_HERE_EN.md`](../../Docs/phase3/MEMBER3_START_HERE_EN.md)
- [`PHASE3_LORA_WORKFLOW_EN.docx`](../../Docs/phase3/PHASE3_LORA_WORKFLOW_EN.docx)
