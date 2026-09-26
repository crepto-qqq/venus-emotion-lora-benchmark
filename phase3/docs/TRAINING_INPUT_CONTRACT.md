# Training input contract

## Upstream loader shape

The pinned Qwen-VL fine-tuning path expects a JSON array. Each sample contains
exactly two conversation turns. The user turn contains one image tag followed
by the task prompt; the assistant turn contains the supervised target.

```json
[
  {
    "id": "contentment_05000__classification",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>/workspace/phase3/data/bridge-train-v1/image/contentment/contentment_05000.jpg</img>\n<task prompt>"
      },
      {
        "from": "assistant",
        "value": "contentment"
      }
    ]
  }
]
```

The path inside `<img>...</img>` must resolve on the Pod. Images stay outside
Git at `/workspace/phase3/data/bridge-train-v1/image/`. The Git-safe sidecar
manifest preserves each source ID, class, split, image and annotation hashes,
source-record identity, review state, task type, and prompt identity.

The source JSONL `instruction` field is annotation-generation metadata. It
discloses the assigned EmoSet label and must never become a training user
prompt. BridgeTrain-v1 creates its conversations from the two frozen prompts
in `phase3/configs/bridge-train-v1.json`.

## BridgeTrain-v1 protocol

BridgeTrain-v1 uses all 480 unique source images while keeping the original
50/10 per-class train/validation split:

| Split | Unique images | Classification conversations | Joint-guidance conversations | Total conversations |
| --- | ---: | ---: | ---: | ---: |
| Train | 400 | 400 | 80 | 480 |
| Validation | 80 | 80 | 16 | 96 |
| Total | 480 | 480 | 96 | 576 |

Every image produces one label-only emotion-classification conversation. A
frozen, balanced subset of 96 reviewed images also produces one structured
joint-guidance conversation using the exact B1 prompt. Those 96 images appear
twice by design, once per task; no image ID or image hash may appear under a
different source ID or in both splits.

Classification targets come from the immutable EmoSet archive label and do not
depend on the generated guidance text. Joint-guidance targets must have
`review_status` equal to `accepted` or `edited`, must have
`label_supported: true`, must match the assigned class in both emotion fields,
and must pass the text-integrity checks. A guidance-ineligible record remains
usable for classification.

## Release gates

The builder and independent verifier enforce:

1. exact eight-class and split quotas;
2. archive image and annotation presence and SHA-256 equality;
3. source ID, class, path, and archive-annotation agreement;
4. no duplicate IDs, paths, or image bytes;
5. train/validation disjointness by ID and image hash;
6. frozen Eval20 and Eval80 exclusion by both ID and image hash;
7. exact prompt text, prompt hashes, target structure, and task pairing;
8. artifact hashes and summary-count consistency; and
9. a separate human-signoff lock for post-audit text corrections.

The current release has passed all machine gates. Twelve selected guidance
records were corrected after the second image audit and approved by Member 1
on 2026-09-26. The approval list is now empty, the signed release was rebuilt,
and the summary records `full_dataset_ready: true`. The dataset schema keeps
`formal_training_authorized: false` as a permanent separation between dataset
readiness and training execution. Member 3B reviews Member 3A's pilot evidence
and records the internal formal-run decision in
`phase3/reviews/member3/pilot-review-v1.json`, bound to the exact formal LoRA
configuration. No approval from another team member is required.

## Member 1 fixture boundary

Member 1's one-record `contentment_05000` fixture proves that the pinned model,
environment, and PEFT wrapper complete a real-image BF16 forward/backward pass.
It is not part of the 576-conversation release and does not replace Member 2's
dataset validation.

Training remains command-line based. The simple API planned for the final demo
wraps inference only; no training, upload, or evaluation API is required.
