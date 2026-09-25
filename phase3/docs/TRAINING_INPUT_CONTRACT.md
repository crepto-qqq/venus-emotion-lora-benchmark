# Training input contract

## Upstream loader shape

The pinned Qwen-VL fine-tuning path expects one JSON **array**, not JSONL. Each
sample has a `conversations` list. The user turn includes the image using an
`<img>...</img>` tag, followed by the instruction; the assistant turn contains
the supervised target. The minimum project mapping is:

```json
[
  {
    "id": "contentment_05000",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>/workspace/phase3/smoke/contentment_05000.jpg</img>\n<instruction>"
      },
      {
        "from": "assistant",
        "value": "<target_response>"
      }
    ]
  }
]
```

The image path must resolve on the Pod. Preserve the full instruction and
target text exactly; do not infer a new label, strip the structured sections,
or embed image bytes. Conversion must also preserve source ID, emotion, split,
image hash, review status, and provenance in a sidecar manifest when the
upstream sample shape has no place for those project fields.

The current project files under `data/datasets/` are line-delimited project
records with fields such as `source_image_id`, `emotion`, `image_relpath`,
`instruction`, `target_response`, `split`, and `provenance`. They are useful
source records, but they are not direct input to the upstream loader because
they are JSONL and do not contain the required `conversations` structure or
image tag.

## Member 1 fixture

Member 1 creates exactly one runtime conversion for `contentment_05000`. Its
purpose is to exercise the real-image forward/backward path with BF16 floating
base parameters, BF16 input entering the PEFT wrapper, and FP32 trainable LoRA
adapter parameters. It lives under `/workspace/phase3/smoke/`, stays outside
Git, and must not be presented as the full converter or a training dataset.

The builder also writes a sidecar manifest for the source record, annotation,
image hashes, split, and available Eval20/Eval80 disjointness evidence. The
source record and image are fixed by `configs/member1-smoke.json`; substituting
another accepted example would create a different test and is not allowed in
Member 1's acceptance. The locked image is 126,584 bytes, 720 x 480 pixels, and
has SHA-256
`f4552a57efd8ff17e0a7a9fe28e1e94e98401cc5a5ace21ee6c246d74766d082`.
The immutable attempt includes both `technical-fixture.json` and
`technical-fixture-manifest.json`; both are covered by the fixture sidecar and
the complete attempt checksums.

Member 2 owns the complete deterministic converter, its validation, all
accepted-record filtering, collision/duplicate checks, and the final training
and validation manifests. Member 1's acceptance report must therefore retain
`full_dataset_ready: false`.

## Current 480-record snapshot

The latest pulled snapshot contains 480 records. Its record count is not a
readiness signal:

```bash
python -m src.datasets.annotation_pipeline validate \
  --output-root data/datasets \
  --require-accepted
```

| Class | Current state | Consequence |
| --- | --- | --- |
| `amusement` | 60 records with the intended 50 train / 10 validation split, but none has an `accepted` or `edited` review status | 60 review-status findings under `--require-accepted`; exclude them from released conversion |
| `excitement` | 60 records use the legacy target and provenance shape | 60 target-structure findings plus 300 missing-provenance-field findings |
| Remaining classes | Present in the pulled snapshot | Still require final converter-wide validation and leakage checks |

With `--require-accepted`, the validator reports 420 findings affecting 120
records: 60 amusement review-status findings, 60 excitement target-structure
findings, and 300 excitement missing-provenance-field findings. It does not mean
there are 420 bad records: each excitement record contributes six findings.
These data issues do not block the isolated Member 1 fixture, but the complete
snapshot is not training-ready.

A separate semantic inspection found three exceptions that the 420-finding
breakdown does not count separately:

- `excitement_05052` records `emotion: excitement`, while the target describes
  contentment;
- `excitement_05058` records `emotion: excitement`, while the target describes
  amusement;
- `excitement_05016` remains in the excitement file and ID range, while its
  record emotion and target are contentment. The validator consequently counts
  11 contentment validation records and 9 excitement validation records.

Running validation with both `--require-accepted` and `--require-complete` adds
one `contentment:validation` count finding, so the total becomes 421. The
current `configs/datasets/emotion_annotation.json` lists only `sadness`, `awe`,
and `contentment` in `initial_emotions`. The completeness option therefore
checks expected counts for only those three classes; it is not proof that all
eight classes are complete.

## Converter acceptance owned by Member 2

Before `full_dataset_ready` can become true, the complete converter must:

1. accept only human-reviewed `accepted` or `edited` records;
2. enforce the assigned emotion in the target and reject the known mismatch;
3. create a JSON array of image-bearing user/assistant conversations;
4. resolve every image path and verify it against recorded hashes;
5. preserve the intended train/validation allocation and exclude all frozen
   Eval20 and Eval80 images;
6. reject duplicate IDs, duplicate images, missing fields, empty targets, and
   unsupported labels;
7. write a redacted summary with input/output counts and hashes.

Member 1 implements no API. Any later callable demo is inference-only. Training
remains CLI-based, and no training, data-upload, or evaluation HTTP API is in
scope.
