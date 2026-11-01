# Member 3 formal LoRA run — 10 October 2026

Status: formal training and three clean-process BridgeTrain validation runs completed and independently checked against downloaded files. Selected checkpoint 180 is a technical recommendation pending Member 3B acknowledgment; this is not a fabricated independent review or formal Eval80 result.

## Executed run

- H100 SXM 80 GB; existing pinned CUDA/Python environment, base snapshot and frozen release preserved.
- BF16 Venus Stage 1 with rank 8, alpha 16, dropout 0.05 LoRA; learning rate 2e-4, seed 42.
- 480 training conversations (400 classification + 80 joint guidance), 400 unique training images; 3 epochs, 1440 microbatches, 180 optimizer steps; batch 1 / accumulation 8, maximum sequence length 2048.
- Observed steps: 180; observed microbatches: 1440. Training-only peak allocated VRAM: 22.330 GiB; reserved: 24.189 GiB. Training window: 1000.2 seconds (excludes model load and validation).
- Checkpoints 60, 120 and 180 each saved, hashed, downloaded and loaded in separate validation processes. Frozen visual backbone and scoped trainable parameters are verified by telemetry.
- Two representative classification/joint-guidance forward/backward checks passed before training, with zero optimizer steps and unchanged adapter parameters.
- Interrupted formal3 (approximately 110/180 optimizer steps) and formal4 (before optimizer) were stopped by heartbeat transport protection. Their evidence and reservations are retained; neither is counted as completed training. The full successful attempt uses a persistent terminal heartbeat with the same 30-second cloud freshness check and an independent absolute-deadline Pod watchdog.
- An earlier input-check failure incorrectly required pilot_subset in a formal config. Fixed only select_samples in check_lora_model_inputs.py, added a regression test; 10 input-check tests passed. Failed pre-optimizer attempts are retained, not counted as successful training.

## Same validation for all checkpoints

All 96 frozen BridgeTrain validation conversations: 80 classification + 16 joint guidance on 80 independent images. Eval20 and Eval80 excluded. Greedy generation with max_new_tokens=512. Selection: lowest token-weighted validation loss; ties use classification, guidance format, then earlier checkpoint.

| Step | Token-weighted loss | Classification exact match | Guidance format valid |
| --- | --- | --- | --- |
| 60 | 1.385515 | 58/80 (72.5%) | 16/16 |
| 120 | 1.301796 | 61/80 (76.2%) | 16/16 |
| 180 | 1.300244 | 62/80 (77.5%) | 16/16 |

Recommended checkpoint: **180**. Guidance format validity measures parsing, not emotional grounding or aesthetic quality. Classification and guidance are separate outcomes; no combined headline score.

| Class | Correct/total |
| --- | --- |
| amusement | 6/10 |
| anger | 10/10 |
| awe | 5/10 |
| contentment | 6/10 |
| disgust | 10/10 |
| excitement | 8/10 |
| fear | 9/10 |
| sadness | 8/10 |

Full confusion matrices, raw predictions, individual losses and adapter hashes are in checkpoint-comparison.json and each validation directory. Technical quality findings are in the private handoff package GUIDANCE_QUALITY_REVIEW_CN.md; this is nonblind AI-assisted development review, not formal human scoring.

## Member 4 handoff

Use the included adapter directory with the pinned Venus Stage 1 base. Do not load it as a standalone full model, merge it as a required product, change the tokenizer/base revision, add quantization/offload, or alter A/B0/B1 behavior.

The three validations provide actual clean-process reload + inference evidence. Canonical training-acceptance-v1.json must still be acknowledged by the responsible Member 3B, using these exact hashes. This recommendation does not impersonate Jackie.

Formal Condition C must inherit the frozen B1 prompt and the original Eval80 generation protocol in configs/phase1/eval80.json: do_sample=true, seed=1234, top_p=0.3, top_k=0, max_new_tokens=512, max_window_size=6144. Do not substitute the greedy validation settings. Member 5 owns independent Eval80 and anonymous randomized five-dimension guidance scoring; Member 6 owns API/demo work.

## Reproducibility and limits

Actual bound config, interface/inventory, session records, training attempt, telemetry, checkpoint state and raw validation outputs are preserved in the private local evidence package. Adapter and report file hashes are supplied in the package manifest. No token, temporary API key, base weights, image corpus or shared environment is included.

The pinned training source creates PEFT adapters before Trainer initialization. Seed 42 is supplied to Trainer, but explicit pre-adapter seeding is absent in this executed source/wrapper. Do not claim fully controlled adapter initialization or bitwise reproducibility; see seed-initialization-audit.json. No running source was changed.

The pilots select a learning rate; this formal comparison selects a checkpoint using BridgeTrain validation only. Neither establishes final system accuracy, an improvement over B1, or independent human guidance scores. No extra frozen comparison was run, as agreed.
