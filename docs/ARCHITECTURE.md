# Architecture and experiment flow

## End-to-end view

```mermaid
flowchart LR
    S[Approved image sources] --> A[Emotion annotation pipeline]
    A --> Q[Quota and schema validation]
    Q --> D[Hash deduplication]
    D --> L[Eval20 and Eval80 leakage checks]
    L --> B[BridgeTrain-v1 release]
    B --> T[Train: 400 images]
    B --> V[Validation: 80 images]
    T --> P[Qwen-VL plus PEFT/LoRA preparation]
    V --> P
    P --> G[H100 NVL BF16 forward/backward validation]

    E[Eval80-v1: 80 frozen images] --> C1[A: official baseline]
    E --> C2[B0: unstructured emotion prompt]
    E --> C3[B1: structured emotion prompt]
    C1 --> O[Output and schema validation]
    C2 --> O
    C3 --> O
    O --> R[Anonymous scoring package]
```

## Evaluation contract

Eval80-v1 was treated as a frozen contract rather than an informal image folder:

1. Validate image identities, labels, category quotas, and hashes.
2. Pin the model snapshot, prompt version, generation parameters, and random seed.
3. Run one condition through the same CLI entry point.
4. Validate exactly 80 structurally complete outputs.
5. Preserve configuration, logs, outputs, and acceptance evidence together.
6. Randomise conditions before human scoring to reduce evaluator bias.

## LoRA readiness boundary

The pipeline reached all of the following:

- deterministic training and validation data release;
- PEFT LoRA injection;
- visual-tower freezing;
- targeted-module and trainable-parameter validation;
- real-image BF16 forward and backward execution;
- non-zero-gradient verification.

It did **not** reach optimiser updates, a completed training run, or a saved adapter checkpoint. This boundary is stated explicitly so the portfolio remains technically accurate.

