# Venus Emotion-Aware LoRA Pipeline & Model Benchmarking

> 基于 Venus 的情绪感知 LoRA 训练管线与基线模型对比

An engineering portfolio for a six-person University of Sydney capstone project that explored how emotion-aware prompting and parameter-efficient fine-tuning could improve multimodal image-aesthetic guidance and cropping.

## Portfolio scope

This repository is a public, recruiter-facing project summary. It intentionally does **not** redistribute the private course repository, third-party Venus source code, datasets, model weights, generated outputs, client material, or credentials.

| Item | Summary |
|---|---|
| Project context | University of Sydney SOFT3888 capstone, six-person team |
| My roles | Programmer, tester, and customer liaison on the rotating team roster |
| Core stack | Python, PyTorch, Qwen-VL, PEFT/LoRA, RunPod, Linux/Shell, pytest |
| Evaluation | Eval80-v1, 80 images across 8 emotion categories |
| Data pipeline | 480 unique images and 576 Qwen-VL conversations |
| GPU validation | H100 NVL, BF16 forward/backward technical validation |

## Problem and approach

The project studied whether explicit emotion conditioning could improve multimodal aesthetic guidance while preserving reproducibility and evaluation integrity.

The comparison design contained four conditions:

- **A — Official baseline:** the released Venus prompt.
- **B0 — Unstructured emotion prompt:** emotion recognition and guidance requested jointly.
- **B1 — Structured emotion prompt:** the same task with a constrained structured response.
- **C — LoRA condition:** B1 plus a planned emotion-aware LoRA adapter.

A, B0, and B1 each produced 80 outputs and passed 80/80 technical acceptance, for 240 validated outputs in total. Condition C reached data, environment, PEFT injection, parameter, gradient, and GPU runtime validation. Formal optimiser training and adapter checkpoint production were not completed at the documented handoff point.

## Engineering highlights

### Reproducible evaluation workflow

- Built a Python CLI covering preflight checks, frozen-input loading, inference execution, output validation, blind score-sheet preparation, and summary generation.
- Locked model snapshots, generation settings, random seeds, environment versions, and result hashes for repeatable cloud execution.
- Added checks for missing records, duplicate samples, schema violations, label leakage, configuration drift, and unexpected hash changes.
- Archived the exact RunPod/Linux wrappers used for A/B0/B1 execution and technical acceptance.

### Emotion-aware training-data pipeline

- Produced **BridgeTrain-v1** with 400 training images, 80 independent validation images, and 576 structured multimodal conversations.
- Enforced quotas across 8 emotion classes, hash-based deduplication, train/validation separation, and exclusion of every Eval20 and Eval80 image.
- Used JSON-schema-style validation and deterministic packaging to make the handoff auditable.

### LoRA infrastructure validation

- Adapted the Venus/Qwen-VL stack for PEFT LoRA injection while freezing the visual tower.
- Pinned Python, PyTorch, CUDA, model, and upstream-source versions for a reproducible RunPod environment.
- On an H100 NVL, completed a real-image BF16 forward/backward validation and verified:
  - 128 targeted LoRA modules;
  - 3,506,176 trainable parameters;
  - non-zero gradients through the intended trainable path.

This work demonstrates training-pipeline readiness rather than claiming a completed fine-tuned adapter.

## My contribution

My authored work concentrated on the evaluation system, cloud execution workflow, emotion-annotation pipeline, data-release gates, and LoRA environment validation. The private project history attributes 35 commits to Tian Liao, including merged feature work and handoff commits.

Selected evidence is summarised in [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md). The system and experiment flow is documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Skills demonstrated

- Multimodal LLM evaluation and benchmark design
- PEFT/LoRA integration and GPU runtime validation
- Deterministic dataset construction and leakage prevention
- Python CLI development and automated validation
- Linux/RunPod execution, dependency pinning, and reproducibility
- Git/GitHub pull-request collaboration, technical documentation, and team handoff

## Academic and confidentiality note

The original implementation remains in a private university team repository. This portfolio documents my work without exposing assessable source code, teammate-owned implementation, restricted datasets, or third-party assets. See [NOTICE.md](NOTICE.md).

