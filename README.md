# Venus Emotion Aware LoRA Development and Model Comparison

> 基于 Venus 的情绪 LoRA Adapter 开发与原模型对比评测

A completed six-person University of Sydney SOFT3888 project exploring emotion-aware multimodal aesthetic guidance and cropping. This repository publishes the team's retained engineering source and filtered development history, with third-party source, image datasets, model weights and private review records excluded.

Refreshed on **2 November 2026** from the school's committed `main` at [`f1b6bca`](https://github.sydney.edu.au/tsin0599/SOFT3888_TU12_01_P15/commit/f1b6bcaa66e6da7cfccfc429051e6bcf873f87d2). The school link requires access. Unlike the earlier portfolio snapshot, **formal LoRA training, checkpoint validation, Stage C inference and final team evaluation have now been completed**. See [source provenance](SOURCE_ORIGIN.md).

## What was built

- A Python CLI for frozen-input inference, output validation, benchmark aggregation and scoring workflows.
- Emotion-annotation and BridgeTrain preparation with quotas, deduplication, schemas and evaluation-set isolation.
- PEFT/LoRA preparation, formal training, telemetry, checkpoint validation and bounded cloud-resource controls.
- Condition C adapter inference and image-aware quality/preference evaluation tools.
- Reproducible environment definitions, handover contracts, tests and engineering runbooks.

The core stack is Python, PyTorch, Qwen-VL, PEFT/LoRA, Linux/Shell and RunPod. Venus, Qwen and EmoSet are upstream systems/data, not original team work.

## Completed training and validation

The successful formal run used H100 SXM 80 GB, BF16 and LoRA rank 8 / alpha 16 / dropout 0.05. It completed **3 epochs, 1,440 microbatches and 180 optimiser steps** on 480 training conversations from 400 unique images. The complete private BridgeTrain release contains 576 conversations and 480 images including validation.

Checkpoints 60, 120 and 180 were saved and loaded in separate validation processes. The selected checkpoint 180 achieved **62/80 classification exact matches** and **16/16 valid guidance formats** on the independent BridgeTrain validation set. Format validity is not a measure of guidance quality. Its acceptance record was completed on 19 October by Tian after taking over the review role; this is not presented as an independent teammate's sign-off.

The [formal-run record](phase3/docs/MEMBER3_FORMAL_RESULTS_2026-10-10.md) retains the contemporaneous results and caveats. Its then-pending acceptance status is historical. Adapter initialisation was not fully seed-controlled in that executed workflow, so bitwise reproducibility is not claimed.

## Comparison and final evaluation

| Condition | Definition |
|---|---|
| A | Released Venus prompt |
| B0 | Unstructured joint emotion-and-guidance prompt |
| B1 | Structured joint emotion-and-guidance prompt |
| C | B1 prompt and generation protocol plus the selected LoRA adapter |

Emotion recognition uses the fixed 80-image Eval80 set. The final guidance-scoring scope uses 78 images, with two predefined exclusions applying only to guidance.

Six team reviewers scored **312 answers** across the four conditions. On this fixed set, C averaged **8.29/10**, compared with **5.04/10** for B1; the paired difference calculated before rounding was **+3.26**. C scored higher on 68 images, tied on 7 and lower on 3.

These are small-sample, internal team ratings, with one reviewer assigned per image, not an independent external benchmark or statistical-significance claim. C also had more hallucination flags (**26/78 versus 11/78** for B1). The result supports improved usefulness/emotion-focused guidance on this set, not uniformly better factual grounding. Automated judge scores remain a separate analysis and are not averaged with human ratings or recognition accuracy.

## Source layout

| Path | Purpose |
|---|---|
| `src/eval80/` | Inference, validation, recognition/quality/preference benchmarks and aggregation |
| `src/datasets/` | Annotation processing and BridgeTrain construction |
| `src/phase1/` | Baseline workflow, image cropping and manifests |
| `phase3/tools/` | LoRA training, telemetry, checkpoint validation, environment and resource guards |
| `phase3/contracts/` | Data, training and runtime schemas |
| `tests/`, `phase3/tests/` | Unit, integration and historical contract tests |
| `environment/`, `phase3/environment/` | Dependency and runtime definitions |

## Public checkout and tests

This is a reviewed source portfolio, not a self-contained redistribution of the research environment. Obtain upstream code/models and licensed data separately under their own terms. Private manifests, images, checkpoints, raw outputs and reviewer sheets are not supplied. Historical scripts that refer to those inputs will not run unchanged from this checkout alone.

On 2 November 2026, the following **76 self-contained tests passed** on Windows with the local Python environment, without failures or skips:

```text
python -B -m unittest tests.datasets.test_annotation_pipeline tests.datasets.test_bridge_train tests.eval80.test_benchmark_status tests.eval80.test_cli tests.eval80.test_compare_recognition tests.eval80.test_core tests.eval80.test_preference_final_overview tests.eval80.test_preference_overview tests.eval80.test_preference_recovery tests.eval80.test_validation tests.phase1.test_crop tests.phase1.test_manifest tests.phase1.test_scoring tests.phase1.test_selection phase3.tests.test_formal_bundle phase3.tests.test_runtime_contracts -v
```

Full historical discovery was also attempted and **does not pass in this public checkout**. The core discovery reported 100 tests with 22 error events; Phase 3 reported 97 tests, 2 failures, 41 error events and 10 skips. Those suites depend on omitted private inputs/evidence, upstream fixtures, PyTorch and/or POSIX runtime features. Setup/teardown errors mean these are the runner's reported event counts, not a simple subtraction of failed test cases. No private fixture was fabricated and no historical test was changed to manufacture a pass. This refresh did not rerun GPU training or live cloud workloads.

## My contribution and team authorship

My work centres on evaluation/data workflows, cloud execution, benchmark integration, environment validation, coordination and later checkpoint/evaluation adoption. The formal training implementation also contains teammates' authored contributions; the Git history retains their names and does not relabel them as mine.

The latest filtered source lineage contains **44 retained commits, 40 attributed to Tian Liao**, including merges. Earlier public history remains intact; do not add both import lineages as a measure of unique contributions. See [contribution evidence](docs/CONTRIBUTIONS.md), [architecture](docs/ARCHITECTURE.md), [NOTICE.md](NOTICE.md) and [third-party attribution](THIRD_PARTY_NOTICES.md).
