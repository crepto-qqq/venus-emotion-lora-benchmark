# Venus Emotion-Aware LoRA Infrastructure & Model Benchmarking

> 基于 Venus 的情绪感知 LoRA 基础设施开发与原模型对比评测

This repository is a public source portfolio derived from a six-person University of Sydney SOFT3888 capstone project. The project explored how emotion-aware prompting and parameter-efficient fine-tuning could improve multimodal image-aesthetic guidance and cropping.

It contains a filtered history of the team-authored engineering work that Tian Liao led or implemented. Upstream Venus/Qwen code, restricted datasets, model weights, generated evaluation outputs, assessment administration material, and private team documents are not included.

## Project snapshot

| Item | Summary |
|---|---|
| Project context | University of Sydney SOFT3888 capstone, six-person team |
| Tian's roles | Programmer, tester, customer liaison, evaluation/data pipeline contributor, and Phase 3 handoff owner |
| Core stack | Python, PyTorch, Qwen-VL, PEFT/LoRA, RunPod, Linux/Shell, `unittest` |
| Evaluation | Eval80-v1: 80 images across 8 emotion categories |
| Data pipeline | 480 unique images and 576 Qwen-VL conversations in the private project release |
| GPU validation | H100 NVL, BF16 forward/backward technical validation |
| Public history | 29 source-relevant Tian commits retained from 35 original Tian commits |

## Problem and approach

The project studied whether explicit emotion conditioning could improve multimodal aesthetic guidance while preserving reproducibility and evaluation integrity.

The comparison design contained four conditions:

- **A — Official baseline:** the released Venus prompt.
- **B0 — Unstructured emotion prompt:** emotion recognition and guidance requested jointly.
- **B1 — Structured emotion prompt:** the same task with a constrained structured response.
- **C — LoRA condition:** B1 plus a planned emotion-aware LoRA adapter.

A, B0, and B1 each produced 80 outputs and passed 80/80 technical acceptance in the private experiment, for 240 validated outputs in total. Condition C reached data, environment, PEFT injection, parameter, gradient, and GPU runtime validation. Formal optimiser training and adapter checkpoint production were not completed at the documented handoff point.

## Engineering highlights

### Reproducible evaluation workflow

- Built a Python CLI covering preflight checks, frozen-input loading, inference execution, output validation, blind score-sheet preparation, and summary generation.
- Locked model snapshots, generation settings, random seeds, environment versions, and result hashes for repeatable cloud execution.
- Added checks for missing records, duplicate samples, schema violations, label leakage, configuration drift, and unexpected hash changes.
- Preserved the RunPod/Linux wrappers used for A/B0/B1 execution and technical acceptance without publishing their restricted outputs.

### Emotion-aware training-data pipeline

- Produced BridgeTrain-v1 in the private project with 400 training images, 80 independent validation images, and 576 structured multimodal conversations.
- Enforced quotas across 8 emotion classes, hash-based deduplication, train/validation separation, and exclusion of every Eval20 and Eval80 image.
- Used schema validation and deterministic packaging to make the handoff auditable.
- Published the pipeline and contracts here while excluding the underlying images, annotations, and generated release artefacts.

### LoRA infrastructure validation

- Adapted the Venus/Qwen-VL workflow for PEFT LoRA injection while freezing the visual tower.
- Pinned Python, PyTorch, CUDA, model, and upstream-source versions for a reproducible RunPod environment.
- On an H100 NVL, completed a real-image BF16 forward/backward validation and verified:
  - 128 targeted LoRA modules;
  - 3,506,176 trainable parameters;
  - non-zero gradients through the intended trainable path.

This demonstrates training-pipeline readiness; it does not claim a completed fine-tuned adapter.

## Repository layout

| Path | Purpose |
|---|---|
| `src/eval80/` | Reproducible evaluation, inference orchestration, scoring, and validation |
| `src/datasets/` | Emotion annotation and BridgeTrain release validation |
| `src/phase1/` | Baseline execution, cropping, manifests, and score-sheet preparation |
| `phase3/tools/` | LoRA environment, snapshot, smoke-test, and handoff verification tools |
| `phase3/contracts/` | Machine-readable schemas for data and execution evidence |
| `scripts/eval80/runpod/` | Cloud execution and acceptance wrappers |
| `tests/`, `phase3/tests/` | Automated unit and contract tests |
| `environment/`, `phase3/environment/` | Reproducible dependency and runtime definitions |

## Public snapshot boundary

This repository is a historical, sanitised source snapshot—not a self-contained redistribution of the complete research environment. The included unit and contract tests exercise the public code, but the original end-to-end experiment cannot be rerun unchanged from this checkout alone:

- private dataset records, images, manifests, evaluation handbooks, outputs, and reports are deliberately omitted;
- the reviewed Qwen compatibility patch is described and hash-pinned, but not redistributed with this portfolio;
- Venus/Qwen source and model files must be obtained separately under their own terms;
- historical RunPod scripts preserve the original interfaces and paths, so commands requiring omitted inputs will stop rather than substitute fabricated data.

## Verification

Run the public-source test suites from the repository root:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s phase3/tests -v
```

The core suite contains 48 passing tests. On Windows in this public checkout, the Phase 3 suite passes seven public contract tests and explicitly skips three checks: two require the private dataset fixture and one verifies POSIX-only filesystem semantics. No fake replacement data is included.

## My contribution and retained history

My authored work concentrated on the evaluation system, cloud execution workflow, emotion-annotation pipeline, data-release gates, and LoRA environment validation. The original private history attributes 35 commits to Tian Liao. After restricted paths were removed and empty commits were pruned, this public repository retains 29 source-relevant commits, including their original authorship, dates, messages, and development sequence. The public email is rewritten to the account's GitHub noreply address.

Selected evidence is summarised in [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md). The filtering and provenance process is documented in [SOURCE_ORIGIN.md](SOURCE_ORIGIN.md), and the system flow in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Skills demonstrated

- Multimodal LLM evaluation and benchmark design
- PEFT/LoRA integration and GPU runtime validation
- Deterministic dataset construction and leakage prevention
- Python CLI development and automated validation
- Linux/RunPod execution, dependency pinning, and reproducibility
- Git/GitHub pull-request collaboration, technical documentation, and team handoff

## Attribution and release boundary

This is a team-project portfolio, not a claim of sole ownership over the entire capstone. The retained code is team-authored derivative engineering built around upstream systems; it does not include or relicense Venus, Qwen-VL, AesGuide, EmoSet, or their assets. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [NOTICE.md](NOTICE.md).

