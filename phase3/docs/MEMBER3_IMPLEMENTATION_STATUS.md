# Member 3 implementation status

> Historical preparation or implementation record. For current execution results and remaining actions, read the [Member 3 completion report](../MEMBER3_COMPLETION_REPORT_EN.md), updated 10 October 2026.

## Repository implementation completed
The repository now contains the Member 3 offline implementation required by the canonical handoff:

- LoRA pilot base config and bounded three-candidate experiment matrix;
- Qwen training-interface contract with a mandatory pinned-source parser-inventory gate;
- deterministic pilot-subset generation and RFC 7396 candidate expansion;
- canonical JSON hashing and raw BridgeTrain-v1 release-hash verification;
- static/runtime-inventory/pilot/formal validation modes;
- read-only live LoRA module inventory with visual-target rejection;
- validated pilot/formal launch path, dry-run by default;
- BridgeTrain-v1-only pilot evaluation and Git-safe evidence collection;
- Member 3B post-training acceptance validation;
- JSON schemas for configs, matrix, inventory, subset, evidence, review, acceptance, interface, and validation reports;
- Member 3A runbook, Member 3B review runbook, experiment plan, and Member 4 adapter handoff; and
- focused tests for forbidden methods/settings and lifecycle-gate failures.

## Local verification
At the repository-only boundary:

- `validate_lora_config.py --mode static` passes for the pending-runtime pilot template;
- the deterministic subset generator produces a balanced pilot manifest;
- the three candidate configs expand successfully and differ only in learning rate;
- 48 existing project tests pass;
- 22 Phase 3 tests pass, including 12 focused Member 3 tests;
- all Member 3 Python files compile;
- shell scripts pass `bash -n`; and
- all new JSON schemas pass Draft 2020-12 schema validation.

## Deliberately not claimed
The uploaded archive does not contain the pinned Qwen `finetune.py` checkout or the private Venus model/image runtime. Therefore the following remain runtime gates, not completed facts:

- exact live `HfArgumentParser` field inventory after applying the reviewed patch;
- concrete LoRA matched module names/count and observed model dtypes;
- 480 deployed private-image hash checks;
- paid pilot optimizer runs, losses, gradients, VRAM, throughput, timing, displayed price, or cost;
- adapter/checkpoint save and clean-process reload evidence;
- Member 3B pilot approval for one exact formal config;
- paid formal training and checkpoint selection; and
- final `training-acceptance-v1.json` and Member 4 adapter handoff identity.

Those gates must be executed on the retained RunPod environment with user-supplied resource limits. No placeholder runtime/review/acceptance record has been created.
