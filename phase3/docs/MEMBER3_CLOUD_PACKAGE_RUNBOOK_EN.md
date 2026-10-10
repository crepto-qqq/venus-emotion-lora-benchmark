# Member 3 cloud check package execution order and evidence checklist

> English translation of the corresponding Chinese historical record. Statements such as “pending” or “not run” refer to that record’s preparation milestone. For current execution results and remaining actions, read the [Member 3 completion report](../MEMBER3_COMPLETION_REPORT_EN.md), updated 10 October 2026.

Source: [MEMBER3_CLOUD_PACKAGE_RUNBOOK_CN.md](MEMBER3_CLOUD_PACKAGE_RUNBOOK_CN.md). Command placeholders in Chinese have been translated into English; actual fixed paths and command options are preserved.

## Current purpose

This is a locally prepared source snapshot, not a completed pilot or proof of cloud execution. It contains the original Jackie-branch code based on commit `d2690f5d1535ae26e85788ddeb4f78e1940dc194` and subsequent uncommitted fixes. Per-file hashes in `bundle-manifest.json` identify the package; `base_commit` does not mean those changes have been committed. No model weights, images, API keys or paid-run authorisation are included.

Run the cloud commands below only after cloud execution authorisation, a whole-session budget and actual independent shutdown protection are in place. Extract into a new Member 3 directory, `/workspace/phase3/member3/bundles/<bundle-id>/`. Do not overwrite Tian’s upstream source, Python environment, models or data.

## 0. Verify before local transfer

Verify the package with `prepare_member3_bundle.py --verify <archive>`. The received archive SHA256 must match the external `verification.json`. The manifest contains each file’s SHA256. Do not reuse a directory containing old results.

## 1. Read-only shared environment check

Run from the package root using the existing Python; do not run `pip install` or `git pull`:

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/member3_cloud_preflight.py \
  --full-model-hashes \
  --report /workspace/phase3/member3/evidence/<session-id>/preflight.json
```

Replace `<session-id>` with a unique session name. The tool checks the five uploaded release files, hashes of the 480 images referenced by 576 conversations, 22 model files, upstream commit, exact reviewed patch and four key dependencies. Full model hashing reads approximately 19.3 GB and must be included in session time. This check does not load the neural network or perform a forward pass.

On failure, preserve the report and stop subsequent steps. Do not immediately upgrade dependencies, change the model or reapply patches.

## 2. Inspect the actual parser using a new output file

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/inventory_qwen_interface.py \
  --output /workspace/phase3/member3/evidence/<session-id>/qwen-interface.json
```

After checking the exact source and patch, import the actual `HfArgumentParser` dataclasses and record parser field names, types, defaults and file hashes. Preserve the default pending interface configuration and write a separate report. Stop on import errors; do not manually change pending to verified.

## 3. Model module inventory with no optimizer

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/inventory_lora_modules.py \
  --report /workspace/phase3/member3/evidence/<session-id>/lora-module-inventory.json
```

This tool loads the local model on CPU to list modules, dtypes and candidate LoRA parameter counts. By default it creates no optimizer and does not verify GPU forward execution. Confirm that `visual_matches` is empty and matched modules are nonempty. The earlier rank-2 smoke check does not replace this rank-8 configuration check.

## 4. Generate the pilot subset and verified configuration

Only after runtime inventory passes and the user authorises specific pilot `resource_limits`:

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/make_lora_pilot_subset.py \
  --output-dir /workspace/phase3/member3/pilot-data
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/bind_lora_inventory.py \
  --config phase3/configs/lora-pilot-v1.json \
  --inventory /workspace/phase3/member3/evidence/<session-id>/lora-module-inventory.json \
  --resource-limits /workspace/phase3/member3/evidence/<session-id>/authorized-pilot-limits.json \
  --output /workspace/phase3/member3/evidence/<session-id>/pilot-verified.json
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/expand_lora_candidates.py \
  --base /workspace/phase3/member3/evidence/<session-id>/pilot-verified.json \
  --output-dir /workspace/phase3/member3/evidence/<session-id>/candidates
```

`authorized-pilot-limits.json` records explicitly authorised values only; no budget is prefilled. Expect 24 pilot training conversations and the complete 96-conversation validation set. Keep original release hashes unchanged.

## 5. Dry run each candidate

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/run_lora_pilot.py \
  --config /workspace/phase3/member3/evidence/<session-id>/candidates/lr-1e-4.json \
  --interface /workspace/phase3/member3/evidence/<session-id>/qwen-interface.json \
  --inventory /workspace/phase3/member3/evidence/<session-id>/lora-module-inventory.json \
  --attempt-id <unique-dry-run-id> \
  --attempt-record /workspace/phase3/member3/evidence/<session-id>/<unique-dry-run-id>.json
```

This command omits `--execute` and creates no optimizer. Inspect the printed command for `pilot-train.json`, rank 8, BF16, frozen vision parameters, a single GPU and isolated output directories. Actual execution requires a new attempt ID, `--resource-ledger` pointing to the fixed shared ledger, `--attempt-minutes`, and explicitly authorised whole-session resource limits.

## 6. Two-task LoRA forward and backward checks with zero optimizer steps

The new tool outputs a plan by default. Actual cloud execution requires an authorised check session, valid inventory/interface/configuration and a new report path:

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/check_lora_model_inputs.py \
  --config /workspace/phase3/member3/evidence/<session-id>/pilot-verified.json \
  --interface /workspace/phase3/member3/evidence/<session-id>/qwen-interface.json \
  --inventory /workspace/phase3/member3/evidence/<session-id>/lora-module-inventory.json \
  --report /workspace/phase3/member3/evidence/<session-id>/model-input-check.json \
  --execute
```

Deterministically select one classification and one joint-guidance sample from the predefined pilot subset of the frozen training set. Use the full neural network, actual training preprocessing, current rank configuration, gradient checkpointing, BF16 base parameters and FP32 LoRA parameters. Do not adjust dtypes to hide failure. Require target modules and parameter counts to match inventory, frozen base and vision parameters, nonzero supervision after the causal shift, finite loss, finite gradients with at least one nonzero gradient, and BF16 wrapper inputs. Zero-initialised LoRA B normally causes zero gradients for A; not every gradient must be nonzero.

For each sample separately, clear gradients before forward/backward. Compute before/after digests of all adapter parameters to prove that weights remain unchanged. Do not create an optimizer or save an adapter. A successful report proves only that these two representative checks passed; it does not establish pilot completion, complete validation or training completion. Adapter save/reload in a fresh process is still checked after the pilot.

Local autograd guard tests have passed with a small CPU LoRA model; actual cloud Venus execution had not yet occurred at this milestone. The whole Pod deadline still applies; this tool does not stop the Pod automatically.

## 7. Generate and score outputs after the pilot

Pass the verified configuration, interface, inventory and actual adapter path explicitly to `generate_lora_validation.py`. It plans by default and generates only with `--execute`. `evaluate_lora_pilot.py` reads `predictions.jsonl`; missing, duplicate or extra IDs prevent acceptance. `validation_token_weighted_loss` weights loss by supervised token count; task-specific `validation_loss` currently averages per-sample losses. The predeclared selection rule for the three candidates must specify which measure to use before execution, rather than choosing a favourable metric afterward.

## Evidence checklist

| Stage | Required records | Actual results at this historical milestone |
| --- | --- | --- |
| Package identity | Archive SHA256, per-file manifest, base commit | Generated during local packaging |
| Platform protection | Explicit budget, Pod ID, whole-session deadline, background heartbeat, stop permissions and actual state | Incomplete |
| Shared environment | `preflight.json`, model/data hashes, dependency versions | Package commands not yet executed |
| Training interface | `qwen-interface.json`, source and parser identity | Not yet executed |
| Module inventory | Module names/counts, dtypes, parameter counts | Not yet executed |
| Data and configuration | 24/96 subset, manifest, candidate config hashes, dry-run commands | Awaiting cloud generation |
| Representative model checks | Two-task forward/backward, loss, gradients and frozen-parameter proof | Tool implemented and local autograd tests passed; actual Venus not run |
| Pilot | stdout/stderr, Trainer logs, attempt record, ledger, actual VRAM/time/cost | Incomplete |
| Validation and adapter | 96 predictions, loss, scoring, adapter file hashes, clean-process reload | Incomplete |
| Review | Independent pilot review by Jackie/you, exact formal configuration | Incomplete |

The package does not automatically modify the environment, accept invitations, create keys, or start/stop/delete cloud resources. Its watchdog is a tool only; separately confirm and verify actual activation.

## Actual check addendum dated 5 October 2026

The first actual v3 cloud preflight, parser inventory and module inventory passed. The two-task forward/backward check failed during loss computation for the first sample: the checker passed preprocessing’s int32 labels directly, bypassing formal Trainer’s `DataCollatorWithPadding`. The formal training entry point has a tokenizer; a local test using the pinned tokenizer showed that the default collator converts labels to int64 without changing their values or masks. No upstream `finetune.py` change is needed.

The local checker and validation tool now reuse the same actual collator; another explicitly authorised GPU check was still pending at this milestone. Do not change the v3 failed report to passed or continue using the old package. Regenerate the source package after this addendum and identify it by the corresponding archive/manifest hashes.

The GPU for that check was stopped and its temporary API key disabled. Full results and failure evidence are in the project-level `Member3_Local_Checks/CLOUD_CHECK_RESULT_2026-10-05_CN.md` and `cloud-evidence/20261005-h100-check/`.
