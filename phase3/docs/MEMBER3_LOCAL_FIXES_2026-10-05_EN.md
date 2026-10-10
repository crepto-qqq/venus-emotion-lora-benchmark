# Member 3 local runtime protection fixes on 5 October 2026

> English translation of the corresponding Chinese historical record. Statements such as “pending” or “not run” refer to that record’s preparation milestone. For current execution results and remaining actions, read the [Member 3 completion report](../MEMBER3_COMPLETION_REPORT_EN.md), updated 10 October 2026.

Source: [MEMBER3_LOCAL_FIXES_2026-10-05.md](MEMBER3_LOCAL_FIXES_2026-10-05.md). Command placeholders in Chinese have been translated into English; actual fixed paths and command options are preserved.

This work changed local code only. It did not start RunPod, train a model or alter the frozen release at the recorded milestone.

## Fixes completed

- Pilot defaults read `/workspace/phase3/member3/pilot-data/pilot-train.json` and `pilot-validation.json`. Before launch, compare every row with the subset generated from the frozen release using the predefined seed. Reject wrong files, the full training set or modified content. Expected sizes are 24 train / 96 validation.
- The attempt record’s `config_sha256` retains original configuration identity; `resolved_config_sha256` separately identifies the configuration after output-path expansion. Reject inconsistent candidate IDs, reused attempt records and existing output directories.
- Apply a process-group timeout using the smaller of the wall-clock limit and the limit derived from budget/maximum hourly rate. Send TERM on expiry, then KILL after five seconds if necessary. This is not actual billing monitoring and does not stop the Pod. Cumulative multi-attempt budget control and independent Pod shutdown were still pending at this milestone.
- Evidence requires successful execution status, matching configuration, self-checking hashes, explicit successful save/reload, complete evaluation, valid adapter hashes and finite positive resource records. Exceeding configuration limits fails the check.
- Scoring rejects duplicate, missing or extra prediction IDs. Format validity does not establish guidance quality.

## Data paths and next steps

The original uploaded release is at `/workspace/phase3/data/bridge-train-v1/`; the frozen release inside the local repository remains in place. Generate the pilot subset with `make_lora_pilot_subset.py --output-dir /workspace/phase3/member3/pilot-data` during an authorised cloud session; this command is not an execution record for this local work. Then use `expand_lora_candidates.py` to generate candidates. Preserve image references from the frozen release and confirm actual image readability on the shared volume before running.

Do not copy pilot data paths into a formal configuration. Point it at the uploaded complete train/validation JSON files; the launcher compares every row against the complete release.

## Outstanding work and limitations at this milestone

- Actual parser/module inventory remained pending. Do not fabricate verified configuration or bypass the full launch gate.
- Actual model prediction generation, per-task loss and weight-reload evidence still required implementation/checking against the real training interface. At this point `evaluate_lora_pilot.py` only scored supplied predictions.
- Local evidence checks establish record consistency, not independent proof that weights exist or manually entered runtime facts are true. Actual logs and weight verification are also necessary.
- Cumulative attempt limits, independent Pod shutdown and new paid-run authorisation were incomplete. These local changes did not authorise paid training.

## Verification

The original 12 Member 3 tests and four new tests passed. New coverage includes real-process timeout, 24/96 subsets and rejection of the full training set, invalid budget values, and rejection of save failures, incomplete validation or wrong-config evidence.

Across Phase 3, 26 tests ran: 22 passed and four errored because the selected Python lacked `jsonschema`. The whole suite could not be described as passing. `git diff --check` passed.

## Follow-up adapter validation generation tool

Added `generate_lora_validation.py`, which outputs a plan by default and loads the model only with `--execute`. It accepts only the exact frozen BridgeTrain validation set of 96 conversations, rejecting other datasets or subsets. Generation reads user turns only; assistant references are used solely for training preprocessing and teacher-forced loss.

Execution requires the pilot gate, exact upstream commit, parser inventory’s `finetune_py_sha256`, readable images and complete adapter files. Use BF16 on a single CUDA GPU without offload. After loading the adapter, record each prediction, loss and valid supervised-token count. Scoring separately reports token-weighted validation loss. Output directories cannot be reused; failures preserve a failed `run.json`.

The local dry-run plan is in project-level `Member3_Local_Checks/validation-generation-plan.json`. Example adapter paths are illustrative and do not establish real weights. Three new input/identity/reference-isolation tests were added; all 29 Phase 3 tests passed after installing `jsonschema==4.19.0` in the independent test environment. The earlier four dependency errors were resolved.

The tool had not yet been verified with an actual GPU adapter. Actual `model.chat`/PEFT compatibility, full preprocessing imports, model-weight identity and the formal loss selection rule still required cloud checking. The tool does not control cumulative budget or Pod shutdown and did not authorise paid execution. External scheduling of intermediate-checkpoint validation remained pending.

## Follow-up cumulative training limits and shutdown preparation

Added `member3_budget.py` and connected it to the launcher. Use one fixed shared ledger per stage; conservatively reserve attempt count, minutes and cost before execution. Failed attempts do not automatically refund reservations. Added `runpod_stop_watchdog.py` and `prepare_watchdog_launchagent.py` for deadline-based shutdown of a specified Pod, state reads and macOS background-task configuration generation. All 36 tests passed; no watchdog was actually activated and no cloud resource was created.

This local implementation superseded the earlier statement that cumulative attempt controls were unimplemented. Whole-platform session cost, actual background installation/permission checks, shared-volume file locking and pre-launch watchdog-state verification remained outstanding. See [resource guards](MEMBER3_RESOURCE_GUARDS_EN.md) for scope and limitations.

## Follow-up cloud runtime package

Added read-only preflight, actual inventory binding and verifiable source packaging. Parser inventory now requires a fresh output file instead of overwriting the pending template, with exact commit/patch reconstruction checks. LoRA module inventory forces `local_files_only` and rejects report overwrites. The launcher and validation generator accept explicit `--interface` to reuse a new independent interface report.

See the [cloud package runbook](MEMBER3_CLOUD_PACKAGE_RUNBOOK_EN.md) for complete cloud steps and evidence requirements. Full-model forward/backward, real adapter validation, actual watchdog operation and new budget authorisation remained incomplete; none of those commands had been executed in the cloud at this milestone.

## Follow-up two-task full-model input check

Added `check_lora_model_inputs.py`. It plans offline by default and imports the CUDA model only with `--execute`, performing forward/backward on one real classification and one real joint-guidance training sample. No optimizer steps or weight saves occur. Checks cover frozen parameters, dtypes, supervision, target modules/counts, finite loss/gradients, BF16 wrapper inputs and before/after adapter digests. On errors, preserve a failed report and completed-task records; do not label the result passed.

The independent test environment ran 45 tests: 39 passed and six real autograd tests were skipped because torch was absent. The existing Python 3.10/torch 2.0.1 CPU environment separately passed seven tests in this module, including those six. Thus all 45 distinct tests were actually covered successfully across suitable local environments, rather than in a single environment. Logs are `Member3_Local_Checks/phase3-tests.log` and `model-inputs-cpu-tests.log`.

The new `cloud-bundle-2026-10-05-v2.tar.gz` contains these tools, tests and updated runbook. Keep the old package but use the new one; its new SHA256 and per-file manifest were verified. Actual Venus GPU checks, training and save/reload had not yet occurred at this milestone.
