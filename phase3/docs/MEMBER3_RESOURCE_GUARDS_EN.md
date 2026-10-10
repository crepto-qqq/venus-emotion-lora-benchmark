# Member 3 cumulative resource limits and independent shutdown preparation

> English translation of the corresponding Chinese historical record. Statements such as “pending” or “not run” refer to that record’s preparation milestone. For current execution results and remaining actions, read the [Member 3 completion report](../MEMBER3_COMPLETION_REPORT_EN.md), updated 10 October 2026.

Source: [MEMBER3_RESOURCE_GUARDS.md](MEMBER3_RESOURCE_GUARDS.md). Command placeholders in Chinese have been translated into English; actual fixed paths and command options are preserved.

Status at this historical milestone: local implementation and mock verification completed. No background task was activated, RunPod API contacted or cloud resource started. No real budget authorisation, API key or Pod ID was written into the configuration.

## 1. Cumulative training attempts

`run_lora_training.py --execute` now additionally requires `--resource-ledger` and `--attempt-minutes`. Use fixed locations within each stage:

- Pilot: `/workspace/phase3/member3/pilot/resource-ledger.json`.
- Formal: `/workspace/phase3/member3/formal/resource-ledger.json`.

Before each run, acquire a file lock and check maximum attempts, cumulative reserved minutes and cumulative reserved dollars estimated at the maximum permitted hourly rate. Different candidates must not switch ledgers. An existing ledger binds its stage and `resource_limits`; changed limits cause execution to be rejected. Duplicate attempt IDs, tampered ledger hashes and invalid numeric values are rejected. Failures or interruptions do not automatically refund reservations, preventing unaccounted retries. Shared-volume file-lock support still required actual verification at this milestone.

This ledger records conservative optimizer-process reservations, not a provider bill. Environment checks before model loading, independent validation generation, shutdown delays and storage are not automatically included. Whole-Pod time/cost control requires the independent measures below and accounting for Phase 3 costs already incurred.

## 2. Independently stop a specified Pod

`runpod_stop_watchdog.py` outputs a plan by default. Only `--execute` reads an ordinary RunPod API key file held by the current user with permissions 0600; S3 keys are unsuitable. Key contents are not written into arguments, reports or logs.

The tool uses only `GET /v1/pods` and `POST /v1/pods/<id>/stop`. It cannot create, start, edit or delete Pods. It checks the target Pod state, requests shutdown at a timezone-aware absolute deadline, then reads the API to verify EXITED. A missing target or failed request cannot be marked successful. If restarted after the deadline, it attempts shutdown immediately instead of waiting another full interval.

The API’s `desiredStatus` is a desired state. The report field `stop_state_observed` means only that the API returned EXITED. Confirm actual shutdown and rates in the console. This does not guarantee actual stop time, billed charges or network reliability.

`prepare_watchdog_launchagent.py` generates a macOS plist only; it neither runs `launchctl` nor installs the task. Its configuration uses RunAtLoad, restart on failure and separate log paths, without key contents. It wraps the watchdog with `/usr/bin/caffeinate -is` to suppress idle sleep while active, releasing that protection afterward without modifying system power settings. Lid closure, shutdown and network loss remain outside this protection. Generated ProgramArguments contain `--execute`: loading that background task later really contacts the API and stops the specified Pod at the deadline. The task was not loaded during this preparation work.

## 3. Checklist before a paid session

1. Establish new stage-specific limits and confirm previous cumulative cost. Do not reuse the completed $2/30-minute check authorisation.
2. Obtain explicit authorisation for an ordinary API key with the minimum available permissions. No key was created or read during this preparation work.
3. Identify the exact Pod ID, actual total hourly rate and absolute deadline measured from Pod billing start. Allow time for model loading, validation and shutdown/API delays.
4. Install and verify the background task outside the chat. Keep the Mac powered on, online, logged in and awake; otherwise use a reliable always-on machine. A login task cannot stop a cloud Pod while the Mac is powered off.
5. Actually verify Stop permissions, the background process and heartbeat. Local mocks do not replace this. Before training, manually confirm the protections are active; at this milestone the launcher did not automatically verify a watchdog receipt.
6. Stop immediately when the session finishes instead of waiting for the final deadline. Confirm actual shutdown and compute rate in the console, preserving billing/state evidence. The retained shared volume continues to incur storage charges.

## 4. Local test evidence

All 36 Phase 3 tests passed. New coverage includes cumulative counts, cross-candidate budgets, changed limits, duplicate attempts, ledger tampering, post-Stop state verification, network failures, missing targets, timezones and LaunchAgent arguments. All API tests used a simulated transport; no real request was sent.

Official references retained from the Chinese source, not newly verified for this translation:

- https://docs.runpod.io/pods/manage-pods
- https://docs.runpod.io/api-reference/pods/GET/pods
