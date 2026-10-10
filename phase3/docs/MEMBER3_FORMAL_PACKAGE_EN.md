# Member 3 formal training preparation package

> English translation of the corresponding Chinese historical record. Statements such as “pending” or “not run” refer to that record’s preparation milestone. For current execution results and remaining actions, read the [Member 3 completion report](../MEMBER3_COMPLETION_REPORT_EN.md), updated 10 October 2026.

Source: [MEMBER3_FORMAL_PACKAGE_CN.md](MEMBER3_FORMAL_PACKAGE_CN.md). Command placeholders in Chinese have been translated into English; actual fixed paths and command options are preserved.

This package contains local preparation materials. It is not evidence of completed formal training and does not authorise starting a Pod.

## Package contents

- Current Phase 3 code, tests, frozen data-release artifacts and lock files.
- `formal-preparation/formal-config-DRAFT.json`: 480 training conversations, 3 epochs, 180 optimizer steps, checkpoint saves every 60 steps and provisional learning rate 2e-4.
- `formal-preparation/reviewed-evidence/`: comparison verification for three pilots, diagnosis of 16 validation guidance outputs and review of 80 training guidance records. Original contents and SHA256 hashes are retained.
- `formal-preparation/delegated-decision-PORTABLE-PENDING.json`: evidence paths converted to package-relative paths. Status remains pending; it is neither an independent Jackie review nor user approval of the formal configuration.
- `bundle-manifest.json`: per-file sizes and SHA256 hashes. Whole-package verification is stored separately beside the archive.

No model weights, images, Python environment, API keys or SSH private keys are included. Separately perform read-only checks of the models, images and existing environment on the shared volume. The packager selects only Phase 3 file types allowed by the existing source bundler; do not place credentials inside these source or JSON files.

## Using the package on another machine

Place the archive in a new isolated directory without overwriting Tian’s checkout, environment or frozen data. Before extraction, check that its SHA256 matches the local `.verification.json`. The verifier’s `--verify` mode reads the archive without extracting it:

```bash
python /known-trusted-source/phase3/tools/prepare_member3_bundle.py --verify /actual-path/formal-preparation.tar.gz
```

After extraction into an isolated directory, convert the review draft’s evidence paths to absolute paths on that machine. Replace the two directory placeholders below. Conversion preserves pending status and does not execute training:

```bash
python /extracted-directory/phase3/tools/prepare_formal_bundle.py materialize \
  --bundle-root /extracted-directory \
  --output /new-output-directory/delegated-decision-PENDING.json
```

Do not paste checkpoint commands previously generated on the Mac directly into the cloud: regenerate commands after deployment so their tool paths are correct. `plan_formal_validation.py --mode plan` creates a plan; `--mode collect` verifies only actual completed results. It does not automatically train or stop resources.

## Outstanding requirements before actual execution

1. Connect to an available machine and verify the shared volume, frozen data, images, models and existing Python environment. Collect a fresh module inventory and bind the final configuration.
2. Deploy and verify `run_formal_session.py`. Training and validation for all three checkpoints share one deadline; timed-out subprocesses are terminated and any failed stage stops the chain. The optimizer ledger is not whole-Pod cost control. Code and local tests were complete at this milestone, but actual cloud connectivity was not yet verified.
3. Activate the independent local Pod stop watchdog. Training-process exit does not mean the Pod stopped; query and confirm its state afterward. Keep credentials local, outside the cloud package.
4. Generate a fresh record using the final configuration, actual GPU rate, budget and time limit, and obtain corresponding explicit confirmation. Rebind the config SHA256 after changes. Do not change the frozen release summary’s historical authorisation field to true.
5. After formal training, reload checkpoints 60/120/180 in separate clean processes, run the same 96 validation conversations, and compare by the fixed rule. Review guidance content before accepting a candidate adapter.

Formal Eval80, B1/C comparison and anonymous randomised five-dimension human guidance scoring remain subsequent work. Validation loss or format pass rate cannot replace them.

## Connecting session execution and shutdown protection

The following procedure applies after actual connection details are ready. Local tests do not prove cloud execution; the pending draft is not directly executable. Fill all path placeholders, Pod IDs, ports and deadlines from the actual environment.

### 1. Define one session

The session record requires at least the following fields. Do not reuse an old pilot session or approval record:

```json
{
  "pod_id": "ACTUAL_POD_ID",
  "attempt_id": "formal1",
  "started_at_utc": "Actual session start time including timezone",
  "deadline": "Absolute session deadline including timezone",
  "max_session_usd": null,
  "actual_hourly_usd": null,
  "training_minutes": null,
  "resource_ledger": "/workspace/phase3/member3/formal/resource-ledger.json",
  "config_sha256": "Final configuration SHA256",
  "session_sha256": "Canonical SHA256 of this record excluding session_sha256"
}
```

`started_at_utc` begins at actual Pod start for this session, including deployment and checks; do not reset it when training begins. The runner checks that the time limit and estimated compute cost fall within the final configuration’s resource scope, but cannot independently verify provider billing or the actual billing start. Confirm GPU, actual rate and start time in the console. Storage charges and platform delays are not automatically capped.

Compute session and final configuration SHA256 values using `member3_common.canonical_sha256`, rather than ordinary file-byte hashing. If configuration, inventory or resource scope changes, update the bindings and have the user confirm the change.

### 2. Prepare the independent watchdog on the Mac first

`prepare_watchdog_launchagent.py` generates a LaunchAgent file only; it runs after separate loading. Add `--stop-request /session-directory/stop-now`, using the same path as the relay in the next step. Keep the API key file only on the Mac with mode 0600. Use the actual Pod ID, the same deadline and fresh report/trigger paths.

Confirm a watchdog report of `armed`, `observed_state=RUNNING` and a fresh heartbeat. `waiting_for_start` is not evidence of an active run.

### 3. Start the relay on the Mac

Every five seconds, send the watchdog heartbeat without its key to the cloud through SSH, and read cloud task status. When the task completes, fails or connectivity breaks, create local `stop-now` so the independent watchdog requests shutdown. Each SSH call has a timeout; the watchdog’s absolute deadline does not depend on SSH connectivity.

```bash
caffeinate -is python /local-source/phase3/tools/relay_formal_watchdog.py \
  --session /session-directory/session.json \
  --watchdog-report /session-directory/watchdog.json \
  --identity /existing-ssh-private-key \
  --target root@ACTUAL_SSH_HOST --port ACTUAL_PORT \
  --remote-guard /workspace/phase3/member3/formal/unique-session-directory/guard.json \
  --remote-status /workspace/phase3/member3/formal/unique-session-directory/evidence/session-status.json \
  --stop-request /session-directory/stop-now \
  --relay-report /session-directory/relay.json --execute
```

The cloud `unique-session-directory` must already exist, while its `evidence` directory must not yet exist because the runner creates it. Complete initial SSH host verification normally; the relay uses BatchMode and does not enter passwords or accept unknown hosts. Keep the Mac online and awake. If connectivity is lost, the cloud task stops when its heartbeat is over 30 seconds old; the local watchdog still attempts API shutdown. Mac power loss or API failure prevents a shutdown guarantee; manually confirm platform state.

### 4. Run the cloud session runner

First omit `--execute` to inspect the plan. Confirm an isolated directory, approved final configuration and newly collected/bound inventory, which must be no more than 24 hours old at execution. Add `--execute` for actual execution:

```bash
/workspace/phase3/envs/venus-phase3/bin/python /deployed-source/phase3/tools/run_formal_session.py \
  --config /final-config-directory/formal-config.json \
  --session /session-directory/session.json \
  --inventory /session-directory/lora-module-inventory.json \
  --interface /session-directory/qwen-interface.json \
  --review /session-directory/delegated-decision-APPROVED.json \
  --guard /workspace/phase3/member3/formal/unique-session-directory/guard.json \
  --data-dir /workspace/phase3/data/bridge-train-v1 \
  --output /workspace/phase3/member3/formal/unique-session-directory/evidence
```

The fixed sequence is full-model hash preflight → formal configuration/review gate → two-task forward/backward checks → complete formal training → separate clean-process validation for checkpoints 60/120/180 → strict evidence comparison. The runner does not retry training or add a fourth checkpoint candidate. `training_minutes` is a separate training-stage limit, while the complete chain must also fit the session deadline. Failure at any step prevents the next step.

`session-status.json` records execution status, command exit codes and input-file bindings. `stop_requested=true` means shutdown was requested only. Check the Mac watchdog’s `stop_state_observed` and confirm Not running in the console before reporting that the Pod stopped. Without confirmation, do not claim shutdown. Outputs remain on the shared volume; after shutdown, download, verify, back up and prepare the handoff.

Local tests use mock commands for control flow and real CPU subprocesses for timeout/nested-process termination. At this milestone they did not call the actual RunPod API, train a model or produce formal validation scores.
