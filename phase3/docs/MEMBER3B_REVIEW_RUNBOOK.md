# Member 3B review runbook

Member 3B reviews the exact Member 3A evidence commit, not a moving branch name. Verify release hashes, parser/interface inventory, model inventory, canonical candidate hashes, matrix hash, attempt limits, stop-rule compliance, validation metrics, adapter/checkpoint hashes, and clean-process reload evidence.

The pilot review must be a schema-valid `phase3/reviews/member3/pilot-review-v1.json` with `decision: approved` before formal execution. It binds exactly one formal-config SHA-256, the matrix hash, the pilot evidence hashes, and the user's separately supplied formal resource limits. A changed formal config requires a new review version.

After formal training, select one checkpoint only from the predeclared BridgeTrain-v1 validation rule. Write `training-acceptance-v1.json` only if the executed config and review hashes match, cost/time constraints were respected, adapter contents are identified by SHA-256, and clean-process reload plus representative inference passed. Otherwise record rejection and do not hand an adapter to Member 4.
