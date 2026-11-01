# Phase 3 shared-work handoff checklist

Use this checklist when a member begins or hands off Phase 3 work. It protects
the accepted Member 1 environment while allowing all six members to work in
parallel after account onboarding.

## Before starting

- [ ] Use your own RunPod account and SSH public key. Do not share credentials.
- [ ] Confirm the Pod mounts the retained `phase3-shared-120gb` volume
  (`bk4fycduml`) at `/workspace`; do not create another Phase 3 volume.
- [ ] Record the Pod's GPU, displayed price, start time, and automatic stop
  limit in the private cost record.
- [ ] Classify the planned cost as training or non-training. Track all
  non-training Phase 3 spend against the cumulative USD 20 ceiling, including
  the existing H100 smoke, retained-volume charges, and future non-training
  work; start no non-training operation that would exceed the ceiling.
- [ ] Read [`MEMBER1_HANDOFF.md`](MEMBER1_HANDOFF.md) and confirm the required
  source and model revisions.
- [ ] Treat `/workspace/models/Venus-Q-Stage1`,
  `/workspace/phase3/envs/venus-phase3`, and
  `/workspace/phase3/upstream` as read-only shared inputs.
- [ ] Work only in your own `/workspace/phase3/members/<member-id>/`,
  `/workspace/phase3/runs/<member-id>/`, and
  `/workspace/phase3/reports/<member-id>/` directories.

## Before training

- [ ] Confirm the released full dataset passes every completeness, acceptance,
  provenance, schema, and Eval20/Eval80 leakage check.
- [ ] Confirm the signed BridgeTrain-v1 summary records
  `full_dataset_ready: true`. Member 1 `attempt-001` is historical technical
  evidence and deliberately leaves that value false.
- [ ] Confirm the dataset summary still records
  `formal_training_authorized: false`. Keep this dataset flag unchanged; the
  Member 3 review records govern the later training stages.
- [ ] Freeze the formal LoRA target modules and hyperparameters in a reviewed
  configuration; the rank-2 Member 1 smoke configuration is infrastructure
  evidence, not the training recommendation.
- [ ] Before Member 3A starts a paid pilot, ask the user to decide the allowed
  GPU, maximum hourly price, pilot total budget, wall-clock/attempt limits, and
  whether an OOM may trigger a GPU or duration change. Record the decision in
  the experiment record; do not assume a training budget.
- [ ] After the pilot, require Member 3B to verify the evidence and write
  `phase3/reviews/member3/pilot-review-v1.json`. A passing review authorizes
  Member 3A's exact formal configuration without approval from another member.
- [ ] Before Member 3A starts the paid formal run, ask the user again for the
  formal-training GPU, maximum hourly price, total budget, and response to OOM
  or a required extension.
- [ ] Create a copy of
  [`EXPERIMENT_RECORD_TEMPLATE.md`](EXPERIMENT_RECORD_TEMPLATE.md) and fill in
  the immutable inputs, command, budget, and stop conditions.
- [ ] Acquire the shared writer lock for any approved operation that changes
  shared state. Do not edit a completed attempt or another member's directory.

## During the run

- [ ] Write checkpoints, adapters, raw logs, and private evidence only to the
  volume under your member-owned paths.
- [ ] Preserve failed attempts and use a new attempt number for every retry.
- [ ] Record configuration changes before applying them. Never repair a shared
  checkout or environment while another member is using it.
- [ ] Monitor elapsed time and cost. Stop the run at the approved limit or when
  acceptance criteria can no longer be met.

## Before handoff

- [ ] Validate the output schemas, hashes, parameter-freeze state, and required
  metrics from a clean process.
- [ ] After formal training, require Member 3B to select the final checkpoint
  and record the adapter hash, clean-process reload result, selection evidence,
  and acceptance in
  `phase3/reviews/member3/training-acceptance-v1.json`.
- [ ] Hand the accepted adapter to Member 4 for adapter inference and Condition
  C integration. Member 5 independently runs the formal Eval80 against that
  frozen integration; Member 6 then packages the minimal inference API.
- [ ] Save a package checksum and make the evidence paths readable to the next
  member without changing the accepted Member 1 bundle.
- [ ] Remove credentials, invitation links, signed URLs, host details, image
  content, and private paths from every Git-bound file.
- [ ] Keep models, images, caches, checkpoints, adapters, rendered output, and
  raw logs out of Git.
- [ ] Commit only source, configuration, and deliberately redacted summaries on
  a task branch, then use the repository's pull-request workflow.
- [ ] Stop the Pod and confirm the stopped state in the console. Closing SSH is
  not sufficient.
- [ ] Tell the next owner the exact commit, data release, evidence paths,
  unresolved failures, and next bounded action.

## Demo API boundary

Keep training and evaluation as reproducible command-line workflows. Add the
small inference-only Python API/caller after Member 4's adapter inference and
Condition C integration are stable and Member 5 has completed the independent
formal Eval80. Member 6's API must load the pinned base model and Member
3B-accepted adapter without changing training artefacts or shared inputs.
