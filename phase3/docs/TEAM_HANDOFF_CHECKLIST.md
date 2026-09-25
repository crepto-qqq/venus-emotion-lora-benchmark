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
- [ ] Confirm `full_dataset_ready` and `formal_training_authorized` have been
  changed only through the team's reviewed release process. Member 1
  `attempt-001` deliberately leaves both values false.
- [ ] Freeze the formal LoRA target modules and hyperparameters in a reviewed
  configuration; the rank-2 Member 1 smoke configuration is infrastructure
  evidence, not the training recommendation.
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
small inference-only Python API/caller after the adapter and inference path are
stable, close to the demonstration. The API must load the pinned base model and
approved adapter without changing training artefacts or shared inputs.
