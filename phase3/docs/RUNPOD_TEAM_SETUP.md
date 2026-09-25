# RunPod team setup

## Recorded state after Member 1 technical acceptance

The following state was confirmed on 2026-09-25:

- the RunPod Team has been created, Member 1 is its `Admin`, and Member 1 is
  currently its only member;
- one 120 GB Standard Network Volume named `phase3-shared-120gb`
  (`bk4fycduml`) exists in `US-NE-1` and is retained for Phase 3;
- the shared environment, pinned upstream sources, model snapshot, and
  technical fixture are present on the volume;
- H100 NVL `attempt-001`, Member 1 clean-shell verification, and verification
  from a replacement Pod attached to the same volume all passed; and
- every temporary Pod is stopped. Only the retained volume continues to incur
  charges.

Member 1's technical environment is complete. Invitations for Members 2-6 and
independent cross-account identity verification are deferred, so shared-access
completion is not claimed. Check capacity and price immediately before any
future compute window.

## Target account and role policy

The existing normal RunPod Team keeps Member 1 as `Admin`. When onboarding is
resumed, Members 2-6 receive `Dev` roles so all six members can access the
shared project through their own accounts. Every member uses an individual
RunPod account and an individual SSH public key. Invitation acceptance and a
separate-account verification must finish before the handoff is called shared.

Follow RunPod's current guidance:
<https://docs.runpod.io/accounts-billing/manage-accounts>. An invitation link
must be treated as a sensitive bearer link. Share it only through the team's
approved private channel. Never share a login, password, API key, private SSH
key, or shell account.

All users attached to one Pod can inspect or affect its files. Team roles,
member directories, and the writer lock coordinate trusted collaborators; they
are not isolation between hostile users.

## Retained storage and future compute

Reuse the existing **120 GB Standard Network Volume** and mount it at
`/workspace` on at most one approved **on-demand Secure Cloud Pod**. Do not
create another Phase 3 volume or a Pod per member. The technical acceptance
enforces at least **44 GiB total GPU memory** and at least **40 GiB free GPU
memory immediately before model loading**. For a future run, select a
compatible 48 GB-or-larger GPU only after confirming current deployment
capacity, displayed hourly price, CUDA 11.8/PyTorch 2.0.1 compatibility, and
BF16 support. Potential compatible families include A40, RTX A6000, L40S, and
A100; this list does not claim current availability.

At the documented Standard rate of USD 0.07/GB/month, 120 GB is approximately
USD 8.40/month before taxes or policy changes. Pod compute is billed separately
at the rate displayed at deployment. Review the current storage constraints at
<https://docs.runpod.io/storage/network-volumes>; a Network Volume is tied to
its data-center location and must be attached during deployment.

The target volume contains:

```text
/workspace/models/Venus-Q-Stage1
/workspace/phase3/
  envs/venus-phase3
  upstream
  cache
  data
  smoke
  members
  runs
  reports
  locks
```

The shared environment is built from pinned top-level dependencies, and
bootstrap captures the fully resolved package freeze as evidence. This is not a
claim that every transitive package is pre-locked before installation.
Member-specific environments are not created for this handoff. Members write
only to:

```text
/workspace/phase3/members/<member-id>/
/workspace/phase3/runs/<member-id>/
/workspace/phase3/reports/<member-id>/
```

These `/workspace` locations are the standard handoff paths. CLI path overrides
are reserved for isolated tests or recovery and do not qualify an acceptance
run as the standard shared handoff.

## Completed and deferred sequence

The following items are complete:

1. The Team was created and Member 1 was confirmed as Admin.
2. `phase3-shared-120gb` was created in `US-NE-1` and mounted at `/workspace`
   for the accepted compute windows.
3. Bootstrap downloaded only missing files from the pinned
   `popo28/Venus-Q-Stage1` revision into
   `/workspace/models/Venus-Q-Stage1`, using the persistent Hugging Face cache.
4. The full 22-file snapshot was verified at revision
   `0f5c00c8d07ba889e9c5d12f828129dc322aae6a`, manifest SHA-256
   `13dbd18f9848ecdb7520bfe48af2704149435f2895fc67593df31c7552405298`,
   19,315,474,586 total bytes, ten safetensor shards totaling 19,312,732,032
   bytes, and exact per-file hashes. Bootstrap writes `VENUS_MODEL_SOURCE.json`
   and verifies that every pinned file, the marker, and model root are
   read-only.
5. Member 1 technical acceptance and both Member 1 verification passes
   succeeded at project commit `ea2dace2026523b6a498d634301927ddc335e015`.
6. All temporary Pods were stopped and the retained volume was confirmed.

The remaining access-onboarding sequence is deliberately deferred:

1. Invite Members 2-6 as Dev. Each member registers their own public key.
2. Member 2 mounts the retained volume from a separate RunPod account and
   verifies the immutable attempt from a member-owned clone pinned to commit
   `ea2dace`. Do not switch the shared checkout in place.
3. Preserve private Team and SSH identity evidence outside Git. This access
   check does not require an H100 to stay running.

## One-writer rule

Only Member 1 writes shared state during bootstrap and technical acceptance.
Both commands use `flock` on
`/workspace/phase3/locks/shared-writer.lock`. A contender fails immediately
instead of waiting and changing shared state later. The lock file contains
append-only owner records for diagnosis; the kernel lock itself is released
automatically when the process exits.

Treat `bootstrap.sh` and `member1_acceptance.sh` as the public shared-write
entry points. Lower-level environment creation, source fetch, and fixture-build
commands are internal steps and may run against shared storage only through
those Member 1 wrappers while the writer lock is held.

Other members may read the model, environment, sources, and Member 1 report
while writing only to their own paths. `verify_handoff.sh` does not take the
shared writer lock or rerun the GPU smoke. It accepts only a flat immutable
bundle of real regular files, checks the complete fixture and checksum chain,
snapshots the source attempt before and after validation, and writes a sealed
result below the caller's own report directory. A failure receives a stable
`failure_code` and preserves any checks completed before the failure whenever
the caller-owned output can be created.

The verifier checks the bundle and filesystem behavior under the supplied
`--member-id`; it cannot determine which RunPod account or SSH key owns the
current shell. Separate-account completion will require private evidence that
the invited member joined the Team and connected with their individual SSH
key. This check is deferred. Keep future identity evidence out of Git and
redacted reports.

## Access and credential rules

- Store only public SSH keys in the supported RunPod account or Pod field.
- Keep private keys, API keys, passwords, invitation links, signed URLs, and
  payment details out of the volume, repository, reports, and screenshots.
- Do not persist Git credentials in the shared checkout.
- Use a new SSH login for Member 1's clean-shell verification and a different
  RunPod account/key for Member 2's cross-account verification.
- Revoke access promptly if a team member leaves the project.

## Cost control and backup

Record Pod ID, GPU type, displayed hourly rate, start/stop UTC timestamps,
billed duration, and estimated/actual cost in the team's private cost record.
The completed setup used approximately USD 1.39 of compute. The retained 120 GB
volume is approximately USD 8.40 per month at the recorded rate, so the first
month's observed setup total is approximately USD 9.79. Every temporary Pod is
stopped. For future windows, set a short automatic stop timer and confirm the
console state after stopping; closing SSH does not stop the Pod.

The Network Volume is working storage, not the only backup. Push source and
redacted reports to Git, and copy irreplaceable private evidence to a separate
access-controlled team location. Model files and caches remain replaceable from
their pinned sources. Recheck the approved spending limit before each compute
window.
