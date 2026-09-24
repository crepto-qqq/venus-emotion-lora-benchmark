# RunPod team setup

## Recorded state before Phase 3 provisioning

The following is a dated observation from the RunPod console on 2026-09-23. It
is a starting point, not a statement that Phase 3 cloud setup is complete:

- the account is still personal and offers **Convert to a Team Account**;
- two legacy A40 Pods are stopped;
- each stopped Pod has its own 120 GB Pod Volume Disk mounted at
  `/workspace`;
- the two disks are separate and neither is a portable Network Volume;
- the Network Storage page contains no Network Volume;
- the console showed approximately USD 0.03/hour storage for each legacy Pod
  disk and a combined account spend rate of about USD 0.067/hour;
- starting a legacy A40 Pod showed USD 0.49/hour at the time of inspection.

Prices and capacity must be checked again immediately before any paid action.
No Team conversion, Network Volume creation, legacy Pod start, migration, or
new Pod deployment is asserted by this document. Keep both stopped Pods and
their disks until migration evidence is reviewed; deleting either one is a
separate destructive action.

## Target account and role policy

Use a normal RunPod Team. Member 1 receives the `Admin` role and Members 2-6
receive `Dev` roles. Every person uses an individual RunPod account and an
individual SSH public key. Members may be invited after Member 1 prepares the
environment; invitation acceptance and a separate-account verification must
still finish before the handoff is called shared.

Follow RunPod's current guidance:
<https://docs.runpod.io/accounts-billing/manage-accounts>. An invitation link
must be treated as a sensitive bearer link. Share it only through the team's
approved private channel. Never share a login, password, API key, private SSH
key, or shell account.

All users attached to one Pod can inspect or affect its files. Team roles,
member directories, and the writer lock coordinate trusted collaborators; they
are not isolation between hostile users.

## Target compute and storage

Create one **120 GB Standard Network Volume** in Secure Cloud and mount it at
`/workspace` on one **on-demand Secure Cloud Pod** with at least **48 GB of GPU
memory**. Select the lowest currently available price among the approved
48 GB class, such as A40, RTX A6000, or L40, after checking framework and BF16
support. Do not create six Pods or six volumes.

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

## Approval and migration sequence

Team conversion changes account permissions, while volume creation, adding
funds, starting an old Pod, and deploying a new Pod incur charges. Execute them
only after the user has reviewed the exact pending actions, current prices, and
funding amount and has confirmed them at action time.

After that confirmation:

1. Convert the account to a Team and assign Member 1 as Admin.
2. Create the 120 GB Standard Network Volume in a location with an approved
   48 GB GPU available.
3. Start one legacy A40 Pod only after identifying which of its separate Pod
   Volume Disks contains the authoritative Stage 1 model and fixture source.
4. Deploy the target Pod with the Network Volume attached at deployment time;
   RunPod does not attach a new Network Volume to an already-created legacy
   Pod. Copy from the running source Pod to the target Pod with an authenticated
   `rsync` transfer, or transfer through RunPod's S3-compatible Network Volume
   API. Copy only the pinned model, required technical fixture source, and other
   explicitly retained project inputs.
5. Verify the model repository revision
   `0f5c00c8d07ba889e9c5d12f828129dc322aae6a`, ten safetensor shards,
   19,312,732,032 total shard bytes, index file, and fixture image SHA-256
   `f4552a57efd8ff17e0a7a9fe28e1e94e98401cc5a5ace21ee6c246d74766d082`.
6. Stop the legacy Pod immediately after the copy and verification.
7. Deploy one target Pod with the Network Volume at `/workspace`, add only
   Member 1's public key initially, and run bootstrap and acceptance.
8. Invite Members 2-6 as Dev. Each member registers their own public key.
9. Member 2 performs the separate-account verification, then Member 1 stops the
   target GPU Pod.

Do not delete either legacy Pod or disk during this sequence. Migration and
deletion have different acceptance criteria.

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
shared writer lock, does not rerun the GPU smoke, and proves that the source
attempt was unchanged while creating a result below the caller's own report
directory.

The verifier checks the bundle and filesystem behavior under the supplied
`--member-id`; it cannot determine which RunPod account or SSH key owns the
current shell. Separate-account completion also requires private evidence that
the invited member joined the Team and connected with their individual SSH
key. Keep that identity evidence out of Git and redacted reports.

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
Set an approximately six-hour auto-stop or equivalent timer for the Member 1
GPU window. Confirm the console state after stopping; closing SSH does not stop
the Pod.

The two legacy Pod disks continue to incur storage charges while retained.
Review them after successful migration, but do not remove them as part of
Member 1's automated setup. The new Network Volume is working storage, not the
only backup. Push source and redacted reports to Git, and copy irreplaceable
private evidence to a separate access-controlled team location. Model files and
caches remain replaceable from their pinned sources.
