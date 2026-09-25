# RunPod team setup

## Recorded state before Phase 3 provisioning

The following is a dated observation from the RunPod console on 2026-09-25. It
is a starting point, not evidence that Phase 3 cloud setup is complete:

- the RunPod Team has been created, Member 1 is its `Admin`, and Member 1 is
  currently its only member;
- there is no Phase 3 Pod and no Network Volume;
- one 120 GB Standard Network Volume, invitations for Members 2-6, a shared GPU
  Pod, and the real GPU acceptance are all pending.

Every compatible 48 GB-or-larger candidate checked on the deployment page --
L40S, A40, RTX A6000, and A100 PCIe 80 GB -- reported `Out of capacity`. The
A100 PCIe listing showed a USD 1.59/hour baseline. No GPU type, region,
capacity, or price is reserved by this observation. Check the deployment page
again immediately before creating the region-bound volume or starting compute.

## Target account and role policy

Use a normal RunPod Team. Member 1 receives the `Admin` role and Members 2-6
receive `Dev` roles, so all six members can access the shared project through
their own accounts. Every member uses an individual RunPod account and an
individual SSH public key. Members 2-6 may be invited after Member 1 prepares
the environment; invitation acceptance and a separate-account verification
must still finish before the handoff is called shared.

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
`/workspace` on one **on-demand Secure Cloud Pod**. The technical acceptance
enforces at least **44 GiB total GPU memory** and at least **40 GiB free GPU
memory immediately before model loading**. In practice, select a compatible
48 GB-or-larger GPU only after confirming current deployment capacity, the
displayed hourly price, CUDA 11.8/PyTorch 2.0.1 compatibility, and BF16 support.
Potential compatible families include A40, RTX A6000, L40S, and A100; this list
does not claim that any one of them is currently available. Do not create six
Pods or six volumes.

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

## Approval and provisioning sequence

The Team conversion is complete. Member invitations and role changes alter
account permissions, while volume creation and Pod deployment incur charges.
Execute each pending action only after Member 1 has reviewed its exact scope,
the current region and prices where relevant, and the approved budget, then
confirmed it at action time. Do not change payment settings or add funds as
part of this flow unless Member 1 separately authorizes that action.

After that confirmation:

1. Find one region that simultaneously offers a 120 GB Standard Network Volume
   and actual deployment capacity for a compatible GPU. Treat `Low` on the
   storage page as a lead only; the final deployment page must show capacity.
2. Record the exact GPU, region, displayed compute price, estimated storage
   price, and intended maximum runtime for review.
3. Confirm the existing Team still lists Member 1 as Admin. This was complete
   on 2026-09-25; do not create another Team.
4. Create one 120 GB Standard Network Volume in the selected region and mount
   it at `/workspace` when deploying the single target Pod.
5. Run bootstrap with `--download-model-if-missing`. It downloads only missing
   files from the pinned `popo28/Venus-Q-Stage1` revision into
   `/workspace/models/Venus-Q-Stage1`, using
   `/workspace/phase3/cache/huggingface` as the persistent cache.
6. Verify the full 22-file snapshot: revision
   `0f5c00c8d07ba889e9c5d12f828129dc322aae6a`, manifest SHA-256
   `13dbd18f9848ecdb7520bfe48af2704149435f2895fc67593df31c7552405298`,
   19,315,474,586 total bytes, ten safetensor shards totaling 19,312,732,032
   bytes, and exact per-file hashes. Bootstrap writes `VENUS_MODEL_SOURCE.json`
   and verifies that every pinned file, the marker, and model root are
   read-only.
7. Run the Member 1 technical acceptance and Member 1 clean-shell verification.
8. Invite Members 2-6 as Dev. Each member registers their own public key.
9. Member 2 performs the separate-account verification, then Member 1 stops the
   target GPU Pod and verifies the stopped state. Retain the shared volume for
   the later Phase 3 work.

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

The Network Volume is working storage, not the only backup. Push source and
redacted reports to Git, and copy irreplaceable private evidence to a separate
access-controlled team location. Model files and caches remain replaceable from
their pinned sources. Recheck the approved spending limit before each compute
window.
