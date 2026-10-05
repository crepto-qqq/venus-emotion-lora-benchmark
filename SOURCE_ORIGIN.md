# Source origin and history filtering

## Original repository

This portfolio was derived from the private University of Sydney GitHub Enterprise repository for SOFT3888 team `TU12_01_P15`.

The filtered import used the team's remote-tracking `origin/main` snapshot at original private commit `f23cd02`. At the time of export, that snapshot contained 61 commits, 695 tracked files, and approximately 224.1 MB of repository content.

## Public source history

The source-history branch in this repository contains:

- 29 retained commits attributable to Tian Liao;
- 97 retained files at the filtered source snapshot;
- approximately 0.66 MB of retained source and documentation;
- original author name, author date, commit date, message, and parent ordering for every non-empty retained commit.

Tian's original university email was rewritten to `crepto-qqq@users.noreply.github.com`. Personal, university, and local-host email addresses were removed. Because filtering changes commit contents and parent graphs, all public commit hashes differ from their private originals.

## Retained path groups

- `src/**/*.py`
- `tests/**/*.py`
- `scripts/**/*.sh` and related runbook documentation
- `configs/**/*.json`
- `environment/**`
- `phase3/tools/**/*.py`
- `phase3/tests/**/*.py`
- `phase3/scripts/**/*.sh`
- `phase3/contracts/**/*.json`
- selected `phase3/configs/**`, `phase3/environment/**`, and handoff/runbook Markdown
- project context and roadmap documentation needed to interpret the retained work

## Removed from every retained commit

- `External/**` and every bundled upstream/third-party source tree;
- `data/**`, `results/**`, Phase 3 releases, reports, and Word exports;
- image datasets, annotations, generated outputs, hashes tied to private releases, and model/checkpoint files;
- group contracts, meeting notes, client or course administration records, and unrelated assessment artefacts;
- upstream patch bundles and any file that could redistribute third-party code;
- Git references and commits that became empty after the removals.

The raw private `.git` directory was never copied into this public repository. A path-filtered branch was created, scanned, and merged into the pre-existing portfolio documentation so the public repository retains a transparent two-parent import point without exposing excluded objects.

## Authorship and permission boundary

This filtering preserves evidence of Tian's source contributions while keeping the six-person team context explicit. Team permission covers publication of the retained team-authored paths. It does not grant ownership of, or a licence to redistribute, any upstream system or dataset. See [NOTICE.md](NOTICE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
