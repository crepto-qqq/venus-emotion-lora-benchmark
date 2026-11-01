# Source provenance and refresh status

Refreshed on 2 November 2026 from the committed school repository without altering its working tree.

| Item | Recorded value |
|---|---|
| Original repository | [SOFT3888 team repository](https://github.sydney.edu.au/tsin0599/SOFT3888_TU12_01_P15) |
| Source branch | `main` |
| Source commit | `f1b6bcaa66e6da7cfccfc429051e6bcf873f87d2` |
| Original reachable commits | 84 |
| Retained source-history commits | 44 |
| Tian-authored retained commits | 40, including merges |
| Retained files | 177 |
| Filtered source tip | `3cd4b65236fee72627b08e9fa8be605499893b02` |

The earlier public snapshot came from `f23cd02`. Its publication commits and filtered history remain intact. This refresh merges a newly reviewed filtered lineage without force-rewriting the public branch. Counts above refer to the new source import, not both overlapping lineages combined.

Some later upstream commits contain only excluded demonstration/report/review artefacts and become empty after filtering. Consequently, the filtered tip can have an earlier date than the full source snapshot. The root README summarises the final source records without republishing private ratings, images or complete model answers.

## Retained material

- Team Python source, tests and scripts under `src/`, `tests/`, `scripts/` and `phase3/`.
- Schemas, selected configuration, environment definitions and engineering runbooks.
- Project context and roadmap, retained as upstream development records.
- Original author names, dates, relevant messages and retained commit ordering.

## Material excluded from the imported history

- `External/` and bundled Venus/Qwen/other third-party source or patch bundles.
- Image/annotation corpora, generated dataset releases, model/adapter weights and checkpoints.
- Raw inference outputs, private manifests, review sheets, full answers and reviewer assignments.
- Course administration, contracts, meetings, binary final reports/slides and process-record archives.
- Credentials, private contact addresses and personal machine-home paths.

Private emails are replaced consistently; Tian's source author email maps to `crepto-qqq@users.noreply.github.com`, other author/committer emails to an anonymous address. Source names and timestamps are retained. Hashes change because path filtering/redaction changes commit contents. Empty commits are pruned.

The retained source history was checked for common API-token/private-key formats and non-test private email addresses. This precaution is not a formal security certification.

## Completed project evidence

The private snapshot records formal 180-step training, checkpoint-180 acceptance on 19 October, Stage C inference and six-reviewer guidance results on 28 October. The root README separates these project outcomes from the tests executed during this source refresh, and records the evaluation limitations.

## Future synchronisation

This is a manual reviewed snapshot, not an automatic mirror or synchronising fork. The school repository link and exact source commit provide traceability. Future refreshes must fetch committed source, repeat filtering and checks, and merge the public export without force-pushing or publishing raw private refs. Local uncommitted work must be handled separately and is not included implicitly.
