# Phase 3 experiment record template

Use one copy of this template for each training, evaluation, or inference run.
Store the completed record beside the run's Git-safe outputs. Do not put images,
model weights, adapters, checkpoints, credentials, signed URLs, private host
details, or raw logs in Git.

This template does not authorize training. Member 3A uses it only after the
full dataset passes its release checks. A paid pilot or formal run also
requires the user's stage-specific GPU and budget decision. Member 3B is the
internal reviewer: the pilot review authorizes the exact formal configuration,
and no approval from another team member is required.

## Identity and ownership

| Field | Value |
| --- | --- |
| Experiment ID | `phase3-YYYYMMDD-<member>-<purpose>-<attempt>` |
| Owner | `<member-id>` |
| Reviewer | `<member-id>` |
| Run type | `pilot training / formal training / inference / formal Eval80 / API` |
| Start/end time (UTC) | `<ISO-8601>` |
| Purpose or hypothesis | `<one testable sentence>` |
| Status | `planned / running / passed / failed / stopped` |

## Immutable inputs

| Field | Value |
| --- | --- |
| Project branch and commit | `<branch>` / `<40-character commit>` |
| Venus source commit | `44e2971b7fe41f88a88c7e0ed8bf4c85eeb0155f` |
| Qwen source commit | `efa37ba284d56192b246d9b4ed5d3668c1abd163` |
| Qwen patch SHA-256 | `be9e14a60f4f2189a108aa1f46a7384616bfdd527ca6c0a41a4c175a92172eba` |
| Base model revision | `0f5c00c8d07ba889e9c5d12f828129dc322aae6a` |
| Dataset release ID | `<release-id>` |
| Dataset manifest SHA-256 | `<sha256>` |
| Train/validation counts | `<counts by class and task>` |
| Eval20/Eval80 exclusion check | `passed / failed` |
| Environment freeze SHA-256 | `<sha256>` |

## Runtime and cost

| Field | Value |
| --- | --- |
| GPU model and count | `<model>` / `<count>` |
| Displayed hourly price | `USD <value>/hour` |
| Pod start/stop time (UTC) | `<ISO-8601>` |
| Billed duration and cost | `<duration>` / `USD <value>` |
| Cost class | `training / non-training` |
| User budget decision reference | `<date and Git-safe summary, or not applicable>` |
| Stage total budget | `USD <value>, or not applicable` |
| Maximum hourly price | `USD <value>/hour, or not applicable` |
| Wall-clock / attempt limits | `<values, or not applicable>` |
| OOM / extension decision | `<user-set action, or not applicable>` |
| Cumulative non-training spend | `USD <value> of USD 20, or not applicable` |
| Network Volume | `phase3-shared-120gb` (`bk4fycduml`), `US-NE-1` |
| Python / PyTorch / CUDA | `<versions>` |
| Random seed or seed set | `<value>` |

## Formal LoRA configuration

| Field | Value |
| --- | --- |
| Target modules | `<exact module names>` |
| Rank / alpha / dropout | `<r>` / `<alpha>` / `<dropout>` |
| Bias policy | `<value>` |
| Base and adapter dtypes | `<base>` / `<adapter>` |
| Trainable / total parameters | `<counts and percentage>` |
| Vision encoder state | `frozen / trainable, with measured counts` |
| Batch and accumulation | `<micro-batch>` / `<steps>` |
| Epochs or max steps | `<value>` |
| Optimizer and learning rate | `<values>` |
| Scheduler / warmup | `<values>` |
| Gradient clipping | `<value>` |

## Execution

```text
Exact command:
<command with secrets and signed URLs removed>

Run directory:
/workspace/phase3/runs/<member-id>/<experiment-id>/

Report directory:
/workspace/phase3/reports/<member-id>/<experiment-id>/
```

Record checkpoint intervals, early-stop rules, monitored metrics, and any
reviewed deviation from the frozen configuration before the run begins. A
budget, GPU, or duration change for paid training requires a new user decision.

## Results and validation

| Check | Result |
| --- | --- |
| Preflight and dataset contracts | `passed / failed` |
| Base parameters frozen as intended | `passed / failed` |
| Intended LoRA gradients finite/non-zero | `passed / failed` |
| Final training/validation metrics | `<metric names and values>` |
| Adapter save and checksum | `<path>` / `<sha256>` |
| Clean-process adapter reload | `passed / failed` |
| Member 3B checkpoint selection | `<checkpoint ID and validation-only basis, or not applicable>` |
| Inference output validation | `passed / failed / not applicable` |
| Evaluation output path and checksum | `<path>` / `<sha256>` |

## Outcome and handoff

- Decision: `<accept / reject / repeat / investigate>`
- Review artifact: `phase3/reviews/member3/pilot-review-v1.json`,
  `phase3/reviews/member3/training-acceptance-v1.json`, or `not applicable`
- Evidence supporting the decision: `<short factual summary>`
- Failures or deviations: `<what happened and where the evidence is stored>`
- Next owner and action: `<member-id and bounded task>`
- Shared inputs changed: `no`, or `<approved change and review reference>`
- Pod stopped and state confirmed: `yes / no`
- Git-safe artefacts reviewed for secrets and private data: `yes / no`
