# Architecture and completed experiment flow

## Data and training

The annotation pipeline prepares structured records, enforces emotion quotas and schemas, removes duplicate images and checks Eval20/Eval80 isolation. BridgeTrain separates 400 training images from 80 validation images.

The LoRA workflow obtains the separately licensed base model, checks visual inputs and target modules, freezes the visual backbone and prepares PEFT adapters. Formal training adds configuration checks, microbatch/optimiser telemetry, checkpoint writing, clean-process reload validation and resource/watchdog controls.

The successful formal run completed 180 optimiser steps and produced checkpoints 60, 120 and 180. Validation selected checkpoint 180. Earlier zero-step forward/backward checks remain distinct from that completed training run.

## Frozen evaluation

Eval80 uses consistent input identities, model/generation configuration and output validation. Conditions A, B0 and B1 vary prompting; C adds the selected adapter to B1's prompt and generation protocol. Emotion recognition covers 80 images, while the final guidance scoring scope covers 78.

Quality/preference tooling checks output structure, prepares anonymous scoring, validates submitted records and aggregates results. The six-person review covers 312 answers and 1,560 dimension scores. Human results, automated judge comparisons and recognition accuracy remain separate measures.

## Public and private boundaries

This repository includes orchestration, validation, schemas and testable engineering components. Images, private frozen manifests, generated releases, model/adapter weights, complete answers and reviewer records remain outside the public checkout. Full historical integration tests require some of these inputs and the original runtime; the root README gives a verified self-contained public subset.

Configuration and deterministic records support reproducibility, but the executed formal run did not explicitly seed adapter initialisation before PEFT creation. Bitwise reproduction and complete independence of team review are not claimed.
