# Venus Emotion-Aware Extension: Three-Phase Project Roadmap

This document records the team's agreed research workflow. "Phase 1", "Phase 2", and "Phase 3" refer to the project phases defined here, not the original Venus Stage 1 aesthetic-guidance training and Stage 2 cropping activation.

Project phases are assigned by the intervention being tested, not by execution date. Work from different phases may overlap when model, prompt, and training configurations are clearly labelled and stored separately.

In the private project, the canonical controls, dataset partitions, scoring rules, and permitted claims were defined in `Docs/EMOTION_AWARE_EXPERIMENT_PROTOCOL.md`. That assessment document is intentionally not distributed in this public portfolio; this roadmap records only the engineering context needed to interpret the retained source.

## Overall objective

Starting from a reproducible Venus baseline, test whether structured prompting and lightweight parameter adaptation improve image-emotion understanding and whether that understanding improves aesthetic guidance and cropping recommendations.

The core comparison is fixed as:

- A: official checkpoint with the official original prompt;
- B0: frozen Venus with a concise unstructured joint emotion-recognition and aesthetic-guidance prompt, used as the Phase 2 control;
- B1: frozen Venus with the structured version of the same joint task, used as the main Phase 2 intervention;
- C: Venus baseline with an emotion-aware LoRA/QLoRA adapter.

All configurations use the same test images and pinned inference settings. A preserves the official response behaviour, B0 remains free-form, and B1/C share the same structured response schema. All four receive the same guidance rubric, while exact-match recognition applies only where an eight-class label is requested.

## Phase 1: reproduce Venus at runtime

### Objective

Run the officially released Venus checkpoints and record the emotion-related capability already present in Original Venus. This establishes the shared baseline for Phases 2 and 3. Phase 1 is a runtime reproduction: it does not retrain Venus or attempt to reproduce the complete training process and every paper metric.

### Execution route

Use the local-development/cloud-compute route. This team repository is the sole source of truth for code and documents; work is developed on `tlia0262` and merged through a pull request. A Linux GPU cloud server downloads models, hosts the environment, and runs inference. Results are synchronised back to the local repository; the cloud server is neither long-term storage nor the only copy of project work.

Start with a 24GB GPU, sufficient system RAM and disk, BF16, and `batch size = 1`. If the locked formal configuration runs out of memory or cannot complete reliably, do not use CPU offload, quantisation, shorter outputs, or reduced input settings. Move directly to a 48GB GPU and continue with the same configuration.

### Main work

- Build and record a reproducible runtime environment.
- Preserve official seed `1234`, pin the exact Stage 1 and Stage 2 checkpoint revisions, and save the official generation configuration and software versions.
- Retain one formal output per image-condition rather than generating several answers and choosing the best. Report the experiment as one complete fixed-seed run and do not claim that repeated-run stability has been established.
- Run aesthetic-guidance inference with Venus-Q-Stage1 and aesthetic-cropping inference with Venus-Q-Stage2.
- Use only official original prompts in Phase 1. Any prompt that explicitly asks about image emotion belongs to Phase 2.
- Preserve the completed fixed 20-image EmoSet run as the unchanged Eval20 pilot. It covers all eight emotions and records released-model A, B0, and Stage 2 outputs.
- Use Eval20 to document runtime behaviour and motivate the primary experiment; do not overwrite it with Eval80-v1 results or describe it as representative of all EmoSet.
- For Stage 2, retain both successful and failed coordinate parses, render every valid crop privately, and compare original and cropped images for qualitative emotion-preservation analysis. EmoSet has no ground-truth crop boxes, so this route does not calculate FLMS IoU, displacement, or recall.
- Defer the proposed five-image FLMS technical smoke test. Complete the EmoSet-20 route first and then decide whether FLMS adds enough value to run.
- Save the fixed manifest, original prompts, inference settings, model versions, raw outputs, and baseline observations.
- Do not use 8-bit or 4-bit quantisation for the formal Phase 1 Venus Baseline. Any future quantised experiment must be labelled as a quantised baseline and must not be presented as the official BF16 baseline.
- State explicitly that Phase 1 does not retrain the model or reproduce the complete paper training pipeline and paper metrics.

### Deliverables

- A runnable Original Venus baseline.
- A repeatable inference and evaluation workflow.
- Baseline emotion-related results, environment configuration, and known limitations.

### Completion criteria

Both official Stage 1 and Stage 2 checkpoints load successfully and each processes all 20 fixed images. Stage 2 coordinate parse failures remain in the result rather than being silently removed. Original prompts, raw outputs, and runtime configuration are saved completely, and no image is replaced because of model performance. Phase 1 measures a baseline and has no minimum emotion-aware passing score.

## Phase 2: structured prompt experiment

### Objective

Without training or modifying model weights, first measure Venus on a concise unstructured joint emotion-recognition and aesthetic-guidance task, then test whether explicit grounding rules and a structured response schema provide additional improvement.

### Structured output direction

The prompt should guide the model to provide, in order:

1. the primary emotion communicated by the image, with confidence or uncertainty when appropriate;
2. observable visual evidence, such as composition, colour, lighting, facial expression, and background;
3. actionable aesthetic advice consistent with that emotion;
4. when appropriate, a crop recommendation and rationale that preserve or strengthen the emotion.

### Experimental principles

- Keep model weights frozen.
- Apart from the prompt, keep inputs, inference settings, and evaluation conditions as close to Phase 1 as possible.
- Label the concise unstructured joint prompt B0 and the structured joint prompt B1. Both belong to Phase 2.
- Complete blind and reveal review, freeze prompts, settings, scoring handbook, and integrity hashes, and obtain team approval before new Eval80 inference.
- Run A first on all 80 images and continue only after technical acceptance. Run B0 and B1 later without using A performance to revise the frozen experiment.
- Use exact-match emotion accuracy for B0, B1, and C and the shared five-dimension guidance rubric for A, B0, B1, and C. Keep the two 0–10 scores separate.
- Evaluate emotion recognition, whether evidence is grounded in the image, and whether advice is genuinely influenced by the emotion interpretation.

### Deliverables

- A versioned structured prompt template.
- Paired Original Venus, unstructured-prompt, and structured-prompt outputs.
- A/B0/B1 comparisons and error-case analysis.

### Completion criteria

The experiment can answer whether structured prompting gives a stable improvement, which metrics or image types improve, and which problems remain unsolved by prompting alone.

## Phase 3: emotion-aware LoRA/QLoRA adaptation and integration

### Objective

Use available public datasets to construct emotion supervision, train a lightweight LoRA/QLoRA adapter, integrate it with Venus, and test whether trained adaptation is more effective than Original Venus and structured prompting.

The project's "small model" or "patch" is canonically an Emotion-aware LoRA/QLoRA adapter. It is not a standalone complete model; it is a small set of trainable parameters loaded into a Venus-compatible base model at inference or merged before deployment when the method permits.

### Main work

- Audit public datasets for task definition, annotation quality, licence, and available fields.
- Convert source data into emotion-supervision examples compatible with Venus input and output formats.
- Build BridgeTrain-v1 from 400 class-balanced EmoSet images after excluding every Eval20 and Eval80-v1 ID. All 400 images receive classification records, and a balanced 80-image subset receives human-audited joint-guidance records.
- Keep an independent balanced validation set for adapter selection. Eval80-v1 must never be used for hyperparameter, epoch, prompt, or dataset-size decisions.
- Create leakage-free training, validation, and test splits.
- Train and select a LoRA/QLoRA adapter while recording hyperparameters, base model, and checkpoints.
- Load or merge the adapter into the Venus inference workflow as appropriate.
- Complete A/B0/B1/C comparisons and ablation analysis under unified test conditions.

### Deliverables

- Data conversion and quality-checking workflow.
- Emotion-aware LoRA/QLoRA adapter weights.
- Integrated Venus inference configuration.
- A/B0/B1/C evaluation, ablations, and failure-case analysis.

### Completion criteria

The experiment can answer whether lightweight adaptation produces repeatable gains in emotion recognition, visual grounding, emotion-consistent aesthetic guidance, or cropping, and whether those gains justify the additional training cost relative to prompting.

## Unified evaluation dimensions

All three phases share the following evaluation directions:

- Reviewer consistency: the six reviewers approve one shared handbook before it is frozen, then each applies it to a fixed image partition that remains unchanged across A, B0, B1, and C.
- Blind randomisation: hide A/B0/B1/C identities and prompts, randomise condition order per image, and assign all four responses for an image to the same reviewer.
- Emotion recognition: whether the emotion judgement is correct, sufficiently specific, and appropriately calibrated.
- Visual grounding: whether the emotion explanation is supported by genuinely visible evidence.
- Guidance quality: whether advice is specific, actionable, and consistent with the recognised emotion.
- Cropping quality: whether the crop improves composition while preserving or strengthening emotional expression.
- Stability: whether performance remains reliable across samples, prompt perturbations, and repeated runs when those are evaluated.
- Cost: inference expense, training resources, adapter size, and integration complexity.

### Unified 0–10 scoring protocol

Emotion Recognition Score is exact-match eight-class accuracy expressed on a 0–10 scale: `10 × correct / total`. Reports also retain the raw fraction, per-class accuracy, and confusion matrix. It applies to B0, B1, and C; A is not assigned this score because its official prompt is unchanged.

Emotion-Aware Guidance Score is the sum of five dimensions, each scored `0`, `1`, or `2`, producing a direct 0–10 total:

1. visual grounding;
2. emotion-aesthetic linkage;
3. aesthetic validity;
4. actionability; and
5. emotion preservation.

All 80 images receive the guidance rubric under blind randomisation for A, B0, B1, and C. Six reviewers receive fixed, non-overlapping image partitions of 14/14/13/13/13/13 and score all four anonymous condition responses for every assigned image. A wrong emotion label does not automatically zero the guidance dimensions, and the recognition and guidance scores are never averaged together. Because no item is double-scored, the report must not claim conventional inter-rater reliability.

## Data and version control

- All project-owned, repository-shared files and fields use English. Unmodified third-party material under `External/` remains as released upstream.
- Git stores code, prompts, environment and inference configuration, the fixed manifest, anonymisation mappings, raw text and coordinate outputs, score sheets, and summary results.
- Git does not store Venus weights or commit or redistribute EmoSet images. Cloud models and datasets are replaceable runtime resources, not the only stored copy of project work.
- The manifest uses traceable dataset image identifiers and labels. The report must cite EmoSet correctly and respect its non-commercial research terms.

## Current research hypotheses

- Phase 1 will provide a credible boundary for Original Venus capability.
- Phase 2 may improve output structure, explicit emotion reasoning, and advice relevance at the lowest cost.
- Phase 3 may supply stable emotion representations that prompting alone cannot provide, but any advantage over prompting must be demonstrated under unified evaluation rather than assumed.
