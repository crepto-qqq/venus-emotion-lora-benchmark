# Emotion-Aware Venus Extension

This context defines the team's canonical language for extending Venus with explicit image-emotion understanding. It keeps the project aim, adaptation methods, outputs, and evaluation roles distinct.

## Purpose

**Project Aim**:
Adapt Venus into an emotion-aware multimodal system that recognises the emotion conveyed by an image, grounds that interpretation in visual evidence, and uses it to provide emotion-consistent aesthetic guidance and cropping recommendations.
_Avoid_: Proving that LoRA beats prompting, style/vibe-specific guidance as the primary goal

**Emotion-aware**:
Able to recognise image emotion, identify the visual evidence supporting it, and use that understanding in aesthetic guidance and cropping. It means more than merely mentioning emotion words in generated feedback.
_Avoid_: Style-specific, vibe-specific, sentiment-only

## Study Configurations

**Three-Phase Project Workflow**:
The team's research sequence: reproduce Venus, test a structured emotion prompt, then train and integrate a lightweight emotion-aware LoRA/QLoRA adapter. These are project phases, not the two training stages defined by the original Venus framework.
_Avoid_: Venus Stage 1/2, three independent projects

**Project Phase Boundary**:
Experiments are assigned to a project phase by the intervention being tested, not by execution date: official Venus prompts belong to Phase 1, any explicit emotion prompting of the frozen model belongs to Phase 2, and emotion-specific LoRA/QLoRA adaptation belongs to Phase 3. Work from different phases may overlap in time.
_Avoid_: Calendar-only phase, labelling experiments by when they ran

**Venus Training Stage**:
One of the original Venus framework's two stages: aesthetic-guidance capability building or aesthetic-cropping activation. Use this term only for the original Venus training pipeline.
_Avoid_: Project Phase 1, Project Phase 2, Project Phase 3

**Runtime Reproduction Baseline**:
The Project Phase 1 reproduction level: run the official released Venus checkpoints without retraining and characterise their existing emotion-aware behaviour under repeatable conditions. It establishes the comparison baseline rather than reproducing the paper's training process or headline results.
_Avoid_: Training reproduction, full paper reproduction, new emotion-aware method

**Pinned Inference Run**:
A formal evaluation pass that fixes the exact model snapshots, random seed, generation settings, and environment before execution, then retains one output per image-condition rather than selecting among retries. The Venus Baseline preserves the released BF16 precision even when a larger GPU is required.
_Avoid_: Floating latest version, best-of-many output, quantised Venus Baseline

**Venus Baseline**:
The original released Venus checkpoints evaluated with their official prompts and without explicit emotion prompting or emotion-specific adaptation. It establishes what the existing system can provide before the emotion-aware extension.
_Avoid_: Emotion-aware model, trained comparison method, retrained Venus

**Emotion Prompt Baseline**:
The same frozen model explicitly asked about image emotion without additional training. A minimal unstructured emotion question is the Phase 2 control, while the structured emotion prompt is the main Phase 2 intervention.
_Avoid_: Phase 1 official-prompt baseline, trained method, neural-network layer

**B0 Plain Emotion Baseline**:
The frozen released Venus Stage 1 model asked a plain eight-class emotion-recognition question. It measures direct recognition without structured reasoning instructions.
_Avoid_: Original aesthetic prompt, structured prompt condition

**B1 Structured Emotion Baseline**:
The same frozen released Venus Stage 1 model given the structured emotion-aware system prompt while keeping the user task, output constraints, images, and generation settings fixed.
_Avoid_: Trained adapter, changed model weights

**Emotion Prompt Layer**:
The no-training prompting stage in the project workflow. "Layer" describes its role in the comparison pipeline, not a new trainable architectural layer.
_Avoid_: Trainable adapter, model architecture layer

**Emotion-aware LoRA/QLoRA**:
The main trained adaptation method, using emotion-supervised examples to specialise the selected Venus-compatible backbone for emotion-aware outputs while keeping most base-model weights frozen. Its primary artifact is a lightweight adapter loaded into or merged with Venus, rather than a standalone tiny model.
_Avoid_: Prompt-only baseline, full-model retraining, standalone emotion model

**A/B0/B1/C Evaluation**:
The fair comparison of A (official-prompt Venus Baseline), B0 (plain emotion recognition), B1 (structured emotion prompting), and C (the joint emotion-aware LoRA using exactly the B1 prompt and settings). B0 versus B1 isolates the prompt effect; B1 versus C isolates the adapter effect. B0 and B1 are both Phase 2 conditions, so the project still has three phases.
_Avoid_: Project purpose, unrelated prompts or evaluation sets for each condition

## Supervision and Outputs

**Eval20 Pilot**:
The original fixed 20-image EmoSet evaluation and its completed A, B0, and Stage 2 outputs. It remains an unchanged pilot and must not be overwritten or presented as the primary experiment.
_Avoid_: Primary evaluation, LoRA training data, retroactively rescored result

**Eval80-v1**:
The primary exploratory evaluation set: 80 EmoSet images balanced at 10 per emotion class. It is reviewed and frozen before new inference, reused for B0, B1, and C, and permanently excluded from all training and validation data together with Eval20.
_Avoid_: Representative sample of all EmoSet, training data, repeatedly revised test set

**Emotion-Aware Rubric**:
The shared scoring protocol uses exact-match emotion accuracy expressed on a 0–10 scale and five guidance dimensions scored 0/1/2 whose sum is directly reported on a 0–10 scale. The guidance dimensions are visual grounding, emotion-aesthetic linkage, aesthetic validity, actionability, and emotion preservation.
_Avoid_: A subjective single overall mark, paper benchmark metric, different criteria per condition

**Emotion-Preserving Crop Evaluation**:
The current Venus Stage 2 evaluation on the Fixed Emotion Evaluation Set: retain both valid and invalid coordinate outputs, render each valid predicted crop beside its original image, and manually score whether the crop preserves or strengthens the labelled emotion. Because EmoSet has no ground-truth crop boxes, this route does not use FLMS IoU, displacement, or recall metrics.
_Avoid_: Full cropping-benchmark reproduction, silently dropping parse failures, scoring only the coordinate text

**Blind Randomised Evaluation**:
The human-scoring protocol in which A, B0, B1, and C identities are hidden and their outputs are randomly ordered before the same evaluator applies the shared rubric; condition labels are restored only for analysis. It reduces expectation and order bias but does not provide independent inter-rater agreement.
_Avoid_: Evaluator sees method names, fixed method order, claiming multi-rater reliability

**Emotion Supervision**:
Training or verified evaluation information that explicitly connects an image with an emotion interpretation and, where available, supporting evidence and guidance.
_Avoid_: Prompt wording alone, unverified model-generated labels

**Bridge Set**:
A small human-verified set connecting emotion labels with visual evidence, aesthetic guidance, and cropping recommendations across otherwise separate emotion and aesthetic datasets.
_Avoid_: A new large-scale dataset, a replacement for all source datasets

**BridgeTrain-v1**:
The initial joint-LoRA training set: 400 class-balanced EmoSet images provide classification records, and a balanced 80-image subset also provides human-audited joint guidance records. It is completely disjoint from Eval20 and Eval80-v1.
_Avoid_: Evaluation data, unreviewed model-generated targets

**Visual Evidence**:
Observable image elements such as composition, colour, lighting, subject placement, expression, or background that support an emotion interpretation.
_Avoid_: Unsupported emotion claims, generic aesthetic praise

**Emotion-consistent Guidance**:
Actionable aesthetic advice that accounts for the recognised emotion and explains how an edit, composition change, lighting adjustment, or crop would preserve or strengthen it.
_Avoid_: Generic improvement suggestions unrelated to the recognised emotion

**Q-Former Extension**:
An optional later architectural experiment considered only if time and resources permit. It is not required for the core emotion-aware route.
_Avoid_: Main method, mandatory first-stage component

## Canonical Experiment Protocol

Dataset selection, review checkpoints, A/B0/B1/C controls, BridgeTrain construction, the 0–10 Emotion Recognition Score, the five-dimension 0–10 Emotion-Aware Guidance Score, training guardrails, and permitted success claims are defined in [`Docs/EMOTION_AWARE_EXPERIMENT_PROTOCOL.md`](Docs/EMOTION_AWARE_EXPERIMENT_PROTOCOL.md). That protocol takes precedence over earlier planning drafts when terminology or evaluation rules differ.
