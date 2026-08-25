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

**Venus Training Stage**:
One of the original Venus framework's two stages: aesthetic-guidance capability building or aesthetic-cropping activation. Use this term only for the original Venus training pipeline.
_Avoid_: Project Phase 1, Project Phase 2, Project Phase 3

**Venus Baseline**:
The reproduced original Venus capability without explicit emotion supervision. It establishes what the existing system can provide before the emotion-aware extension.
_Avoid_: Emotion-aware model, trained comparison method

**Emotion Prompt Baseline**:
The same frozen model guided by a structured prompt to produce emotion recognition, visual evidence, guidance, and cropping outputs without additional model training.
_Avoid_: Main trained method, neural-network layer

**Emotion Prompt Layer**:
The no-training prompting stage in the project workflow. "Layer" describes its role in the comparison pipeline, not a new trainable architectural layer.
_Avoid_: Trainable adapter, model architecture layer

**Emotion-aware LoRA/QLoRA**:
The main trained adaptation method, using emotion-supervised examples to specialise the selected Venus-compatible backbone for emotion-aware outputs while keeping most base-model weights frozen. Its primary artifact is a lightweight adapter loaded into or merged with Venus, rather than a standalone tiny model.
_Avoid_: Prompt-only baseline, full-model retraining, standalone emotion model

**A/B/C Evaluation**:
The fair comparison of the Venus Baseline, Emotion Prompt Baseline, and Emotion-aware LoRA/QLoRA on the same test images and criteria. It verifies how well the adaptations achieve the Project Aim; it is not the Project Aim itself.
_Avoid_: Project purpose, three separate final products

## Supervision and Outputs

**Emotion Supervision**:
Training or verified evaluation information that explicitly connects an image with an emotion interpretation and, where available, supporting evidence and guidance.
_Avoid_: Prompt wording alone, unverified model-generated labels

**Bridge Set**:
A small human-verified set connecting emotion labels with visual evidence, aesthetic guidance, and cropping recommendations across otherwise separate emotion and aesthetic datasets.
_Avoid_: A new large-scale dataset, a replacement for all source datasets

**Visual Evidence**:
Observable image elements such as composition, colour, lighting, subject placement, expression, or background that support an emotion interpretation.
_Avoid_: Unsupported emotion claims, generic aesthetic praise

**Emotion-consistent Guidance**:
Actionable aesthetic advice that accounts for the recognised emotion and explains how an edit, composition change, lighting adjustment, or crop would preserve or strengthen it.
_Avoid_: Generic improvement suggestions unrelated to the recognised emotion

**Q-Former Extension**:
An optional later architectural experiment considered only if time and resources permit. It is not required for the core emotion-aware route.
_Avoid_: Main method, mandatory first-stage component
