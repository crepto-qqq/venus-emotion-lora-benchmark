# Member 3 LoRA experiment plan

## Scope
Member 3 trains only a BF16 PEFT LoRA adapter on the pinned Venus Stage 1 checkpoint and BridgeTrain-v1. Eval20 and Eval80 are never training, pilot-selection, validation-selection, or hyperparameter inputs. Condition C inference belongs to Member 4.

## Offline gates
1. Verify the five raw BridgeTrain-v1 release hashes and `full_dataset_ready=true` while preserving dataset-level `formal_training_authorized=false`.
2. Fetch the exact pinned Qwen checkout and apply the reviewed patch. Run `inventory_qwen_interface.py`; pilot execution is blocked until `source_inventory_state=verified`.
3. Run read-only `inventory_lora_modules.py`. It must report concrete language-model targets, zero visual targets, observed dtypes, and a canonical report hash.
4. Copy `lora-pilot-v1.json` to an inventory-verified version and bind `target_modules`, `matched_module_count`, and `inventory_sha256`. Do not guess these values.
5. Generate the deterministic pilot subset and expand the at-most-three learning-rate candidates.

## Pilot design
The three candidates differ only in learning rate: `5e-5`, `1e-4`, and `2e-4`. All use rank 8, alpha 16, dropout 0.05, BF16 base/input, FP32 LoRA trainables, micro-batch 1, gradient accumulation 8, single GPU, frozen visual encoder, max length 2048, and at most 30 optimizer steps for the pilot. Rank 2 is explicitly rejected because it belongs only to Member 1's infrastructure smoke.

The default validation input is the full 96-conversation BridgeTrain-v1 validation split. Candidate selection uses the predeclared rule in `lora-experiment-matrix-v1.json`; no Eval20/Eval80 result may affect it.

## Paid-run stop conditions
Before pilot execution the user must supply GPU allowance, maximum displayed hourly price, total pilot budget, maximum wall-clock time, maximum attempts, and OOM policy. Nonfinite loss, zero supervised tokens, unexpected trainable parameters, identity-hash drift, OOM under an unauthorized resource change, time breach, or cost breach stops the attempt.

## Formal training
Member 3B reviews pilot evidence. A formal config is created only after one pilot configuration is selected and separate formal-run resource limits are supplied by the user. `validate_lora_config.py --mode formal` requires an approved review that binds the exact canonical formal-config SHA-256. The primary experiment remains capped at three epochs.

## Completion
Member 3 is technically complete only after formal training evidence exists, Member 3B records `training-acceptance-v1.json`, one adapter/checkpoint is selected by BridgeTrain-v1 validation evidence, and clean-process reload is proven. Repository-only implementation is not training completion.
