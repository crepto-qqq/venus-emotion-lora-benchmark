# Member 3A runbook

Run from the repository root.

```bash
python phase3/tools/hash_lora_config.py phase3/configs/lora-pilot-v1.json
python phase3/tools/make_lora_pilot_subset.py
python phase3/tools/expand_lora_candidates.py
python -m unittest phase3.tests.test_member3_tools -v
```

On the retained RunPod volume, first inventory the exact patched parser and live model without an optimizer:

```bash
python phase3/tools/inventory_qwen_interface.py
python phase3/tools/inventory_lora_modules.py \
  --report /workspace/phase3/member3/inventory/lora-module-inventory.json
```

Create a new inventory-verified pilot config; bind the concrete target suffixes, inventory report hash, and observed match count. Do not edit the pending template to claim facts not observed at runtime.

After the user supplies pilot resource limits, validate each expanded candidate and create an immutable attempt record. `run_lora_pilot.py` is dry-run unless `--execute` is supplied. The launcher runs the validator before invoking upstream `finetune.py`.

After each attempt, evaluate predictions only on BridgeTrain-v1 validation with `evaluate_lora_pilot.py`, then collect explicit runtime facts with `collect_lora_pilot_evidence.py`. Raw weights/logs stay outside Git; only redacted hashes and metrics enter Git.

For formal training, create a new formal config from the selected pilot configuration, replace pilot step limits with the reviewed formal schedule (maximum three epochs), collect separate formal resource limits from the user, obtain Member 3B's exact approved review, and invoke `run_lora_training.py --mode formal`. Preserve every candidate checkpoint until Member 3B selects one.
