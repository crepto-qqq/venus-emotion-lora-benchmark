# Pinned Qwen-VL fine-tuning integration

Member 1 does not vendor or edit an uncontrolled copy of the upstream
fine-tuning repository. The runtime fetches
`cognitedata/Qwen-VL-finetune` at commit
`efa37ba284d56192b246d9b4ed5d3668c1abd163`, verifies that commit, and applies
[`patches/qwen-vl-finetune-efa37ba-phase3.patch`](patches/qwen-vl-finetune-efa37ba-phase3.patch).

The patch makes three compatibility corrections:

1. passes the selected computation dtype to `from_pretrained`, so `--bf16`
   controls model loading rather than only later computation;
2. freezes the complete visual module when LoRA and `fix_vit` are enabled;
3. replaces the model-path substring heuristic for `modules_to_save` with an
   explicit optional argument whose default is `None`.

It intentionally leaves the upstream JSON-array conversation loader, training
loop, optimizer, checkpoint behavior, QLoRA path, and default LoRA targets
unchanged. The Member 1 smoke configuration is technical infrastructure only;
Member 3 owns every formal training choice. The smoke verifies BF16 floating
base parameters and BF16 input entering the PEFT wrapper while keeping the
trainable LoRA adapter parameters in FP32. It expects exactly 128 matched target
modules and 3,506,176 trainable parameters. These observations do not claim
that every internal LoRA matrix operation runs in BF16.

Verify the patch against a clean checkout:

```bash
git clone https://github.com/cognitedata/Qwen-VL-finetune.git /tmp/qwen-vl-finetune
git -C /tmp/qwen-vl-finetune checkout --detach efa37ba284d56192b246d9b4ed5d3668c1abd163
git -C /tmp/qwen-vl-finetune apply --check \
  "$PWD/phase3/upstream/patches/qwen-vl-finetune-efa37ba-phase3.patch"
```

Expected patch SHA-256:

```text
be9e14a60f4f2189a108aa1f46a7384616bfdd527ca6c0a41a4c175a92172eba
```
