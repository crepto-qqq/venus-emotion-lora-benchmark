# Eval80 RunPod execution wrappers

This directory archives the exact shell scripts used for the accepted
Eval80-v1 A, B0, and B1 cloud runs. During those runs the scripts were deployed
to `/workspace/run_scripts/`, outside the Git checkout at
`/workspace/project/`. They were copied into this repository after the runs so
that the complete execution chain is reviewable and reproducible.

These files are orchestration wrappers, not the model implementation. The
formal inference path is:

```text
eval80_*_run.sh
  -> python -m src.eval80.cli run
  -> src.eval80.cli dispatches the `run` command
  -> src.eval80.inference.run_command
  -> tokenizer.from_list_format(image + frozen prompt)
  -> model.chat(...)
  -> records.jsonl, run_config.json, environment.json, and summary.json
```

The validation path is:

```text
eval80_*_validate.sh
  -> python -m src.eval80.cli validate-output
  -> src.eval80.validation.validate_output_command
  -> src.eval80.validation.validate_run
  -> validation.json
```

## Script responsibilities

| Script | Responsibility |
| --- | --- |
| `eval80_a_preflight.sh` | Checks the A40/BF16 runtime, frozen hashes, manifest, checkpoint provenance and shards, and refuses to overwrite the A output directory. |
| `eval80_a_run.sh` | Activates the pinned environment and invokes the repository CLI for Condition A while teeing the full log. |
| `eval80_a_validate.sh` | Runs the repository output validator for the completed A attempt. |
| `codex_eval80_a_final_gate.sh` | Recomputes counts and hashes and prints the final A technical-acceptance decision. The historical filename is preserved verbatim. |
| `eval80_b_preflight.sh` | Performs the same frozen checks for B0/B1 and additionally requires the accepted A result. |
| `eval80_b_run.sh` | Accepts `B0` or `B1`, then invokes the same repository inference code with the selected frozen prompt. |
| `eval80_b_validate.sh` | Runs the repository output validator for the selected B condition. |
| `codex_eval80_b_final_gate.sh` | Recomputes counts and hashes and prints the selected B condition's technical-acceptance decision. The historical filename is preserved verbatim. |

All paths intentionally reflect the RunPod layout used by the accepted runs.
The scripts contain no credentials, private labels, or EmoSet image pixels.

## Archived source hashes

The following SHA-256 values match the copies downloaded from the accepted
RunPod attempts:

```text
codex_eval80_a_final_gate.sh  dc33a5492e4463a178fbe9af957bf4378d3450831e94baa9376d502d2b2bbd76
eval80_a_preflight.sh         b2d9ad19e6bda3dd726a2aaf17f65df94dccd8be57a0018d28afa6891dc88c85
eval80_a_run.sh               7be7be0b4721bb4d6042d95b31c15e76cf1c625906b10f73a5493327a2413829
eval80_a_validate.sh          5cc007b15d94189be5e386c718e3559a4b4a2be52d9ee681b9d4fe990917495e
codex_eval80_b_final_gate.sh  a5cb3d531e9f47f502ea7b9800745c3920bc9c368296aceb99ba40d0d64ecef6
eval80_b_preflight.sh         46039ebc9747f2c513e636dce60b2467c0792d555b47373424901637abcec56f
eval80_b_run.sh               41b131a2fe2e8b98429111d8d154ccbb19568ee94d1cd23abdeb2cf109b919c6
eval80_b_validate.sh          fa1de9ea44a16cfecca6d281306aa58ce86402b471744d6f4cf332af9db2a658
```

Use `bash scripts/eval80/runpod/<script>.sh` from the repository checkout when
the same `/workspace` layout is present. The accepted historical logs retain
the original `/workspace/run_scripts/<script>.sh` invocation.
