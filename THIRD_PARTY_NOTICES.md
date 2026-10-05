# Third-party notices

This project was engineered around third-party research systems and assets. Their names are retained for accurate attribution and reproducibility; their source code, datasets, model weights, and generated assets are **not** bundled in this repository.

## Referenced upstream systems

| System | Reference used by the private project | Included here? |
|---|---|---|
| Venus CVPR 2026 | [PKU-ICST-MIPL/Venus_CVPR2026](https://github.com/PKU-ICST-MIPL/Venus_CVPR2026), pinned private-project revision `44e2971b7fe41f88a88c7e0ed8bf4c85eeb0155f` | No |
| Qwen-VL fine-tuning utilities | [cognitedata/Qwen-VL-finetune](https://github.com/cognitedata/Qwen-VL-finetune), pinned private-project revision `efa37ba284d56192b246d9b4ed5d3668c1abd163` | No |
| Venus-Q Stage 1 model | `popo28/Venus-Q-Stage1`, pinned private-project revision `0f5c00c8d07ba889e9c5d12f828129dc322aae6a` | No |
| AesGuide assets | Referenced by the upstream research workflow | No |
| EmoSet dataset | Used as the separately obtained source dataset in the private project | No images or annotation records; selection IDs/configuration metadata only |

The code retained here consists of team-authored orchestration, validation, evaluation, contracts, tests, and integration work built to operate with separately obtained upstream dependencies. EmoSet selection identifiers and non-media configuration metadata remain only for provenance and deterministic-pipeline tests. This repository should not be read as authorship of the underlying models, upstream algorithms, or datasets.

Anyone attempting to reproduce the private research environment must obtain each dependency and asset from its authorised source and comply with the applicable upstream licence, research-use agreement, access conditions, and dataset terms. This repository does not supply those permissions.
