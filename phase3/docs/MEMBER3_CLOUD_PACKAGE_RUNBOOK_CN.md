# Member 3 云端检查包：执行顺序与证据清单

English version: [MEMBER3_CLOUD_PACKAGE_RUNBOOK_EN.md](MEMBER3_CLOUD_PACKAGE_RUNBOOK_EN.md).

> Historical preparation or implementation record. For current execution results and remaining actions, read the [Member 3 completion report](../MEMBER3_COMPLETION_REPORT_EN.md), updated 10 October 2026.

## 当前定位

这是本地准备的源码快照，不是已经完成的 pilot，也不是云端执行证明。包含 Jackie 分支基于 commit d2690f5d1535ae26e85788ddeb4f78e1940dc194 的原代码与后续未提交修正。bundle-manifest.json 的逐文件哈希定义本次包的身份；base_commit 不代表修改已提交。包不含模型权重、图片、API key 或付费授权。

只在新获得的云端运行授权、整段预算与实际独立停机保障落实后执行以下云端命令。包应解压到新的 Member 3 专属目录 `/workspace/phase3/member3/bundles/<bundle-id>/`，不要覆盖 Tian 的上游源码、Python 环境、模型或数据。

## 0. 本地传输前验证

用 prepare_member3_bundle.py --verify <archive> 校验包。收到的 archive SHA256 应与包外 verification.json 对应。manifest 中含每个文件的 SHA256。解压目录不能复用旧结果。

## 1. 共享环境只读检查

在包根目录运行，复用已有 Python，不执行 pip install 或 git pull：

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/member3_cloud_preflight.py \
  --full-model-hashes \
  --report /workspace/phase3/member3/evidence/<session-id>/preflight.json
```

`<session-id>` 必须换成唯一会话名称。该工具核对上传的 5 个 release、576 条对话中的 480 张图片哈希、模型 22 个文件、上游 commit、精确已审阅补丁、四项关键依赖。完整模型哈希需读取约 19.3GB，必须计入会话时间。此检查不加载神经网络、不做 forward。

若失败：保留 report，停止后续步骤。不要直接升级依赖、改模型或重新打补丁。

## 2. 检查实际 parser：新文件输出

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/inventory_qwen_interface.py \
  --output /workspace/phase3/member3/evidence/<session-id>/qwen-interface.json
```

预检精确源码与补丁后导入实际 HfArgumentParser dataclasses，记录 parser 名称、类型、默认值和文件哈希。默认 pending 接口配置保留，新报告单独输出。导入错误直接停止，不能手动将 pending 改成 verified。

## 3. 模型层 inventory：零 optimizer

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/inventory_lora_modules.py \
  --report /workspace/phase3/member3/evidence/<session-id>/lora-module-inventory.json
```

该工具在 CPU 加载本地模型以列出层、dtype 和候选 LoRA 参数量，默认不创建 optimizer，也不验证 GPU forward。结果必须检查 visual_matches 为空、匹配层非空。此前 rank-2 smoke 不替代本次 rank-8 配置。

## 4. 生成 pilot 子集与已验证配置

仅在 runtime inventory 通过、具体 pilot resource_limits 已由用户授权后：

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/make_lora_pilot_subset.py \
  --output-dir /workspace/phase3/member3/pilot-data
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/bind_lora_inventory.py \
  --config phase3/configs/lora-pilot-v1.json \
  --inventory /workspace/phase3/member3/evidence/<session-id>/lora-module-inventory.json \
  --resource-limits /workspace/phase3/member3/evidence/<session-id>/authorized-pilot-limits.json \
  --output /workspace/phase3/member3/evidence/<session-id>/pilot-verified.json
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/expand_lora_candidates.py \
  --base /workspace/phase3/member3/evidence/<session-id>/pilot-verified.json \
  --output-dir /workspace/phase3/member3/evidence/<session-id>/candidates
```

authorized-pilot-limits.json 只记录已明确授权的值，没有预填预算。预期 24 条 pilot train、完整 96 条 validation。保持原始 release 哈希不变。

## 5. 每个候选的 dry-run

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/run_lora_pilot.py \
  --config /workspace/phase3/member3/evidence/<session-id>/candidates/lr-1e-4.json \
  --interface /workspace/phase3/member3/evidence/<session-id>/qwen-interface.json \
  --inventory /workspace/phase3/member3/evidence/<session-id>/lora-module-inventory.json \
  --attempt-id <unique-dry-run-id> \
  --attempt-record /workspace/phase3/member3/evidence/<session-id>/<unique-dry-run-id>.json
```

此步骤不含 --execute，不会创建 optimizer。检查打印命令：pilot-train.json、rank 8、BF16、冻结 vision、单 GPU、输出目录隔离。真实执行必须用新的 attempt ID、--resource-ledger 固定共享账本、--attempt-minutes，以及已明确授权的整段运行限制。

## 6. 两任务 LoRA forward/backward：零 optimizer step

新增工具默认只输出 plan。云端实际执行需已获准检查会话、有效 inventory/interface/config 和新 report 路径：

```bash
/workspace/phase3/envs/venus-phase3/bin/python phase3/tools/check_lora_model_inputs.py \
  --config /workspace/phase3/member3/evidence/<session-id>/pilot-verified.json \
  --interface /workspace/phase3/member3/evidence/<session-id>/qwen-interface.json \
  --inventory /workspace/phase3/member3/evidence/<session-id>/lora-module-inventory.json \
  --report /workspace/phase3/member3/evidence/<session-id>/model-input-check.json \
  --execute
```

从冻结训练集的既定 pilot 子集中确定性选分类与联合指导各一条。使用完整神经网络、实际训练 preprocess、当前 rank 配置、gradient checkpointing、BF16 基础参数与 FP32 LoRA；不调整 dtype 来掩盖失败。要求目标层与 inventory 一致、参数量一致、基础与视觉参数冻结、因果移位后的监督非零、loss 有限、梯度有限且至少一个非零、BF16 wrapper 输入。零初始化的 LoRA B 导致 A 梯度为零是正常现象，不要求全部梯度非零。

两条样本分别 zero_grad 后 forward/backward。每次对所有 adapter 参数计算前后 digest，确保没有权重变化；不创建 optimizer、不保存 adapter。成功报告只证明两任务的小样本检查通过，不能声明 pilot、完整验证或训练完成。新进程的 adapter 保存/重载仍在 pilot 后验证。

本地已用小型 CPU LoRA 模型完成 autograd 防护测试；云端真实 Venus 尚未执行。运行仍由整段 Pod deadline 覆盖，该工具不自动停止 Pod。

## 7. Pilot 后生成与评分

generate_lora_validation.py 要明确传入本次 verified config、interface、inventory 和实际 adapter 路径，默认 plan，--execute 才生成。评分 evaluate_lora_pilot.py 读取 predictions.jsonl；缺失/重复/额外 ID 阻止结果通过。validation_token_weighted_loss 是监督 token 加权 loss；分任务 validation_loss 当前为每样本均值。三候选的预先选择规则还需明确采用哪一项，不能跑完再挑有利指标。

## 证据清单

| 阶段 | 必须保留 | 当前是否已有真实结果 |
|---|---|---|
| 包身份 | archive SHA256、逐文件 manifest、base commit | 本地包制作时生成 |
| 平台防护 | 明确预算、Pod ID、整段 deadline、后台 heartbeat、停机权限与实际状态 | 未完成 |
| 共享环境 | preflight.json、模型/数据哈希、依赖版本 | 本包命令待运行 |
| 训练接口 | qwen-interface.json、源码及 parser 身份 | 待运行 |
| 层 inventory | 模块名、数量、dtype、参数量 | 待运行 |
| 数据与配置 | 24/96 子集、manifest、候选 config 哈希、dry-run 命令 | 待云端生成 |
| 小样本模型检查 | 两任务 forward/backward、loss、梯度与冻结证明 | 工具已写，本地 autograd 测试通过；真实 Venus 未运行 |
| Pilot | stdout/stderr、Trainer 日志、attempt、账本、实际显存/耗时/费用 | 未完成 |
| 验证与 adapter | 96 预测、loss、评分、adapter 文件哈希、干净进程重载 | 未完成 |
| 复核 | Jackie/你的独立 pilot review，确切正式配置 | 未完成 |

本包不自动修改环境，不自动接收邀请或创建密钥，不启动、停止或删除云端资源。包内 watchdog 仅是工具；实际启用前另行确认并验证。

## 2026-10-05 真实检查补充

v3 首次真实云端预检、parser、module inventory 均通过。两任务 forward/backward 在首条样本的 loss 计算失败：检查器直接传递 preprocess 的 int32 labels，绕过了正式 Trainer 的 DataCollatorWithPadding。正式训练入口有 tokenizer，默认 collator 在本地固定 tokenizer 测试中会转为 int64，数值与 mask 不变。无须修改上游 finetune.py。

当前本地检查器与 validation 工具复用同一真实 collator；尚待下一次明确授权的 GPU 检查。不得把 v3 失败报告改成通过，或继续使用旧包。该补充后的源码需要重新生成包；以对应 archive/manifest 哈希识别。

本次 GPU 已停止，临时 API key 已禁用。完整结果和失败证据位于项目总文件夹 Member3_Local_Checks/CLOUD_CHECK_RESULT_2026-10-05_CN.md 与 cloud-evidence/20261005-h100-check/。
