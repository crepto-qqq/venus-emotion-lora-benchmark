# Member 3 本地运行防护修正（2026-10-05）

English version: [MEMBER3_LOCAL_FIXES_2026-10-05_EN.md](MEMBER3_LOCAL_FIXES_2026-10-05_EN.md).

> Historical preparation or implementation record. For current execution results and remaining actions, read the [Member 3 completion report](../MEMBER3_COMPLETION_REPORT_EN.md), updated 10 October 2026.

本次只改本地代码，未启动 RunPod、未训练、未修改冻结 release。

## 已修正

- Pilot 默认读取 `/workspace/phase3/member3/pilot-data/pilot-train.json` 与 `pilot-validation.json`。启动前逐条比较冻结 release 按既定 seed 生成的子集，拒绝错文件、完整训练集或篡改内容。当前预期 24 train / 96 validation。
- 运行记录的 config_sha256 保留原配置身份；resolved_config_sha256 单独记录输出路径展开后的配置。拒绝不一致 candidate ID、复用 attempt record 和已有输出目录。
- 训练进程组设置超时，使用 wall-clock 与预算/最高时薪折算时限中较小者；到时 TERM，5 秒后仍未退出则 KILL。这不是实际费用监控，更不停止 Pod。多次尝试累计预算和独立 Pod 停机仍待实施。
- 证据必须有成功运行状态、匹配配置、自校验哈希、明确保存与重载成功、完整 evaluation、合法 adapter 哈希和有限正数资源记录；超出配置上限则失败。
- 评分拒绝重复、缺失、额外预测 ID。格式通过不代表指导质量通过。

## 数据路径与后续步骤

上传的原始 release 在 `/workspace/phase3/data/bridge-train-v1/`；本地 repo 中的冻结 release 仍保持原位。Pilot 子集应由 `make_lora_pilot_subset.py --output-dir /workspace/phase3/member3/pilot-data` 生成（该命令留待已授权云端会话，不是本次执行记录）。随后 expand_lora_candidates.py 才生成候选配置。图片引用保持冻结 release 原值，运行前仍须确认共享卷实际图片可读。

Formal 配置不能复制 pilot 数据路径；应指向已上传的完整 train/validation JSON，启动器会逐条比对完整 release。

## 未完成与限制

- 真实 parser/module inventory 仍 pending，不能伪造 verified 配置或通过完整 launch gate。
- 实际模型验证预测生成、分任务 loss、权重重载证据仍须实现/与真实训练接口核对。当前 evaluate_lora_pilot.py 只是预测评分工具。
- 本地证据检查核对记录一致性，不等于独立验证权重存在或证明人工填写的 runtime facts 为真；还需实际日志与权重校验。
- 尚未完成累计 attempt 限制、独立 Pod 停机与新的付费授权；不得据此直接运行付费训练。

## 验证

Member 3 原有 12 项与新增 4 项测试通过。新增覆盖真实进程超时、24/96 子集及错误完整训练集拒绝、非法预算值、保存失败/不完整验证/错误配置证据拒绝。
整个 Phase 3 共 26 项测试：22 项通过，4 项因所用 Python 缺少 jsonschema 而出错，不能声明全套通过。git diff --check 通过。

## 后续：adapter 验证预测生成工具

新增 generate_lora_validation.py，默认仅输出 plan，--execute 才加载模型。只接受与冻结 BridgeTrain validation 完全相同的 96 条对话，拒绝其他集合或子集。生成接口只读取 user turn，assistant reference 仅用于训练预处理与 teacher-forced loss。

执行时要求 pilot gate、精确上游 commit、parser inventory 的 finetune_py_sha256、可读图片及完整 adapter 文件；使用 BF16 单 CUDA GPU、不允许 offload；加载 adapter 后逐条记录预测、loss 与有效监督 token 数，评分输出另外提供 token-weighted validation loss。输出目录禁止复用，失败保留 failed run.json。

本地 dry-run plan 位于项目总目录 Member3_Local_Checks/validation-generation-plan.json。示例 adapter 路径仅用于说明，不代表存在真实权重。新增 3 项输入/身份/答案隔离测试，Phase 3 全部 29 项通过（独立 test-env 已安装 jsonschema==4.19.0）。此前 4 项缺依赖已解除。

该工具尚未经真实 GPU adapter 验证。model.chat 与 PEFT 的实际兼容性、完整预处理导入、模型权重身份和正式 loss 选择规则仍需云端核查；当前工具不负责预算累计或 Pod 停机，不能据此直接开始付费运行。训练中间 checkpoint 的外部验证调度尚待接入。

## 后续：累计训练限制与停机准备

新增 member3_budget.py 并接入启动器；同阶段固定共享账本，执行前保守预留尝试次数、分钟与费用，失败不自动退还。新增 runpod_stop_watchdog.py 与 prepare_watchdog_launchagent.py，实现指定 Pod 的截止停止、状态读取和 macOS 后台配置生成。全部 36 项测试通过，未真实启用 watchdog 或创建云端资源。

此前“累计尝试控制待实施”已被上述本地实现取代，但平台整段费用、真实后台安装/权限测试、共享卷文件锁与启动前 watchdog 状态验证仍待完成。详细范围与限制见 MEMBER3_RESOURCE_GUARDS.md。

## 后续：云端运行包

新增只读 preflight、真实 inventory 绑定工具及可校验源码打包工具。parser inventory 改为必须指定新的输出，不覆盖 pending 模板；增加精确 commit+补丁重建校验。LoRA module inventory 强制 local_files_only 并拒绝覆盖报告。launcher 与 validation generator 支持显式 --interface，以复用独立的新接口报告。

云端完整步骤与证据清单见 MEMBER3_CLOUD_PACKAGE_RUNBOOK_CN.md。完整模型 forward/backward、真实 adapter 验证、实际 watchdog 与新的预算授权仍未完成；所有命令均未在云端执行。

## 后续：两任务完整模型输入检查工具

新增 check_lora_model_inputs.py，默认 offline plan；--execute 才导入 CUDA 模型并对分类/联合指导各一条真实训练样本 forward/backward。零 optimizer step、零权重保存；检查冻结、dtype、监督、目标层/参数量、有限 loss/梯度、BF16 wrapper 输入和 adapter 参数前后 digest。错误时保存 failed report 与已完成任务，不能写 passed。

已有独立 test-env 全套运行 45 项，其中 39 项通过、6 项真实 autograd 测试因没有 torch 跳过。另用已存在的 Python3.10/torch2.0.1 CPU 环境运行本文件 7 项（含这 6 项），全部通过。因此所有 45 个不同测试均在适合的本地环境实际执行通过，不表示在单一环境跑完。日志分别在 Member3_Local_Checks/phase3-tests.log 与 model-inputs-cpu-tests.log。

新增包 cloud-bundle-2026-10-05-v2.tar.gz 包含本次工具、测试及更新 runbook。旧包保留，使用新包；新 SHA256 与逐文件 manifest 已校验。真实 Venus GPU 检查、训练和保存/重载尚未执行。
