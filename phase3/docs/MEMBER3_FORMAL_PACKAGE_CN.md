# Member 3 正式训练准备包

English version: [MEMBER3_FORMAL_PACKAGE_EN.md](MEMBER3_FORMAL_PACKAGE_EN.md).

> Historical preparation or implementation record. For current execution results and remaining actions, read the [Member 3 completion report](../MEMBER3_COMPLETION_REPORT_EN.md), updated 10 October 2026.

本包是本地准备材料，不是正式训练完成证据，也不授权启动 Pod。

## 包内有什么

- 当前 Phase 3 代码、测试、冻结数据发布文件与锁定文件。
- `formal-preparation/formal-config-DRAFT.json`：480 条训练对话、3 epochs、180 optimizer steps；每 60 steps 保存 checkpoint；暂定 lr=2e-4。
- `formal-preparation/reviewed-evidence/`：三个 pilot 的比较核验、16 条 validation 指导诊断、80 条训练指导审阅。保留原始内容与 SHA256。
- `formal-preparation/delegated-decision-PORTABLE-PENDING.json`：证据路径改成包内相对路径；仍为 pending，不是 Jackie 独立审阅，也不是用户批准正式配置。
- `bundle-manifest.json`：逐文件大小与 SHA256。整包校验记录另存于压缩包旁。

本包没有模型权重、图片、Python 环境、API key 或 SSH 私钥。共享卷里的模型、图片和已有环境需另行只读核验。打包程序仅选取既有 source bundler 所允许的 Phase 3 文件类型；不要把凭据放进这些代码或 JSON 文件。

## 换机器后如何使用

先把压缩包放到独立的新目录，不覆盖 Tian 的 checkout、环境或冻结数据。解压前，应检查压缩包 SHA256 与本地 `.verification.json` 一致。校验程序的 `--verify` 只读取归档，不解压：

```bash
python /已知可信代码目录/phase3/tools/prepare_member3_bundle.py --verify /实际路径/formal-preparation.tar.gz
```

在包解压到独立目录后，把审批草案里的证据路径转换成该机器的绝对路径。下面两个路径必须替换；转换不改变 pending 状态，不执行训练：

```bash
python /解压目录/phase3/tools/prepare_formal_bundle.py materialize \
  --bundle-root /解压目录 \
  --output /新的输出目录/delegated-decision-PENDING.json
```

不要把之前在 Mac 生成的 checkpoint 命令直接粘到云端：命令中的工具路径需在部署后重新生成。`plan_formal_validation.py --mode plan` 生成计划；`--mode collect` 只核验已完成的真实结果。它没有自动执行训练或停机的功能。

## 实际执行前还差什么

1. 连接可用机器，核验共享卷、冻结数据、图片、模型和既有 Python 环境；重新采集 module inventory，并绑定最终配置。
2. 部署并核验整个会话执行器 `run_formal_session.py`：训练及三个 checkpoint 的 validation 共用 deadline，子进程超时后终止，任一环节失败即停；不要将 optimizer ledger 当作完整 Pod 费用控制。代码与本地测试已完成，云端接通尚未验证。
3. 在本地启用独立 Pod stop watchdog。训练进程退出不等于 Pod 停止；停机后需查询确认。凭据保留本地，不打包进云端。
4. 用最终配置、实际 GPU 费率、预算、时限生成新记录，取得对应的明确确认。更改配置后须重新绑定 config SHA256；不能把冻结发布 summary 的历史授权字段改成 true。
5. 正式训练完成后，分别在干净进程重载 checkpoint 60/120/180，运行同一 96 条 validation，再用固定规则比较。审阅指导内容后才能接受候选 adapter。

正式 Eval80、B1/C 比较和匿名随机五维人工指导评分仍是后续工作，不能由 validation loss 或格式通过率替代。

## 执行器与停机保护怎样衔接

以下是准备好实际连接信息后使用的操作顺序。本地测试通过不代表云端已运行，当前草案不可直接执行。所有 `/实际路径/`、Pod ID、端口和 deadline 均须据真实环境填写。

### 1. 确定同一轮 session

会话记录至少包含以下字段；不能把旧 pilot 的 session 或批准记录拿来复用：

```json
{
  "pod_id": "实际PodID",
  "attempt_id": "formal1",
  "started_at_utc": "实际本轮开始时间，含时区",
  "deadline": "本轮绝对截止时间，含时区",
  "max_session_usd": null,
  "actual_hourly_usd": null,
  "training_minutes": null,
  "resource_ledger": "/workspace/phase3/member3/formal/resource-ledger.json",
  "config_sha256": "最终配置的SHA256",
  "session_sha256": "本记录除session_sha256字段外的canonical SHA256"
}
```

`started_at_utc` 从本轮 Pod 实际启动开始计算，包含部署和检查的时间，不能从训练开始重新计时。执行器核对时限与计算费用估计是否落在最终配置的资源范围内，但无法自行核实平台账单或实际计费起点。需从控制台确认 GPU、实际费率与开始时间；存储费用和平台延迟不能视为自动封顶。

session 和最终配置 SHA256 的计算必须使用 `member3_common.canonical_sha256`，不要用普通文件字节哈希代替。配置、inventory 或资源范围变化后，更新其绑定并由用户确认。

### 2. Mac 先准备独立 watchdog

`prepare_watchdog_launchagent.py` 仅生成 LaunchAgent 文件，另外加载后才会运行。增加 `--stop-request /本轮目录/stop-now`，与下一步 relay 使用同一路径。API key 文件仅存在 Mac，权限为 0600。使用实际 Pod ID、同一个 deadline 和全新报告/触发文件路径。

必须确认 watchdog 报告为 `armed`、`observed_state=RUNNING` 且 heartbeat 新鲜，不能将 `waiting_for_start` 当作已经运行。

### 3. Mac 启动 relay

目的：每 5 秒把不含 key 的 watchdog heartbeat 通过 SSH 送给云端；读取云端任务状态。一旦任务结束、失败或连接异常，就创建本地 `stop-now` 文件，让独立 watchdog 请求停机。单次 SSH 有超时，watchdog 的绝对 deadline 不依赖 SSH 连接。

```bash
caffeinate -is python /本地代码目录/phase3/tools/relay_formal_watchdog.py \
  --session /本轮目录/session.json \
  --watchdog-report /本轮目录/watchdog.json \
  --identity /已有SSH私钥路径 \
  --target root@实际SSH主机 --port 实际端口 \
  --remote-guard /workspace/phase3/member3/formal/本轮独立目录/guard.json \
  --remote-status /workspace/phase3/member3/formal/本轮独立目录/evidence/session-status.json \
  --stop-request /本轮目录/stop-now \
  --relay-report /本轮目录/relay.json --execute
```

云端 `本轮独立目录` 需先存在，`evidence` 目录须尚不存在，由执行器创建。首次 SSH 主机身份核验需正常完成；relay 使用 BatchMode，不会替你输入密码或接受未知主机。Mac 必须保持联网且不休眠。网络断开时，云端 heartbeat 超过 30 秒即停止任务；本地 watchdog 仍尝试 API 停机。若 Mac 断电或 API 不通，自动停机不能保证，必须手动确认平台状态。

### 4. 云端运行执行器

先不带 `--execute` 检查命令计划，确认使用了独立目录、已批准的最终配置以及新采集并绑定的 inventory（执行时要求 24 小时内）。正式启动时，增加 `--execute`：

```bash
/workspace/phase3/envs/venus-phase3/bin/python /部署代码目录/phase3/tools/run_formal_session.py \
  --config /最终配置路径/formal-config.json \
  --session /本轮目录/session.json \
  --inventory /本轮目录/lora-module-inventory.json \
  --interface /本轮目录/qwen-interface.json \
  --review /本轮目录/delegated-decision-APPROVED.json \
  --guard /workspace/phase3/member3/formal/本轮独立目录/guard.json \
  --data-dir /workspace/phase3/data/bridge-train-v1 \
  --output /workspace/phase3/member3/formal/本轮独立目录/evidence
```

执行顺序固定：全模型文件哈希 preflight → formal 配置审批 gate → 两类输入 forward/backward 检查 → 全量正式训练 → checkpoint 60/120/180 各自干净进程 validation → 严格证据比较。不会重试训练或增加第四个 checkpoint 候选。`training_minutes` 是训练阶段单独上限，同时整条链不能超过 session deadline。任何一步失败，不继续下一步。

`session-status.json` 是执行过程状态，包含命令退出码和输入文件绑定。它的 `stop_requested=true` 只表示请求停机；需查看 Mac watchdog 的 `stop_state_observed` 并确认控制台 Not running，才能报告已停止 Pod。没有停机确认时，不得写成已停止。任务输出保存在共享卷，本轮停机后再下载核验、备份和整理交接。

本地测试使用模拟命令验证控制逻辑，并用真实 CPU 子进程验证超时及嵌套进程终止；没有调用真实 RunPod API，没有训练模型，也没有产生正式 validation 分数。
