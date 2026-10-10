# Member 3 累计运行限制与独立停机准备

English version: [MEMBER3_RESOURCE_GUARDS_EN.md](MEMBER3_RESOURCE_GUARDS_EN.md).

> Historical preparation or implementation record. For current execution results and remaining actions, read the [Member 3 completion report](../MEMBER3_COMPLETION_REPORT_EN.md), updated 10 October 2026.

状态：本地实现与模拟验证完成，未启用后台任务、未联系 RunPod API、未启动云端资源。没有真实预算授权、API key 或 Pod ID 写入配置。

## 1. 累计训练尝试

run_lora_training.py 的 --execute 现在额外要求 --resource-ledger 与 --attempt-minutes。同一阶段使用固定位置：

- Pilot：/workspace/phase3/member3/pilot/resource-ledger.json
- Formal：/workspace/phase3/member3/formal/resource-ledger.json

每次运行前在文件锁内检查最大次数、累计预留分钟、按最高允许时薪估计的累计预留美元。不同候选配置不能换账本；已有账本绑定阶段与 resource_limits，修改限制会拒绝执行。重复 attempt ID、篡改的账本哈希、非法数值都拒绝。失败或中断不自动退还预留值，避免未经核算重复尝试。文件锁在共享卷上的支持仍需实测。

这本账本记录 optimizer 进程的保守预留，不是平台账单：模型加载前的环境检查、独立验证生成、停机延迟、存储等没有自动计入。整个 Pod 的限时限费仍需以下独立控制，并核算 Phase 3 已发生的费用。

## 2. 独立停止指定 Pod

runpod_stop_watchdog.py 默认只输出计划。--execute 才读取当前用户持有的权限为 0600 的普通 RunPod API key 文件；S3 key 不适用。key 不写入参数、报告或日志。

工具只使用 GET /v1/pods 与 POST /v1/pods/<id>/stop，没有创建、启动、编辑、删除 Pod 的功能。预检目标 Pod 状态，到带时区的绝对 deadline 后请求停止，再读取 API 验证 EXITED 状态；缺失目标或失败请求不能标记成功。进程重启时，若 deadline 已过则立即尝试停止，不会重新等待一段完整时长。

API 返回 desiredStatus 是期望状态，report 中的 stop_state_observed 仅表示 API 已返回 EXITED；最终仍需控制台检查实际停止及费率。不是对实际停止时间、到账费用或网络可靠性的保证。

prepare_watchdog_launchagent.py 只生成 macOS plist 文件，不执行 launchctl、不安装任务。配置使用 RunAtLoad、失败自动重启、独立日志路径，不包含 key 内容。通过 /usr/bin/caffeinate -is 包裹 watchdog，在任务运行期间抑制空闲休眠；任务结束后释放，不修改系统电源设置。合盖、关机、断网仍不受此保障。生成后的 ProgramArguments 含 --execute：一旦将来加载后台任务，会真实联系 API 并在截止时间停止指定 Pod，因此本次没有加载。

## 3. 付费会话前待办

1. 明确新阶段资源限制，并确认此前累计费用；不能沿用已经结束的 $2/30 分钟检查授权。
2. 取得普通 API key 的明确授权及最小可用权限；本次没有创建或读取 key。
3. 确定准确 Pod ID、实际总时薪，以及从 Pod 开始计费算起的绝对截止时间。预留模型加载、验证、停机/API 延迟余量。
4. 在聊天之外安装并验证后台任务。Mac 必须持续开机、联网、保持登录且不休眠；否则应改用可靠的常在线执行机。登录任务不能在 Mac 关机时停云端 Pod。
5. 真正验证 Stop 权限、后台进程和 heartbeat；本地 mocks 不能替代这一步。启动训练前还需人工核对这些保障已经生效，启动器目前不会自动验证 watchdog receipt。
6. 会话结束立即停止，不必等最终 deadline；控制台确认实际停止和计算费率，保留账单/状态证据。共享卷仍保留并按存储费率计费。

## 4. 本地测试证据

Phase 3 36 项测试通过。新增覆盖累计次数、跨候选预算、限制变更、重复 attempt、账本篡改、Stop 后状态验证、网络失败、目标缺失、时区及 LaunchAgent 参数。所有 API 测试均使用模拟 transport，未发起真实请求。

官方依据：
- https://docs.runpod.io/pods/manage-pods
- https://docs.runpod.io/api-reference/pods/GET/pods
