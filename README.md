# 番茄采摘 Agent 初始代码稿

2026-11-10。面向双 FR3＋双 Wuji。当前实现等级：**离线模拟与只读影子监督**。

已实现阶段任务图、三值证据、有限恢复选择、事件记忆、预算、可撤销决策、模拟交接、JSONL 日志，以及按本机 `li-agent-mono` 编写的只读观测和 π0.5 预测适配。没有启动真机或修改已有模型服务。

## 当前能做什么

- 保留 S1～S7，支持 `S2→S7` 提前完成，以及 `S6→S3→S4→S5→S6` 重摆循环。
- 只用当前与过去证据；区分 TRUE/FALSE/UNKNOWN，检查时效、相机时间差、证据冲突和重复帧。
- 分别记录 commanded stage、observed activity、候选阶段、轮次、尝试、分离、投放和结果。
- 在无进展时建议脱离、持果投放、重摆或预算退出。左手未退出、持果未处理时不建议直接搬回。
- 去重事件；切换阶段不清空目标累计失败。相同恢复命令不会在等待反馈时反复发出。
- STOP/NEED_HELP 锁存；不会因为后续一帧恢复正常就自行继续。仅模拟器提供有审核信息的恢复接口。
- 用版本号隔离旧动作结果和 RTC 前缀；`ChunkFence` 是待接入执行进程的协议实现，不是已经接管 mono 的控制器。
- 本地 SDK 源码审计、只读 ROS 观测、没有硬件输出的 π0.5 推理封装。

## 尚未具备的能力

**没有训练好的视觉事件识别器。** `EvidenceMonitor` 对外部判断做时序稳定和冲突检查，不会凭 RGB 自动识别果实分离。回放输入使用人工/算法提供的证据；默认 shadow 不提供物理证据，所以返回 UNKNOWN、OBSERVE，预算到达后 NEED_HELP，符合当前能力状态。

**没有真机动作入口。** `--simulate` 只更新内存中的模拟阶段。`STOP` 是监督建议，不是已经停止硬件；只读 shadow 无法停止另一个正在运行的部署进程。真正接管前还需将交接协议接到 mono 的单一控制所有者。

**现有 Full 服务不满足阶段指令合约。** 预测适配要求新增 `stage_prompt_mode=per_request`、`effective_prompt_echo=true` 元数据及每次响应的 `effective_prompt`。这些是拟议接口，当前服务缺少时会明确拒绝，不假装换 prompt 已生效。

初稿阈值供离线测试，不是实机标定结果。没有在线权重更新、任意目标选择、掉落拾取或复杂缠连恢复。未增加 LLM/VLM 服务依赖，先用确定性恢复菜单验证任务规则。

## 目录

```text
src/tomato_agent/
  models.py                 数据类型、谓词、阶段技能和候选阈值
  evidence.py               因果证据稳定、冲突和重复观测处理
  supervisor.py             阶段管理、恢复、预算、模拟交接、版本围栏
  adapters/lia_mono.py       本地 SDK 只读读取、预测封装、源码审计
  replay.py                 JSONL 回放与决策日志
  cli.py                    replay / inspect-sdk / shadow
configs/draft.json          待标定配置
examples/                  人造证据示例，不是真实示教
scripts/make_demo.py        可重复生成示例
tests/                     控制边界、预算、证据与接口测试
docs/                      本地 SDK 核对和接入说明
outputs/                   本地实验产物，不进入 Git
```

## 立即运行离线示例

核心仅依赖 Python 3.11+ 标准库，无需安装 ROS、GPU 或模型。

```bash
cd /home/user/ty/new/agent
PYTHONPATH=src python3 -m tomato_agent replay examples/recovery.jsonl \
  --simulate --output outputs/my-recovery.jsonl
```

示例会模拟：拿起 → 采摘 → 脱离左手 → 放回 → 调整 → 再拿起 → 采摘投放 → 裸杆收尾。

提前摘完示例：

```bash
cd /home/user/ty/new/agent
PYTHONPATH=src python3 -m tomato_agent replay examples/early_finish.jsonl \
  --simulate --output outputs/my-early-finish.jsonl
```

输出文件不覆盖；再次运行换一个文件名。省略 `--simulate` 时不会接受模拟交接，也不会自动推进 commanded stage。

可选使用配置：全局 `--config` 写在子命令之前。

```bash
cd /home/user/ty/new/agent
PYTHONPATH=src python3 -m tomato_agent --config configs/draft.json \
  replay examples/recovery.jsonl --simulate --output outputs/custom-config.jsonl
```

## 本地 SDK 检查

仅读取源码和配置，不连接设备：

```bash
cd /home/user/ty/new/agent
PYTHONPATH=src python3 -m tomato_agent inspect-sdk
```

默认目录为 `/home/user/lpy/li-agent-mono` 和 `/home/user/lpy/pi05-server`，可以用 `--mono-root` / `--server-root` 指定其他目录。输出包含路径与哈希、初始审计发现；它不是 ROS 在线话题探测器，不能证明硬件在线。

## 只读影子运行

需要已有 ROS 硬件/相机发布进程和 mono 环境。此命令不会替你启动这些进程，也不启动控制或 Home。

```bash
cd /home/user/lpy/li-agent-mono
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_DISCOVERY_SERVER
PYTHONPATH="/home/user/ty/new/agent/src${PYTHONPATH:+:$PYTHONPATH}" \
  /home/user/.pixi/bin/pixi run --locked python -m tomato_agent shadow \
  --seconds 30 --sample-hz 5 --startup-timeout 10 \
  --output /home/user/ty/new/agent/outputs/my-shadow.jsonl
```

ROS 配置应与已有发布进程一致；以上沿用此前部署配置。没有话题时会超时并记录原因，不发送运动命令。日志是观测时间、关节值和决策元数据，**不保存 RGB 视频**；真实回放应关联既有 workbench 视频记录及其时间戳。

采样频率 5 Hz 是影子读取频率，不是机械臂控制频率，也不是模型推理频率。

## 接入你自己的证据

JSONL 每行必须是 `tomato-agent-replay-v1`，包含：

- `snapshot`：episode/observation/clock 身份、当前源时钟、状态与相机采样时间、54 维实测关节。
- `evidence`：谓词名、TRUE/FALSE/UNKNOWN、对应 observation、源时间、来源及证据引用。
- `events`：尝试失败/未决、确认分离、确认投放的唯一事件；分离与投放需要同一目标 ID。
- `mock_handoff`：是否在该帧显式模拟一次执行交接；只有 `--simulate` 才生效。

完整格式见 `examples/*.jsonl`。所有示例均标记 `synthetic=true`、`synthetic://` 来源，不能作为模型训练数据或实机成功证据。

`source` 和 `references` 应指向真实视频帧/日志，不是模型的内部思维链。源时间和 `snapshot.now` 必须在同一时钟域；不可直接混用 `time.monotonic()` 和 ROS 时间。

接视觉模型时，让它提供 `Evidence`/`Event`；维持 UNKNOWN，不把自报置信度直接当作已校准成功概率。动作 SDK 的回执不能充当“已分离”“已入篮”的视觉真值。

当前失败计数对已标注目标可累计，目标不可靠时使用整串无进展计数；没有实现视觉 ID 关联。`target_id` 是提供方需要负责维护的身份，不能通过每次创建新 ID 绕过预算。

## 如何继续接入执行端

1. 用真实审核片段验证 `Evidence`/`Event`，补齐接触/退出及剩余果实证据。
2. 为阶段条件化模型建立独立服务合约，记录 tokenizer 实际使用的 prompt，关闭该模型不适用的全任务覆盖。
3. 在 mono 控制进程接入 transition 消息、`stage_epoch`、退出技能、停止/保持确认和新鲜状态重新推理。
4. 将 `ChunkFence` 同样部署在动作接纳端。只在高层拒绝旧结果是不够的。
5. 给每项实际通过验证的技能登记适用范围，再增加 L1 人工确认执行入口。当前 `available_skills` 只限制建议候选，不能打开硬件执行。

`Pi05PredictionOnly` 只封装推理，保留现有适配器的关节排列、动作语义和归一化校验；其返回的 `chunk` 不会自动下发。首版语义切换不携带旧阶段 RTC 前缀。

当前进程重启不自动恢复旧 Episode，更不会从旧动作队列继续。后续做恢复时应重新读取状态、人工确认，再关联历史预算。

## 验证

```bash
cd /home/user/ty/new/agent
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

还可在 mono 的 Python 中运行相同测试，额外执行真实 SDK 的无连接构造检查：

```bash
cd /home/user/ty/new/agent
PYTHONPATH=src /home/user/lpy/li-agent-mono/.pixi/envs/default/bin/python \
  -m unittest discover -s tests -v
```

测试覆盖提前完成、恢复路径、未知/冲突/陈旧证据、持果优先、左手未退出、预算退出、重复事件、迟到结果、过期/被替换提案、停止锁存以及 prompt 回执。
