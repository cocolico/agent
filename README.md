# 番茄采摘在线 Agent

2026-11-10。面向双 FR3＋双 Wuji。目标：**本地视觉大模型在线监督 π0.5，并实际切换后续执行技能**。在线代码与模拟测试已完成；尚未接入真实视觉模型或进行实机验证。

仅修改本目录；外部 `li-agent-mono` 和 `pi05-server` 作为只读依赖。没有下载轻量化模型，没有启动真机。离线回放和影子入口继续作为调试工具，在线运行不要求先执行它们。

## 在线主入口

完整接入协议与启动步骤见 [在线闭环接入说明](docs/在线闭环接入说明.md)。

```text
相机/关节 → 本地视觉模型 HTTP 接口 → 结构化状态与技能建议
                                   ↓
                         任务图、证据和预算检查
                                   ↓
                    确认保持 → 重观测 → 准备新技能
                                   ↓
                  π0.5 阶段服务 → SDK 控制器 → 实机
```

新增代码：

| 模块 | 功能 |
|---|---|
| `online/vlm.py` | 多视角时序请求、兼容本地视觉服务的 HTTP 客户端、结构化回复校验、单请求异步推理 |
| `online/coordinator.py` | 视觉建议与任务规则融合、保持后重观测、执行准备回执和阶段提交 |
| `online/execution.py` | 单一控制线程、真实 `body.stop()` 保持确认、新建每技能 π0.5 控制器、动作下发 |
| `online/runner.py` | 连续观测、异步监督、超时处理、运行日志及在线入口 |
| `stage_server.py` | 在本目录启动独立 π0.5 阶段服务，关闭固定指令覆盖，返回有效 prompt 与 epoch |

现在可以进行无连接配置检查：

```bash
cd /home/user/ty/new/agent
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B -m tomato_agent.online.runner \
  --config configs/online.example.json
```

后续接入时填写视觉服务地址、模型名、π0.5 合约及已经验证的技能。真实运行使用独立的 `--execute` 入口，不能与另一个已持有设备控制权的推理执行进程同时运行。

## 当前能做什么

- 保留 S1～S7，支持 `S2→S7` 提前完成，以及 `S6→S3→S4→S5→S6` 重摆循环。
- 只用当前与过去证据；区分 TRUE/FALSE/UNKNOWN，检查时效、相机时间差、证据冲突和重复帧。
- 分别记录 commanded stage、observed activity、候选阶段、轮次、尝试、分离、投放和结果。
- 在无进展时建议脱离、持果投放、重摆或预算退出。左手未退出、持果未处理时不建议直接搬回。
- 去重事件；切换阶段不清空目标累计失败。相同恢复命令不会在等待反馈时反复发出。
- STOP/NEED_HELP 锁存；不会因为后续一帧恢复正常就自行继续。仅模拟器提供有审核信息的恢复接口。
- 用版本号隔离旧动作结果；在线切换关闭旧推理连接与控制器，再创建空时间轴的新控制器，RTC 前缀不跨技能继承。
- 本地 SDK 源码审计、只读 ROS 观测、没有硬件输出的 π0.5 推理封装。

## 尚未具备的能力

**视觉模型尚未部署。** `online/vlm.py` 已提供接口，连接后可以直接使用视觉千问判断状态、异常和下一技能；没有要求先训练一个独立事件分类器。通用视觉模型在本任务上的可靠性仍须评测。`EvidenceMonitor` 负责稳定和检查这些判断。旧 shadow 入口不调用视觉模型，因此仍保持 UNKNOWN。

**真机路径已编写但未实机验证。** 在线 `--execute` 获取 SDK 控制权后可下发动作、确认停止并切换技能；旧 `--simulate` 和 shadow 仍然不输出动作。在线程序从新任务接管，不能隔空修改另一个运行中的控制进程。

**原 Full 服务仍保持原行为。** 本目录的 `stage_server.py` 复用它的加载器，以 `override_prompt=False` 创建策略，实现阶段指令与回执。必须使用这个新入口；普通代理转发到原固定指令服务无法解决覆盖问题。新入口继续严格检查 checkpoint 物理合约，尚未适配的未来导出格式会明确报错。

初稿阈值是待标定参数。没有在线权重更新、任意目标选择、掉落拾取或复杂缠连恢复。视觉模型提出事实和技能建议，任务图负责校验许可条件；不会执行模型输出的任意代码或关节数组。

## 目录

```text
src/tomato_agent/
  models.py                 数据类型、谓词、阶段技能和候选阈值
  evidence.py               因果证据稳定、冲突和重复观测处理
  supervisor.py             阶段管理、恢复、预算、模拟交接、版本围栏
  adapters/lia_mono.py       本地 SDK 只读读取、预测封装、源码审计
  replay.py                 JSONL 回放与决策日志
  cli.py                    replay / inspect-sdk / shadow
  stage_server.py           独立 π0.5 阶段服务，不改原服务
  online/                   本地视觉模型接口、在线协调器和 SDK 执行管理
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

## 后续接入与验证

1. 用真实审核片段验证 `Evidence`/`Event`，补齐接触/退出及剩余果实证据。
2. 使用本目录阶段服务加载相应模型，核对指令跟随能力和实际 tokenizer 输入。
3. 通过本目录在线执行入口，验证 SDK 保持回执、旧推理退出和新技能受控起步。
4. 对每条恢复链验证起始状态和执行结果，再加入 `validated_skills`。
5. 依次测量本地视觉推理时延、π0.5 并发资源争用、误判断和恢复收益。

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
