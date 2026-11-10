# 本地 SDK 核对与接入边界

日期：2026-11-10。仅源码、配置和无连接构造核对；没有连接硬件，没有测试真实视频识别、模型预测或机器人动作。

## 已确认的观测能力

| 信号 | 本次证据 | 初稿处理 |
|---|---|---|
| 54 维关节位置 | `ObservationReader._receive()` 经 codec 解码 `JointState.position`；`get_observation()` 拼成 `joint_pos` | 使用原 SDK 顺序，不手工猜维度 |
| 三路 RGB | cam0、cam1、cam2，相机配置维度 640×400、640×480、640×480 | 订阅原 SDK 只读 reader，保留相机话题 |
| 源时间 | `state_time`、`image_times`、`extra.clock_id`、reader 的 clock | 检查时效和时钟变化；不混用单调时钟 |
| 深度 | 机器人 YAML 有 `head_depth`；Pi05 相机 profile 三路均 `enable_depth: false`，streaming 清空 sensors | 不订阅、不假定可用 |
| 外力矩/末端力 | SDK 有六维力传感器适配，但当前 Pi05 路径未配置；reader 未把 JointState effort 提供为观测字段 | 未确认，不作为决策输入 |
| 手部电流/触觉 | SDK 存在通用传感器扩展；当前 reader 的 Wuji 通路解码位置 | 未确认；不把底层可能支持等同于已接入 |
| 关节速度 | 当前只读 reader 未提供独立速度字段 | 不声明有实测速度；如后续差分估计须明确标注 |

只读 reader 的 joint state 时间是各组最早时间，不提供每组完整时间数组；初稿不能声称已经检查了不可见的每组采样差异。

## 已采用的本机接口

1. `lia_robots.robots.franka_wuji.config()` 读取本体配置。
2. `camera_profiles.load_camera_profile()` 读取已有 Pi05 视角尺寸。
3. `franka_wuji(...)` 构造未连接的本体定义。
4. `definition.observation_reader(...)` 创建独立只读订阅器。
5. `reader.open/get_observation/close` 管理只读 ROS 生命周期。

没有调用 `definition.reset/step/command/stop/route`。只读观察不持有控制权，也不刷新控制保持超时。

初次真实 SDK 构造测试发现：直接用 camera_dimensions 字典替换相机配置会丢失 topic。初稿适配器只替换 width/height，保留原相机配置和话题，随后无连接构造通过。

## 推理接口与限制

- 使用 `Pi05WebsocketClient`、`Pi05Policy`、`FrankaWujiAdapter` 和 checkpoint 自带合约。
- Full profile 的 `override_prompt=True` 与 `OverridePrompt` 是当前阶段调用的阻塞项。
- 新增的预测封装要求服务声明 per-request 指令，并逐次核对 `effective_prompt`。服务升级和真实模型语言可控性验证尚未执行。
- `Pi05Policy.reset()` 调用的 websocket `reset()` 是空操作，不是停止设备和清除动作队列的接口。
- 当前所有预测接口都没有动作下发；`ChunkFence` 仅在初稿中完成离线协议测试，还没有接入 mono 执行进程。

## 代码位置

- `/home/user/lpy/li-agent-mono/lia-robots/src/lia_robots/robots/observation.py`
- `/home/user/lpy/li-agent-mono/lia-robots/src/lia_robots/robots/ros_embodiment.py`
- `/home/user/lpy/li-agent-mono/lia-robots/src/lia_robots/robots/devices.py`
- `/home/user/lpy/li-agent-mono/lia-robots/src/lia_robots/robots/sensors.py`
- `/home/user/lpy/li-agent-mono/data/robots/franka_wuji.real.yaml`
- `/home/user/lpy/li-agent-mono/data/deployments/pi05/franka_wuji/cameras.yaml`
- `/home/user/lpy/li-agent-mono/lia-policies/lia_policies/pi05/client.py`
- `/home/user/lpy/pi05-server/src/pi05_server/profiles.py`
- `/home/user/lpy/pi05-server/src/pi05_server/runtime.py`

初始快照哈希见同目录 `sdk-audit-2026-11-10.json`。文件发生变化时应重新核对，不把旧结论当成动态能力探测。
