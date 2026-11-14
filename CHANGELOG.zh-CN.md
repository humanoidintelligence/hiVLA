# 更新日志

每个 hiVLA 版本用户看得到的变化，新的在上面。格式参照
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。0.1.0 之前任何版本都可能有不兼容的接口或线协议改动，
这类改动列在"移除"或"变更"下，并标 **不兼容**。

## [0.0.3] - 2026-11-15

有状态模型的 WebSocket 连接断开后，客户端会告诉调用方这个 episode 丢了。

### 变更

- `VlaWsClient`：已经开始的 episode 断了连接后，对有状态模型的下一次 `act` 或 `observe` 会重新连接、reset 新会话，
  然后抛出新的 `hivla.client.SessionLost`（`RuntimeError` 的子类）。以前是把请求发给没 reset 过的会话，收到服务端报错。
  捕获到它就从头开始这个 episode，下一次调用照常工作。无状态模型和以前一样只重连。

## [0.0.2] - 2026-11-14

初步支持 Pokebot 云端 RoboAPI。

### 新增

- `pokebot` family：通过 RoboAPI（`https://api.pokebot.com/v1`）调用 Pokebot 云端模型 `poke-aurora`，内置配置
  `pokebot_aurora`。输入一张腕部相机图、末端位姿和夹爪开度，返回末端绝对位姿，作为 `cartesian` 模式的 `ee_targets`。
  这个 family 有状态：每个 episode 一个 RoboAPI 会话，并按 API 要求发送执行回执。请求超时后先向服务端查这一轮的结果，
  不重发，同一段动作不会取两次。API key 从 `POKEBOT_API_KEY` 读取。已对真实 API 验证，还没上真机。
- 指南 `docs/families/pokebot.zh-CN.md`（和英文版），包括上真机前要做的 RoboAPI 标定和工具坐标系设置。
- README 的 family 画廊和 family 表里加了 PokeBot。

### 修正

- `docs/api.zh-CN.md`：`VlaWsClient` 接收服务的基础 URL，`/ws` 由客户端自己加。

## [0.0.1] - 2026-11-14

第一个公开版本：把一个 VLA checkpoint 放在统一的网络接口后面提供服务。机器人一侧发来观测，拿回一段动作
（action chunk）；所有跟模型有关的处理都在服务端。

### 新增

- **模型系列**：`lerobot`（LeRobot v0.6.2 的单帧观测 policy，比如 ACT）、`openpi`（π₀、π₀.₅、π₀-FAST）、
  `gr00t`（GR00T N1.6 / N1.7）、`starvla`（`baseframework` checkpoint，比如 QwenGR00T、QwenOFT）、
  `xpolicylab`（任意 XPolicyLab policy，有状态）和 `rlinf`（RLinf 的 PyTorch 版 π₀ / π₀.₅）。每个系列都用模型自己的代码
  和前后处理运行。`generic` 系列部署普通的 torch / ONNX / TensorRT checkpoint，用配置里的 `norm_stats` 做归一化。
  新增一个系列只需要一个 `FamilyAdapter`。
- **内置配置**，按名字使用（`hivla configs` 列出全部）：`lerobot_act_aloha`、`openpi_pi05_libero`、`gr00t_n1d6_gr1`、
  `starvla_qwengr00t_gr1`、`xpolicylab_demo_g1`、`xpolicylab_demo_ur5e_dual`、`rlinf_pi05_libero`，以及
  `example` / `echo_example` / `router_example`。checkpoint 路径从环境变量读（`${HIVLA_MODELS}`、`${OPENPI_DATA_HOME}`、
  `${RLINF_CKPT}`）；变量没设又没有默认值时直接报错，不会悄悄解析到错误的路径。
- **服务端**：`hivla serve` 在 HTTP 上提供 `POST /act`、`POST /reset`、`POST /observe`、`GET /metadata` 和
  `GET /health`，WebSocket（`/ws`）上是同样的约定。请求体是 JSON（数组用 base64 编码）或 msgpack。object dtype 的数组
  会被拒绝，所以请求里带不进 pickle。可选的 `server.api_key` 保护 HTTP 端点（`/ws` 还不查）。
- **线上用 hiRobot 的字段**：观测和回复用 hiRobot `Observation`、`ActionChunk` 的字段名（`joint_targets`、`mode`、
  `kp`、`kd`、`ee_targets`、`gripper`、`parts` 等）。`policy.state_layout` 用这些字段拼出模型的状态向量，
  `policy.gripper_range` 把模型的夹爪输出映射到 hiRobot 的开合度（0 闭合，1 张开）。`/metadata` 报告
  `protocol_version`（3）、`n_action_steps` 和模型需要机器人提供什么。
- **有状态模型的会话**：在请求之间保留历史的模型，每个 episode 开始时会被 reset，还可以读到每一个执行过的步
  （`/metadata` 里 `observe: every_step`）。有状态模型一次只服务一个会话，第二个会话会收到 `409`。
- **客户端**：`hivla.client.VlaClient`（HTTP）和 `VlaWsClient`（WebSocket）只需要 numpy 和 `hivla[client]`，
  不需要模型的任何依赖。`hivla.robot.run_episode` 让部署好的模型驱动一台 hiRobot `Robot`。
- **更快的服务**：并发请求动态合批（`max_batch_size`）；重复观测直接返回上一段动作的缓存；`torch.compile`；
  `AsyncInferenceEngine` 在机器人执行当前这段动作时预测下一段。
- **多张 GPU 共用一个地址**：`hivla route` 把 `POST /act` 分到多个服务（轮询、最少在途请求或一致性哈希）。
  连不上的后端立刻返回 `502`。
- **每个模型一个环境**：`hivla launch` 从任意环境里用配置指定的那个环境的 Python 启动服务。
  `hivla inspect` 不加载权重就能看配置解析成什么。
- 已验证：π₀.₅ 在 LIBERO 上 4 个 suite 平均 97.05%（openpi 报告 96.85%）；GR00T N1.6 在 RoboCasa GR-1 上
  5 / 10 个 episode；StarVLA QwenGR00T 在 RoboCasa GR-1 上 37 / 50；RLinf Pi05-LIBERO-SFT 跑了 LIBERO spatial；
  LeRobot ACT ALOHA 在 HTTP 和 WebSocket 上的动作形状；XPolicyLab `demo_policy` 在 hiRobot 的 MuJoCo G1 上。
  还没有部署过带真实权重的 XPolicyLab policy。

[0.0.3]: https://github.com/humanoidintelligence/hiVLA/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/humanoidintelligence/hiVLA/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/humanoidintelligence/hiVLA/releases/tag/v0.0.1
