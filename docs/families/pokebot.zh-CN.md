# Pokebot RoboAPI

[English](./pokebot.md) | 简体中文

family：`pokebot`。把 Pokebot 的云端模型（`poke-aurora`）接到 hiVLA 的接口后面，走的是
[RoboAPI](https://developers.pokebot.com/docs/roboapi/latest)：地址 `https://api.pokebot.com/v1`，
兼容 OpenAI 格式的 HTTP API。本地什么都不跑，不需要权重和 GPU。每次请求发一张腕部相机图、末端位姿和夹爪开度，
返回一段末端绝对位姿。

RoboAPI 把同一个任务的请求串成一个**会话（session）**，每次请求是一**轮（turn）**；下一轮必须带上一轮的回复和执行回执
（这一段动作执行了几步）。所以这个 family 是**有状态**的：服务一次只服务一个 hiVLA 会话；`reset` 会关掉 RoboAPI 会话，
之后第一次请求再开一个新的。

## 环境

只需要 `httpx` 和 Pillow：

```bash
pip install -e ".[server,client,vision]"
export POKEBOT_API_KEY=sk-...    # 在 Developer Console 创建：https://platform.pokebot.com
```

key 从 `native_kwargs.api_key_env` 指定的环境变量读（默认 `POKEBOT_API_KEY`），不会写进配置、日志或 `/metadata`。

## 配置

`src/hivla/configs/pokebot_aurora.yaml`：

```yaml
model:
  family: pokebot
  action_horizon: 40       # poke-aurora-preview 每轮返回 40 行
  n_action_steps: 40       # 客户端执行多少行再请求下一轮

native_kwargs:
  base_url: https://api.pokebot.com/v1
  model: poke-aurora-preview     # GET /models 返回的 id
  api_key_env: POKEBOT_API_KEY
  ee: eef                  # hiRobot 末端名（位姿输入、ee_targets 输出）
  camera: wrist            # 作为输入图的相机
  image_format: jpeg       # 或 png
  max_pixels: 230400       # 超过就按比例缩小

runtime:
  cache: false
  warmup: false            # 预热调用会扣余额

policy:
  control_mode: cartesian
```

| `native_kwargs` 键 | 默认 | 含义 |
|---|---|---|
| `base_url` | `https://api.pokebot.com/v1` | 保留 `/v1` 后缀 |
| `model` | `poke-aurora-preview` | `GET /models` 返回的 `id`，加载时检查 |
| `api_key_env` | `POKEBOT_API_KEY` | 存 key 的环境变量 |
| `ee` | 必填 | `ee_poses` 和 `ee_targets` 里的末端名 |
| `gripper` | 同 `ee` | `gripper` 里的夹爪名 |
| `camera` | 必填 | `images` 里的相机名 |
| `image_format` | `jpeg` | `jpeg` 或 `png` |
| `jpeg_quality` | 90 | |
| `max_pixels` | 640 × 360 | 不超过 schema 的 `max_pixels` |
| `instruction` | 无 | 观测里没有指令时用它 |
| `timeout` | 60 | 每个请求的超时秒数 |

加载时调 `GET /models`（模型必须在列表里）和 `GET /models/{model}/schema`（state 必须是 8 个数，输出必须是 8 列）。

## 输入输出

| hiRobot | RoboAPI |
|---|---|
| `ee_poses[ee]` `[x, y, z, qx, qy, qz, qw]` + `gripper[gripper]`（开度，0 闭合，1 张开） | `state` `[x, y, z, qx, qy, qz, qw, gripper]` |
| `images[camera]` HxWx3 RGB | 一个 `image_url` data URL |
| `instruction` | `instruction`（非空，最多 4096 字符） |
| `ee_targets[ee]` (N, 7)、`gripper[gripper]` (N,)、`mode: cartesian` | `execute_actions` tool call 的 `output` (N, 8) |

两边布局一样，不需要重排：hiRobot 的位姿本来就是 `xyzw`。返回的四元数会重新归一化，夹爪裁到 [0, 1]。
比 `action_horizon` 长的部分截掉。

**坐标系。** RoboAPI 要的是法兰中心（不是指尖或 TCP）在它自己标定的任务坐标系下的位姿，返回的目标也在这个坐标系下。
机器人驱动（或 hiRobot 的 body 配置）报告和接收的位姿都要在这个坐标系下，见[上真机前](#上真机前)。
标定后的 Home 位姿是 `[-0.13, -0.46, 0.49, 0, 1, 0, 0]`。

**图像。** 一个腕部相机，整帧、保留鱼眼畸变：不要裁剪、镜像、旋转或去畸变。画面里要能看到两根手指，底边露出一窄条夹爪掌部。
图像和位姿要同时采。

**频率。** 输出的每一行是一个绝对目标，RoboAPI 自己的运行时按 30 Hz 下发。执行时用同样的频率。

## 上真机前

hiVLA 只转发请求，不负责标定机器人。下面的步骤取自 RoboAPI 的
[hardware](https://developers.pokebot.com/docs/roboapi/latest/hardware.md)、
[UR 设置](https://developers.pokebot.com/docs/roboapi/latest/ur-mock.md)和
[calibration](https://developers.pokebot.com/docs/roboapi/latest/calibration.md) 几页，以当前版本为准。

1. **硬件。** RoboAPI 列出的机械臂是 UR5（CB3）、UR5e、UR7e 和 Franka FR3；夹爪是带平行锁定套件的 Robotiq 2F-85；
   腕部相机装在它的标准支架上，可选 Insta360 Ace Pro 2 + Ultra Wide Lens，或 DJI Osmo Action 6 + FOV Boost Lens，
   以 USB 摄像头方式工作在 MJPG 1280 × 720、30 FPS。鱼眼画面是模型输入的一部分，换成别的相机（RealSense、普通摄像头）
   就超出了模型训练时见过的范围。
2. **工具坐标系。** 在 UR 示教器上把 TCP 设成法兰中心、位置偏移为 0，并填真实负载。改 TCP 会改变这台臂上其他程序读到的位姿，
   改之前先记下原来的值。
3. **RoboAPI 运行时。** 装在和机械臂直连的电脑上（Linux x86-64，Python 3.10–3.12；用户要在 `video` 和 `dialout` 组里）：

   ```bash
   uv pip install 'roboapi[ur-cb3,gripper]==0.4.5'   # UR5 CB3
   uv pip install 'roboapi[gripper]==0.4.5'          # UR5e / UR7e
   roboapi init     # 机械臂型号、控制器 IP、夹爪、相机、key
   roboapi http     # 本地控制台 http://127.0.0.1:8080/
   ```

   一台臂同一时间只能有一个程序控制：RoboAPI 连着时要停掉 hiRobot（或其他控制程序）。
4. **Home + A/B/C 标定**（控制台：Settings → Devices → Coordinate calibration）。用手拖动机械臂，停稳后采集每个位姿；
   全程保持同样的工具朝向、夹爪开度和接触点：
   * **Home：** 夹爪竖直、手指朝下，离水平桌面 30–50 cm，回程路径上没有障碍。
   * **A：** 接触点碰到桌面。
   * **B：** 沿桌面朝相机左边移动；A → B 就是任务 +X。
   * **C：** 桌面上不在 AB 连线上的另一个点。之后机械臂会自己回到 Home，夹爪可能会动，要守在急停旁边。

   然后保存、启用控制，再做方向检查（从 Home 向前、向左、向上各 5 cm、30°）。任务 +Z 是指向 Home 的桌面法线。
   结果存在那台电脑上，默认 `~/.config/roboapi/calibration.json`；换了工具、相机支架、Home 或任务坐标轴都要重新标定。
5. **hiRobot 侧的坐标系。** `ee_poses[ee]` 要报告这个任务坐标系下的法兰位姿，`ee_targets[ee]` 发给机器人之前要换回去。
   直接发基座系下的 TCP 位姿，目标位置就是错的。

## 回执

下一轮会报告上一段执行了多少：

* 观测里有 `extra["executed_steps"]` 就用它；
* 否则用 `min(n_action_steps, 行数)`。

整段都执行了 `status` 为 `completed`，否则为 `interrupted`。`reset`（或服务退出）时关闭会话，带上最后一段的回执，
算法同上。

## 失败处理

* **不自动重试。** RoboAPI 强调重试不能让机器人把同一段动作执行两次。
* **超时或连接断开。** 后端向服务端查这一轮（`GET /sessions/{id}/turns/{turn}`），服务端已经算完就用存下的结果；
  否则这次请求失败。
* **其他失败**（HTTP 错误、回复对不上这一轮、`output` 格式不对）会关掉 RoboAPI 会话并让请求失败，下一次请求开新会话。
  再请求前要重新采一次观测。

| HTTP | 含义 |
|---|---|
| 400 | JSON、data URL 或 state 不对 |
| 401 | API key 不对 |
| 403 | 没有模型权限或没有账号 |
| 404 | `base_url` 不对，或模型 id 不在 `GET /models` 里 |
| 413 | 请求太大：调小 `max_pixels` 或用 JPEG |
| 429 | 频率、并发或余额限制 |
| 5xx | 服务故障 |

报错信息里带 RoboAPI 的 `X-Request-ID`，找 Pokebot 支持时提供它（不要提供 key 和原图）。

## 已验证

2026-11-14，`https://api.pokebot.com/v1`，模型 `poke-aurora-preview`（实际由 `poke-aurora-gauss-1030` 提供）：

* 用一张 640 × 360 的图和 Home 位姿请求一轮，返回 (40, 8)，服务端耗时 1.7 s。第一行等于发送的 state，所以目标是绝对位姿。
* 建会话、查询某一轮、带回执关闭会话，都和上面描述的一致。

没验证：真机，包括[上真机前](#上真机前)里的标定和坐标系步骤。

## 常见问题

* **不要放在 `hivla route` 后面。** 路由不转发 `/reset`，有状态模型放在后面会把不同 episode 混在一起。
* **每次请求都扣余额。** 预热默认关闭，`hivla inspect` 不调用 API。
* **位姿坐标系不对。** 发原始控制器位姿（或者在其他位置还发 Home 常量），返回的目标位置就是错的。要发标定后的位姿。
* **TCP 偏移没清零。** 示教器上设了夹爪 TCP，报告的位姿就是指尖而不是法兰。标定前先设成 0。
