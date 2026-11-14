# hiVLA API 参考

[English](./api.md) | 简体中文

本页列出 hiVLA 对外开放的接口：客户端调用的 HTTP / WebSocket 端点、线协议（wire format）、服务配置、CLI，
以及用来构建引擎、添加模型系列或后端的 Python API。[顶层包](#顶层包)下列出的名字都可以用
`from hivla import ...` 导入。

hiVLA 提供的服务是 `observation → action chunk`。预处理（重命名、归一化、补齐）和动作解码都在服务端完成，
所以客户端发送半原始的观测，拿回解码好的 chunk。`hivla.client` 提供了现成的客户端
（见[客户端](#客户端hivlaclient)）；hiAgent 用它们让部署好的模型驱动机器人。

## 目录

- [顶层包](#顶层包)
- [快速开始](#快速开始)
- [网络 API](#网络-api)
- [线协议](#线协议)
- [客户端](#客户端hivlaclient)
- [路由器](#路由器)
- [服务配置](#服务配置)
- [CLI](#cli)
- [ModelSpec](#modelspec)
- [构建引擎](#构建引擎)
- [InferenceEngine 与 AsyncInferenceEngine](#inferenceengine-与-asyncinferenceengine)
- [VlaPolicy](#vlapolicy)
- [变换](#变换)
- [模型系列](#模型系列)
- [后端](#后端)
- [嵌入服务器](#嵌入服务器)

## 顶层包

| 名字 | 类别 | 模块 | 用途 |
|---|---|---|---|
| `ModelSpec` | pydantic 模型 | `hivla.core.spec` | 构建并部署一个模型所需的全部信息 |
| `load_config` | 函数 | `hivla` | 读取 YAML 配置，展开 `~` 和环境变量 |
| `build_engine`, `build_engine_from_spec` | 函数 | `hivla.core.build` | 配置 / spec → `InferenceEngine` |
| `InferenceEngine`, `AsyncInferenceEngine`, `ObservationCache` | 类 | `hivla.serving.engine` | 服务生命周期、缓存、批处理 |
| `VlaPolicy` | 类 | `hivla.core.policy` | 输入变换 → 后端 → 输出变换 |
| `FamilyAdapter`, `get_adapter`, `available_families` | 类 / 函数 | `hivla.families` | 按模型系列扩展的入口 |
| `ModelBackend`, `EchoBackend`, `build_backend`, `infer_backend` | 类 / 函数 | `hivla.runtimes` | 张量进、张量出的模型运行器 |
| `__version__` | str | `hivla` | 包版本 |

其他公开模块：`hivla.client`、`hivla.protocol`、`hivla.core.transforms`、`hivla.serving.server`、`hivla.serving.router`、
`hivla.serving.router`。包里的名字在第一次使用时才导入，所以 `import hivla.client` 只加载 numpy 和
`hivla.protocol`（不加载 pydantic、后端或引擎）。

## 快速开始

```bash
pip install -e '.[server,client]'
hivla inspect --config echo_example       # resolve the config, load no weights
hivla serve   --config echo_example       # HTTP + WebSocket on :8000
```

`src/hivla/configs/echo_example.yaml` 用 echo 后端，不需要 checkpoint，返回形状为 `(action_horizon, action_dim)` 的
全零 chunk。任何没有 `model.checkpoint`（或设了 `model.backend: echo`）的配置都一样。`src/hivla/configs/example.yaml`
列出了所有选项；它写了 checkpoint，所以 `auto` 会选 `torch`。

在 Python 里用 [`hivla.client`](#客户端hivlaclient)：

```python
from hivla.client import VlaWsClient
client = VlaWsClient("http://localhost:8000")
```

要用它驱动机器人，用 hiAgent：`hiagent run --policy vla --config ...`，并把 `vla.endpoint` 指向这个服务。

## 网络 API

`hivla serve` 把同一个引擎挂在下面所有端点后面。每个请求由 `InferenceEngine.infer` 处理，它在事件循环之外运行。

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| `GET` | `/health` | 否 | `{"status": "ok"}` |
| `GET` | `/metadata` | 否 | 模型的元数据（见下文） |
| `POST` | `/act` | 设置了 `api_key` 时需要 | 输入一个编码后的观测，输出一个编码后的 action chunk |
| `POST` | `/reset` | 设置了 `api_key` 时需要 | 开始一个 episode：重置模型，清掉缓存的 chunk，把有状态模型绑定到这个会话 |
| `POST` | `/observe` | 设置了 `api_key` 时需要 | 把一个编码后的观测交给有状态模型，不返回 chunk |
| `WS` | `/ws` | 否 | 在一条长连接上收发 `act` / `observe` / `reset` / `metadata` |

**鉴权。** 设置了 `server.api_key` 后，`/act`、`/reset` 和 `/observe` 要求请求头 `Authorization: Api-Key <key>`，否则返回 `401`。`/ws` 目前还不检查密钥，所以不要把端口暴露到可信网络之外。

**元数据。** 由 spec 生成：

```json
{"protocol_version": 3, "model": "example-vla", "family": "generic",
 "action_horizon": 16, "n_action_steps": 16, "action_dim": 7,
 "control_mode": "position", "cameras": ["head", "left_wrist", "right_wrist"],
 "state_layout": null, "joint_names": null, "gripper": "raw",
 "stateful": false, "observe": "chunk"}
```

`state_layout`、`joint_names` 和 `gripper` 说明模型需要机器人提供什么（见下文线协议），客户端可以在运行前检查。

`n_action_steps` 是模型希望客户端在每个 chunk 里执行多少步再重新请求。它取配置里的 `model.n_action_steps`，
没设时取 `action_horizon`。

`stateful` 表示模型在请求之间是否保留状态。`observe` 为 `chunk` 时，模型只读 `act` 带来的观测；为 `every_step` 时，
模型还要求 chunk 里每执行一步就发一次 `observe`。

**会话。** 走 HTTP 时，会话由请求头 `X-Hivla-Session` 标识（每个 `hivla.client` 自己随机生成一个 id）；走 `/ws` 时，
一条连接就是一个会话。无状态模型不管会话。有状态模型在第一个请求之前要先 `reset`，只服务最近一次 `reset` 的那个会话，
拒绝其他会话的请求（HTTP 返回 `409`，`/ws` 回复 `error`）。新的 `reset` 会把模型交给新的会话。有状态模型不能开观测缓存，
也不能合批（`runtime.cache: false`，`max_batch_size: 1`）。

### `POST /act`

- 请求体：一个编码后的观测（见[线协议](#线协议)）。
- `Content-Type: application/json`（默认）：JSON 请求体，数组用 base64 编码，回复也是 JSON。
- `Content-Type: application/msgpack`：msgpack 请求体，数组是原始字节；回复也是 msgpack。
- `?timings=1` 会在回复里加上 `server_timings_ms: {decode, infer, encode}`。
- 状态码：`200` 带 chunk；`400` 表示请求体格式错误或不安全（例如 `object` dtype 的数组）；`401` 表示 API key 不对；
  `409` 表示有状态模型属于别的会话，或者还没有 `reset`。

### `POST /reset` 和 `POST /observe`

- `/reset` 不带请求体，回复 `{"ok": true}`。
- `/observe` 的请求体是一个编码后的观测，JSON 或 msgpack 和 `/act` 一样；回复 `{"ok": true}`，或者和 `/act` 一样返回 `409`。
- 两者都从 `X-Hivla-Session` 读会话。

### `WS /ws`

1. 连接建立后，服务端先发一个 JSON 文本帧 `{"type": "metadata", "metadata": {...}}`。
2. 客户端发送 `{"type": "act", "observation": {...}}`，可以加上 `"timings": true`。
3. 服务端回复 `{"type": "action", "chunk": {...}}`；消息有问题或会话不对时回复 `{"type": "error", "detail": "..."}`。
   连接保持打开。
4. `{"type": "reset"}` 开始一个 episode，回复 `{"type": "reset"}`；`{"type": "observe", "observation": {...}}` 回复
   `{"type": "observed"}`。

任何时候也可以发 `{"type": "metadata"}`。文本帧是 JSON；二进制帧是 msgpack，回复也是 msgpack。
服务端拒绝 permessage-deflate，因为相机图像几乎压缩不了。

## 线协议

`hivla.protocol`。字段沿用 hiRobot 的 `Observation` 和 `ActionChunk`（单位、四元数顺序 `[x, y, z, w]`、
夹爪开度、控制模式），所以回复可以逐字段转换成 `hirobot.ActionChunk`。hiVLA 不导入 hiRobot；装了 hiRobot 时，
`tests/test_hirobot_contract.py` 会检查两边一致。`/metadata` 会报告 `protocol_version`（目前是 3）。

**数组**编码成：

```json
{"__ndarray__": "<base64>", "dtype": "float32", "shape": [16, 7]}
```

走 msgpack 时，`__ndarray__` 存的是原始字节而不是 base64。只接受 bool / int / uint / float 类型；
`object` dtype 在两个方向上都会被拒绝，也从不使用 pickle。

**观测**（请求）。接受任意键。任何深度的编码数组都会被解码；标量和字符串原样传递。
客户端发送 hiRobot `Observation` 中有值的那部分字段：

| 键 | 值 |
|---|---|
| `joint_pos`, `joint_vel`, `joint_torque` | 数组 `(N,)` |
| `base_orientation` | 数组 `(4,)`，`[x, y, z, w]` |
| `base_angular_velocity`, `base_linear_acceleration`, `base_position`, `base_linear_velocity` | 数组 `(3,)` |
| `ee_poses` | `{name: array (7,)}`，`[x, y, z, qx, qy, qz, qw]` |
| `gripper` | `{name: float}`，开度在 [0, 1] 之间（0 闭合，1 张开） |
| `images` | `{camera: array HxWx3 uint8}` |
| `depth` | `{camera: array HxW}` |
| `image_timestamps` | `{camera: float}` |
| `instruction` | 字符串 |
| `extra` | `{key: array, scalar or nested dict}` |

模型读取哪些字段取决于它的模型系列、`rename_inputs` 和 `state_layout`。`state_layout`（配置的 `policy` 部分）
在模型系列看到观测之前，把结构化字段拼接进 `joint_pos`，例如 `[[ee_poses, eef], [gripper, eef]]`；
缺少字段会报错。

**Action chunk**（回复）。键是 hiRobot `ActionChunk` 的字段，只发送有值的那些：

| 键 | 值 |
|---|---|
| `mode` | 字符串：`position`、`velocity`、`torque`、`pd`、`cartesian`、`cartesian_delta`、`base_velocity`、`motion_latent`、`base_pose` |
| `joint_targets` | 数组 `(horizon, n)` |
| `kp`, `kd` | 数组 `(n,)`，所有步共用 |
| `joint_names` | `n` 个名字的列表，只出现在非关节模式里（只给部分关节的目标） |
| `ee_targets` | `{name: array (horizon, pose_dim)}` |
| `gripper` | `{name: array (horizon,)}` |
| `base_velocity`, `base_target` | 数组 `(horizon, 3)` |
| `latent` | 数组 `(horizon, D)` |
| `extra` | `{key: array, scalar or nested dict}` |
| `parts` | `{part: chunk}`，机器人 assembly 的每个部件一个 chunk，可以嵌套；带 `parts` 的 chunk 不再带其他命令字段 |
| `server_timings_ms` | 只在请求了 timings 时出现 |

服务端在发送 chunk 之前会检查它：`mode` 是已知的；每步数组的第一维（horizon）一致（设了 `action_horizon`
时要等于它）；符合 `joint_names` 的规则；每一层都符合 `parts` 的规则。不通过的 chunk 在服务端抛出异常。
配置里写了 `gripper_range: {name: [closed, open]}` 时，夹爪命令会从模型的取值范围映射到 [0, 1] 的开度，
`/metadata` 报告 `gripper: opening`（否则是 `raw`）。

| 函数 | 说明 |
|---|---|
| `encode_array(arr, binary=False)` / `decode_array(obj)` | 单个数组 |
| `encode_observation(obs, binary=False)` / `decode_observation(payload)` | 观测 dict |
| `encode_action_chunk(chunk, binary=False)` / `decode_action_chunk(payload)` | action chunk dict |
| `pack(payload)` / `unpack(data)` | msgpack 字节 |
| `encode_tree(value, binary=False)` / `decode_tree(value)` | dict / list 任意深度的数组 |
| `is_encoded_array(obj)` | `obj` 是否是编码后的数组 |
| `PROTOCOL_VERSION`, `CHUNK_FIELDS`, `CONTROL_MODES` | 线协议版本、chunk 的键、接受的模式 |
| `MSGPACK_MEDIA_TYPE` | `"application/msgpack"` |

**推理中的错误。** 如果模型本身抛出异常，`/act` 返回 `500`，WebSocket 连接以代码 `1011` 关闭。
只有格式错误的请求才会得到 `400` 或 `error` 消息。

## 客户端（`hivla.client`）

`pip install 'hivla[client]'`。这个模块只导入 numpy 和 `hivla.protocol`；`httpx` / `websockets` / `msgpack`
在第一次使用时才加载。

```python
from hivla.client import VlaWsClient

client = VlaWsClient("http://localhost:8000")         # or VlaClient(...) for HTTP
meta = client.reset()                                 # /metadata: n_action_steps, ...
chunk = client.act({"joint_pos": q, "images": {"head": img}, "instruction": "pick"})
chunk["joint_targets"]                                # (action_horizon, action_dim)
client.close()
```

| 名字 | 说明 |
|---|---|
| `VlaClient(endpoint, timeout=10.0, api_key=None, wire="json", timings=False)` | 每次调用一个 HTTP 请求（`POST /act`、`/reset`、`/observe`，`GET /metadata`），复用连接池。`session` 是它随机生成的会话 id，作为 `X-Hivla-Session` 发出 |
| `VlaWsClient(endpoint, timeout=10.0, api_key=None, wire="msgpack", timings=False)` | 一条 `/ws` 长连接，这条连接就是它的会话；重连后是新会话。`endpoint` 是基础 URL，可以是 `http(s)://` 或 `ws(s)://`；客户端自己会加 `/ws` |
| `act(observation) -> dict` | 发送一个普通的观测 dict（numpy 数组、嵌套的 `images` / `ee_poses`、`instruction`），返回解码后的 chunk dict（`mode`、`joint_targets`、`kp`、`kd`、`ee_targets`、`gripper`；只含服务端发来的键） |
| `reset() -> dict` | 开始一个 episode：把服务端的元数据取到 `metadata` 里，通知服务端 reset，返回元数据。服务端没起来时 `metadata` 是 `{}`，错误会在下一次 `act` 时出现；服务端起来了但拒绝 reset 时直接抛错 |
| `observe(observation)` | 发送一个观测，不要 chunk；用于 `observe: every_step` 的模型 |
| `metadata` | 最近一次的元数据；`VlaWsClient` 在连接握手时就填好 |
| `close()` | 释放连接。可以重复调用 |
| `SERVER_TIMINGS_KEY` | `"server_timings_ms"`：设了 `timings=True` 时，chunk 在这个键下带上服务端各阶段的耗时（毫秒） |

两个客户端都用锁串行化调用，所以预取线程和控制循环可以共用一个客户端。`VlaWsClient` 从不启用
permessage-deflate，也不限制消息大小。超时或连接断开后它会丢掉这条连接，迟到的回复不会被当成下一个请求的回复；
下一次调用会重新连接。服务端回复 `error` 时抛出 `RuntimeError`；HTTP 错误抛出 `httpx.HTTPStatusError`。

客户端只负责传数据：缓存 chunk、重放 chunk、按 `n_action_steps` 截断 chunk 都由调用方负责。每个 episode 之前调用
`reset()`；有状态模型在 reset 之前拒绝请求，用 `VlaWsClient` 时重连之后也要再 reset。

## 路由器

`hivla route --config router.yaml` 启动一个网关，把 `POST /act` 负载均衡到多个 `hivla serve` 后端
（例如每块 GPU 一个）。它自己不加载模型。客户端连路由器的方式和连单个服务完全一样。需要
`pip install 'hivla[server,client]'`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "backends": [...], "policy": "..."}` |
| `POST` | `/act` | 转发给一个后端。请求体和 `?timings=` 原样转发 |

`/ws`、`/metadata`、`/reset` 和 `/observe` 不经过路由，所以有状态模型不要放在路由后面；需要这些端点时直接连后端。连不上的后端返回 `502`；后端返回的错误状态码
（例如 `400`）原样透传。

```yaml
router:
  backends:                 # base URLs; ${VAR}, ~ and ${VAR:-default} are expanded
    - http://127.0.0.1:8001
    - http://127.0.0.1:8002
  policy: round_robin       # round_robin | least_outstanding | consistent_hash
  host: 0.0.0.0
  port: 8000
  timeout: 30.0             # seconds per forwarded request
  # api_key: secret         # required on the router's /act and forwarded upstream
```

| 策略 | 行为 |
|---|---|
| `round_robin` | 轮流分配；不解码请求体 |
| `least_outstanding` | 选正在处理的请求最少的后端 |
| `consistent_hash` | 对观测内容做哈希，相同的观测会到达同一个后端并命中它的缓存。需要解码请求体；格式错误的请求体返回 `400` |

Python：`hivla.serving.router.create_router_app(backends, policy="round_robin", api_key=None, timeout=30.0)`
返回 FastAPI app，`run_router(...)`（参数相同，外加 `host`、`port`）运行它。`hivla.serving.router.make_policy(name)`
构建一个路由策略。

## 服务配置

`hivla serve` / `hivla inspect` 通过 `load_config` 读取的 YAML 文件。字符串值里，`~` 展开为主目录，
`$VAR` / `${VAR}` 展开为环境变量，`${VAR:-default}` 展开为该变量或默认值。引用了未设置且没有默认值的变量会抛出
`KeyError`。示例在 `src/hivla/configs/` 里。

```yaml
model:
  name: example-vla
  family: generic           # generic | lerobot | openpi | gr00t | starvla | rlinf
  checkpoint: ${CKPT_DIR}/example.pt
  backend: auto             # generic family only: auto | echo | torch | onnx | tensorrt | remote
  action_horizon: 16
  n_action_steps: 8         # steps a client executes per chunk; default action_horizon

runtime:
  device: cuda              # cuda | cpu
  precision: fp16           # fp32 | fp16 | bf16
  max_batch_size: 1         # > 1 enables dynamic batching
  batch_timeout: 0.002      # seconds to collect a batch
  cache: false              # replay the last chunk for an identical observation
  cache_ttl: 0.5
  warmup: true              # one dummy forward pass at start

policy:
  action_dim: 7
  control_mode: position    # `mode` attached to chunks that do not set one
  cameras: [head, left_wrist]
  norm_stats: path/to/norm_stats.json   # or an inline dict; generic family only
  rename_inputs: {joint_pos: state}     # observation key -> model key
  rename_outputs: {action: joint_targets}
  state_layout: [[joint_pos], [gripper, eef]]   # observation fields -> joint_pos, in order
  gripper_range: {eef: [1, -1]}       # model's value when [closed, open] -> opening [0, 1]
  joint_names: [joint_1, joint_2]     # training joint order, reported in /metadata

native_kwargs: {}           # top level: family-specific loader arguments

server:
  transport: http           # informational; /act and /ws are always mounted
  host: 0.0.0.0
  port: 8000
  api_key: secret           # optional
  extensions: []            # "module:attr" callables (app, engine) -> None
```

每个字段对应 `ModelSpec` 中同名的字段。`model` / `runtime` / `policy` / `server` 之外的顶层键
（例如 `native_kwargs`）作为额外字段保留在 spec 上，由模型系列读取自己需要的部分。

`hivla serve` 目前绑定 `--host` / `--port`，默认是 `0.0.0.0` 和 `8000`，所以配置里的 `server.host` /
`server.port` 对它不起作用。要改地址就传这两个参数。`hivla route` 在没给参数时会使用 `router.host` / `router.port`。

`norm_stats` 把一个键映射到它的统计量。模式有 `zscore`（`mean`、`std`）、`quantile`（`q01`、`q99`，映射到
`[-1, 1]`）和 `minmax`（`min`、`max`，映射到 `[-1, 1]`）。省略 `mode` 时，按出现了哪些字段推断：

```json
{"state":  {"mean": [...], "std": [...]},
 "action": {"q01": [...], "q99": [...], "mode": "quantile"}}
```

## CLI

安装后的命令名是 `hivla`。`--config` 接受文件路径，或 `src/hivla/configs/` 里内置配置的名字（`gr00t_n1d6_gr1`）；
同名时已有的文件优先。

| 命令 | 说明 |
|---|---|
| `hivla --version` | 打印版本 |
| `hivla configs` | 列出内置配置 |
| `hivla inspect --config PATH` | 打印解析后的元数据、`auto` 解析成的后端（非 generic 模型系列显示 `family:<name>`）以及可用的模型系列。不加载权重 |
| `hivla serve --config PATH [--host H] [--port P]` | 构建引擎、加载模型，提供 HTTP + WebSocket 服务（阻塞） |
| `hivla launch --config PATH [--env ENV] [--host H] [--port P] [--ready-timeout S] [--log FILE]` | 用配置里 `launch.env`（conda 环境名、前缀或解释器路径）的 Python 运行 `hivla serve`，等到 `/health` 应答 |
| `hivla route --config PATH [--host H] [--port P]` | 按配置的 `router:` 部分运行[路由器](#路由器) |

## ModelSpec

`hivla.core.spec.ModelSpec`，一个允许额外字段的 pydantic 模型，是对一个可部署模型的唯一描述。

| 分组 | 字段 | 默认值 |
|---|---|---|
| model | `name` | `"vla"` |
| | `family` | `"generic"` |
| | `checkpoint` | `None` |
| | `backend` | `"auto"` |
| | `action_horizon` | `16` |
| | `n_action_steps` | `None`（对外报告为 `action_horizon`）；必须在 1..`action_horizon` 之间 |
| runtime | `device` | `"cuda"` |
| | `precision` | `"fp32"` |
| | `max_batch_size` | `1` |
| | `batch_timeout` | `0.002` |
| | `cache` | `False` |
| | `cache_ttl` | `0.5` |
| | `warmup` | `True` |
| policy | `action_dim` | `7` |
| | `control_mode` | `"position"` |
| | `cameras` | `[]`（列表，或 `{name: [h, w]}`） |
| | `norm_stats` | `None`（路径或 dict） |
| | `rename_inputs`, `rename_outputs` | `{}` |
| | `state_layout`, `gripper_range`, `joint_names` | `None` |
| server | `transport` | `"http"` |
| | `host` | `"0.0.0.0"` |
| | `port` | `8000` |
| | `api_key` | `None` |

| 方法 | 说明 |
|---|---|
| `ModelSpec.from_config(config)` | 接受嵌套的 YAML 结构或扁平的 dict。顶层键覆盖嵌套部分里的同名键 |
| `load_norm_stats() -> Optional[Dict[str, NormStats]]` | 从 JSON 路径或内联 dict 解析 `norm_stats` |
| `metadata() -> dict` | `/metadata` 的内容 |

额外的键（例如 `native_kwargs`）可以通过 `spec.model_extra` 取到。

## 构建引擎

```python
from hivla import load_config, build_engine
from hivla.serving.server import serve

engine = build_engine(load_config("example"))
serve(engine, host="0.0.0.0", port=8000)
```

| 函数 | 说明 |
|---|---|
| `load_config(path) -> dict` | 读取 YAML（路径或内置配置名），展开 `~` / 环境变量 |
| `build_engine(config) -> InferenceEngine` | 先 `ModelSpec.from_config(config)`，再 `build_engine_from_spec` |
| `build_engine_from_spec(spec) -> InferenceEngine` | 查找模型系列适配器，构建它的后端和变换，包进一个 `VlaPolicy`；设了 `spec.cache` 时加上 `ObservationCache` |

返回的引擎还没启动。`serve()` 会启动它；在进程内直接使用引擎时，自己调用 `engine.start()`。

## InferenceEngine 与 AsyncInferenceEngine

`hivla.serving.engine`。

### `InferenceEngine(policy, cache=None, max_batch_size=1, batch_timeout=0.002, warmup=True)`

| 成员 | 说明 |
|---|---|
| `start()` | 加载后端；设了 `warmup` 时运行 `backend.warmup()`；`max_batch_size > 1` 时启动批处理线程。可以重复调用 |
| `infer(observation, session=None) -> dict` | 解码后的观测 dict → action chunk dict。在 `start()` 之前调用会抛出 `RuntimeError`，有状态模型属于别的会话时抛出 `SessionError`。线程安全；开了批处理时，并发调用会合并成一次 `VlaPolicy.infer_batch` 调用 |
| `reset(session=None)` | 重置模型、清空缓存；有状态模型绑定到 `session` |
| `observe(observation, session=None)` | 把一个观测交给有状态模型。无状态模型忽略它 |
| `stateful` | 模型是否在请求之间保留状态。有状态模型不接受缓存或 `max_batch_size > 1` |
| `stop()` | 停止批处理并卸载后端 |
| `metadata` | policy 的元数据 dict |

批处理最多等待 `batch_timeout` 秒，或者等到积压了 `max_batch_size` 个请求。一个批次里出现的异常会抛给该批次的每个请求。

### `ObservationCache(max_size=16, ttl=0.5)`

从观测内容（每个数组的形状、dtype 和 CRC32）到对应 chunk 的 LRU 缓存。`ttl` 秒内见过完全相同的观测时，
`InferenceEngine` 直接返回缓存的 chunk。方法：`get(obs)`、`put(obs, chunk)`、`clear()`、`len()`。

### `AsyncInferenceEngine(policy)`

在后台线程里推理，让推理和 chunk 的执行重叠。只在库层面使用，服务端不用它。

| 方法 | 说明 |
|---|---|
| `start()` | 加载后端并启动工作线程。不运行 warmup |
| `submit(observation)` | 交上最新的观测（非阻塞）。被更新观测取代的旧观测会被丢弃。在 `start()` 之前调用会抛出 `RuntimeError` |
| `get_action() -> Optional[dict]` | 最近一次完成的 chunk；还没有就返回 `None`。在更新的 chunk 完成之前，它一直返回同一个 chunk |
| `stop()` | 停止工作线程并卸载后端 |
| `metadata` | policy 的元数据 dict |

## VlaPolicy

`hivla.core.policy.VlaPolicy(backend, input_transforms=(), output_transforms=(),
action_horizon=None, default_mode="position", metadata=None)`

流水线是 `observation → input_transforms → backend.predict → output_transforms → chunk`。

| 方法 | 说明 |
|---|---|
| `infer(observation) -> dict` | 对一个观测运行流水线 |
| `infer_batch(observations) -> list` | 逐个变换输入，调用一次 `backend.predict_batch`，再逐个解码输出 |
| `metadata` | `metadata` 参数，给了 `action_horizon` 时再加上它 |

输出变换没有设置 `mode` 时，chunk 的 `mode` 取 `default_mode`。然后进行校验：`joint_targets` 必须是 2 维，
`ee_targets` 的每一项是 2 维，`gripper` 的每一项是 1 维；设了 `action_horizon` 时，它们的第一维必须等于它。
校验失败抛出 `ValueError`。

## 变换

`hivla.core.transforms`。每个变换都是一个 frozen dataclass，提供 `__call__(dict) -> dict`。它返回一个新的 dict，
不存在的键直接跳过。`compose(transforms)` 把一串变换串成一个可调用对象。

| 变换 | 参数 | 作用 |
|---|---|---|
| `RenameKeys` | `mapping: {src: dst}` | 重命名键 |
| `SelectKeys` | `keys` | 只保留列出的键 |
| `Normalize` | `stats: {key: NormStats}`, `mode=None`, `eps=1e-6` | 用各键的统计量做归一化。`mode` 覆盖所有统计量自带的模式 |
| `Unnormalize` | 同上 | `Normalize` 的逆变换 |
| `Slice` | `key`, `dim` | 保留最后一维的前 `dim` 项 |
| `PadToDim` | `key`, `dim`, `value=0.0` | 把最后一维补齐（或截断）到 `dim` |
| `SinCos` | `key` | 把最后一维替换成 `[sin(x), cos(x)]`（宽度翻倍） |
| `AsFloat` | `key`, `scale=1.0`, `offset=0.0` | `float32(x) * scale + offset` |
| `ResizeImages` | `key`, `size` | 把一张 `HxWxC` 图像或一个 `{name: image}` dict 缩放到 `size x size`。需要 `vision` extra（Pillow） |

`NormStats(mean, std, q01, q99, min, max, mode)` 存放一个键的统计量。`NormStats.from_dict(d)` 从 JSON 构建它。
`resolve_mode()` 返回 `mode`，没设时按出现的字段推断（默认 `zscore`）。统计量比数据宽时，按数据最后一维的宽度截取。

只有 `generic` 模型系列会应用 `norm_stats`。每个模型系列都会在后端之前应用 `rename_inputs`，在后端之后应用
`rename_outputs`。`build_engine` 在每个模型系列外面加两个变换：输入端最前面的 `ConcatState`（`state_layout`），
输出端最后面的 `GripperToOpening`（`gripper_range`）。

## 模型系列

`hivla.families`。模型系列适配器决定一个系列的模型怎么加载（后端）以及输入输出怎么包装（变换）。添加一个模型系列
就是注册一个适配器，不需要继承 `VlaPolicy`。每个系列的环境、配置项和字段映射见它的
[指南](families/README.zh-CN.md)。

| 模型系列 | 加载方式 | 说明 |
|---|---|---|
| `generic` | `build_backend(spec.backend, ...)` | 归一化由 hiVLA 负责：`rename_inputs` → `Normalize` → 后端 → `Unnormalize` → `rename_outputs` |
| `lerobot` | LeRobot v0.6.2 checkpoint 及其预处理 / 后处理器 | 观测的键是 lerobot 的特征名（`observation.state`、`observation.images.<cam>`） |
| `openpi` | openpi `create_trained_policy`（π0 / π0.5 / π0-FAST，JAX 或 PyTorch） | 需要 `native_kwargs.config` 和 `native_kwargs.cameras` |
| `gr00t` | Isaac GR00T N1.6 / N1.7 `Gr00tPolicy` | 需要 `native_kwargs.embodiment_tag`。把 `joint_pos` 拆成 GR00T 的各个状态组，再把各个动作组拼成 `joint_targets` |
| `starvla` | StarVLA `baseframework` checkpoint | 状态编码、图像缩放和反归一化在服务端完成 |
| `rlinf` | RLinf 用 PyTorch 重写的 openpi | 需要 `native_kwargs.config` |

非 generic 的模型系列忽略 `model.backend` 和 `norm_stats`，加载选项从顶层的 `native_kwargs` 读取。
`src/hivla/configs/` 里每个模型系列的示例配置列出了它需要的键。

```python
from hivla.families import FamilyAdapter, register_family

class MyFamily(FamilyAdapter):
    name = "my_family"
    def build_backend(self, spec):
        return MyBackend(spec.checkpoint)
    def build_transforms(self, spec):
        return [], []           # (input_transforms, output_transforms)

register_family(MyFamily())
```

| 函数 / 类 | 说明 |
|---|---|
| `FamilyAdapter` | 抽象类：`name`、`build_backend(spec)`、`build_transforms(spec) -> (inputs, outputs)` |
| `register_family(adapter)` | 以 `adapter.name` 注册一个适配器实例。同名的已有适配器会被替换 |
| `get_adapter(family)` | 找不到时抛出 `ValueError`，并列出可用的模型系列 |
| `available_families()` | 按注册顺序列出名字 |
| `rename_transforms(spec)` | 只含 spec 的 `rename_inputs` / `rename_outputs` 的 `(inputs, outputs)`，给自己做归一化的模型系列用 |
| `GenericFamily` | `generic` 的适配器 |
| `NativeFamily(name, module, factory, call_method="infer", default_kwargs=None)` | 给这样一种包用的适配器：它的工厂函数 `module.factory(checkpoint=..., **native_kwargs)` 返回一个 policy。推理时，policy 本身可调用就直接调用它，否则调用它的 `call_method`。返回值不是 dict 时当作 `joint_targets` |

## 后端

`hivla.runtimes` 按 checkpoint 格式运行；各系列自己的后端在 `hivla.families.<name>` 里。后端把一个已经预处理好的输入 dict 映射成一个原始输出 dict。重的依赖在 `load()` 里才导入，
缺少时抛出 `ImportError`，并指明要装哪个 extra。

### `ModelBackend`

| 方法 | 说明 |
|---|---|
| `load()` | 抽象方法。加载权重或打开连接 |
| `predict(inputs) -> dict` | 抽象方法。一次前向推理 |
| `predict_batch(batch) -> list` | 默认对每一项调用 `predict`；重写它可以把一批堆叠成一次前向 |
| `warmup()` | `load()` 之后跑一次假输入的前向。默认什么也不做 |
| `unload()` | 释放资源。默认什么也不做 |

### 内置后端（generic 模型系列）

| `model.backend` | 类 | Extra | 从 spec 取的参数 |
|---|---|---|---|
| `echo` | `EchoBackend(action_dim=7, action_horizon=16)` | 无 | `action_dim`、`action_horizon`。返回全零的 `(horizon, action_dim)` 作为 `joint_targets` |
| `torch` | `TorchBackend(checkpoint, device="cuda", precision="fp32", module_factory=None)` | `torch` | `checkpoint`、`device`、`precision` |
| `onnx` | `OnnxBackend(checkpoint, device="cpu")` | `onnx` | `checkpoint` |
| `tensorrt` | `TensorrtBackend(checkpoint, device="cuda")` | `tensorrt` | `checkpoint`、`device` |
| `remote` | `RemoteBackend(endpoint, timeout=30.0, api_key=None, wire="json")` | `client` | 代理到另一个服务的 `/act` |

`build_backend(name, cfg)` 按名字用一个 dict 构建后端。`name="auto"` 时它调用 `infer_backend(checkpoint)`：
`.engine` / `.plan` 返回 `tensorrt`，`.onnx` 返回 `onnx`，没有 checkpoint 返回 `echo`，其他情况返回 `torch`。
checkpoint 是目录时，按目录里文件的扩展名决定。

`GenericFamily` 只把 `checkpoint`、`action_dim`、`action_horizon`、`device` 和 `precision` 传给 `build_backend`。
所以 `backend: remote` 没法从 YAML 配置（它需要一个 `endpoint`），`TorchBackend` 的 `module_factory` 也没法从
YAML 设置。这些后端请在 Python 里构建，再自己包进一个 `VlaPolicy`。

## 嵌入服务器

`hivla.serving.server`：

| 函数 | 说明 |
|---|---|
| `create_app(engine, api_key=None, enable_ws=True, extensions=None)` | 构建 FastAPI app。不启动引擎 |
| `serve(engine, host="0.0.0.0", port=8000, api_key=None, enable_ws=True, extensions=None)` | 先 `engine.start()`，再在关闭 permessage-deflate 的情况下运行 uvicorn（阻塞） |
| `load_extensions(paths)` | 把 `"module:attr"` 字符串解析成可调用对象 |

扩展是一个 `(app, engine) -> None` 的可调用对象，在核心路由挂载之后运行，所以可以添加自己的端点。

在进程内部署一个自定义后端：

```python
from hivla import InferenceEngine, VlaPolicy
from hivla.runtimes.base import ModelBackend
from hivla.serving.server import serve
from hivla.core.transforms import RenameKeys

class MyBackend(ModelBackend):
    def load(self): ...
    def predict(self, inputs):
        return {"joint_targets": ...}            # (horizon, dof)

policy = VlaPolicy(MyBackend(),
                   input_transforms=[RenameKeys({"joint_pos": "state"})],
                   action_horizon=16,
                   metadata={"model": "mine", "action_dim": 7})
serve(InferenceEngine(policy), port=8000)
```
