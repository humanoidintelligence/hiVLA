# openpi（π₀ / π₀.₅ / π₀-FAST）

[English](./openpi.md) | 简体中文

系列：`openpi`。它用 [openpi](https://github.com/Physical-Intelligence/openpi) 自己的 `create_trained_policy`
和 `Policy.infer` 部署 Physical Intelligence 的 checkpoint，JAX 或 PyTorch 都行。归一化、分词，以及具名训练
配置里针对机器人的输入 / 输出 transform（`LiberoInputs`、`AlohaInputs` 等）都归 openpi 管。hiVLA 做 openpi
评测客户端在发送前做的事（`examples/libero/main.py`）：把 hiRobot 的字段映射到 openpi 的键，缩放图像，把
动作向量拆成 chunk 字段。

## 环境

openpi 在 `uv.lock` 里锁定了自己的 Python（3.11）和 jax / torch 版本；按它的说明安装，再把 hiVLA 装进同一个
环境：

```bash
cd openpi && uv sync && source .venv/bin/activate
pip install -e <hiVLA>"[server,client]"
```

checkpoint 经 openpi 的缓存解析，所以要设置 `OPENPI_DATA_HOME`；`src/hivla/configs/openpi_pi05_libero.yaml` 从它读取
checkpoint。第一次用到 `gs://openpi-assets/...` 的 checkpoint 时，openpi 会把它下载到这里。

## 配置

`src/hivla/configs/openpi_pi05_libero.yaml`（π₀.₅-LIBERO，JAX）：

```yaml
model:
  family: openpi
  checkpoint: ${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_libero
  action_horizon: 10
  n_action_steps: 5        # openpi's replan_steps: execute 5 of 10

native_kwargs:
  config: pi05_libero      # openpi train-config name (required)
  cameras:                 # robot camera -> openpi key (required)
    agentview: observation/image
    wrist: observation/wrist_image
  state_key: observation/state
  image_size: [224, 224]
  action_layout:           # how the action vector leaves the server
    - [ee_targets, eef, 6]
    - [gripper, gripper, 1]

policy:
  action_dim: 7
  control_mode: cartesian_delta
  cameras: [agentview, wrist]
```

| `native_kwargs` 键 | 默认值 | 含义 |
|---|---|---|
| `config` | 必填 | openpi 训练配置，例如 `pi05_libero`、`pi0_aloha_sim` |
| `cameras` | 必填 | `{机器人相机: openpi 图像键}` |
| `state_key` | `observation/state` | `joint_pos` 放到哪个键 |
| `image_size` | `[224, 224]` | openpi `resize_with_pad` 的 `[h, w]` |
| `action_layout` | 无 | `[field, name, size]` 切片列表；不设时整个向量是 `joint_targets` |
| `task` | 无 | 默认 prompt |
| `seed` | openpi 的（0） | JAX 采样种子 |
| `sample_kwargs` | 无 | 传给 `sample_actions`，例如 `{num_steps: 10}` |

`action_horizon` 不能超过训练配置的 horizon；加载时会检查。

## 观测与动作

| 观测字段 | 送到 | 说明 |
|---|---|---|
| `images[cam]` | `obs[cameras[cam]]` | 带填充缩放到 `image_size`，uint8 |
| `joint_pos` | `obs[state_key]` | 模型的状态向量，展平 |
| `instruction` | `prompt` | 没有时用 `native_kwargs.task` |

`action_layout` 从左到右切 `actions (action_horizon, action_dim)`。`field` 是 `ee_targets`（保持 `(T, size)`）、
`gripper`（每个名字 `(T,)`）或 `joint_targets`。LIBERO 的 7 维动作变成 `ee_targets["eef"]` `(10, 6)`（位置增量
加轴角增量）和 `gripper["gripper"]` `(10,)`，`mode: cartesian_delta`。ALOHA 和 DROID 的关节配置不设
`action_layout`，得到 `joint_targets`。

LIBERO 的机器人一侧照 openpi 的评测来：两路图像发送前旋转 180°，`joint_pos` 放 8 维状态（末端位置 3 |
轴角 3 | 夹爪 qpos 2），而不是手臂关节。

## 已验证

π₀.₅-LIBERO（JAX）在一块 H200 上，热态每次请求约 71 ms，经 hiRobot 的 LIBERO driver 走 `/ws` msgpack，
每个 suite 500 集：

| Suite | spatial | object | goal | 10 | 平均 |
|---|---|---|---|---|---|
| hiVLA | 99.0 | 98.0 | 98.2 | 93.0 | **97.05** |
| openpi 公布值 | | | | | 96.85 |

## 常见问题

- `Policy.infer` 一次只接受一个观测，所以并发请求排队执行；`max_batch_size > 1` 不会把它们合批。改为在
  `hivla route` 后面跑多个服务。
- LIBERO 的夹爪返回 [-1, 1]（-1 张开，+1 闭合），这是旧 LIBERO driver 的约定，不是 hiRobot 的开合度（0 闭合，
  1 张开）。要开合度的机器人需要 `policy.gripper_range: {gripper: [1, -1]}`。
