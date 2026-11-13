# LeRobot

[English](./lerobot.md) | 简体中文

系列：`lerobot`。它用 [LeRobot](https://github.com/huggingface/lerobot) 自己的流程部署 **v0.6.2** 的
policy，也就是 LeRobot 的 `SyncInferenceEngine` 跑的那条链：

```
prepare_observation_for_inference -> preprocessor -> policy.predict_action_chunk -> postprocessor
```

前后处理器是存在权重旁边的 `policy_preprocessor.json` / `policy_postprocessor.json` 流水线，所以归一化用的
是 checkpoint 自己的。hiVLA 只改键名，并把 chunk 截到 `action_horizon`。

## 支持的 checkpoint

- **只支持 processor 格式的 checkpoint**（LeRobot >= 0.4：`model.safetensors` 旁边有两个 processor 文件）。
  原来的 `lerobot/act_*` 和 `lerobot/diffusion_*` hub 仓库是旧格式，归一化缓冲在权重里；在 LeRobot 环境里
  转换一次：

  ```bash
  python <lerobot>/src/lerobot/processor/migrate_policy_normalization.py \
      --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
      --output-dir checkpoints/act_aloha_sim_transfer_cube_human
  ```

  加载旧格式的 checkpoint 会抛出 LeRobot 的 `ProcessorMigrationError`，报错信息里就有这条命令。
- **只支持单帧观测的 policy**（`n_obs_steps: 1`，例如 ACT）。带观测历史的 policy（`n_obs_steps > 1`）在加载
  时就被拒绝，因为对一个新请求调用 `predict_action_chunk` 时没有历史。

## 环境

```bash
conda create -n lerobot python=3.12 -y && conda activate lerobot
pip install -e <lerobot checkout at v0.6.2>
pip install -e ".[server,client]"              # hiVLA, from its checkout
```

## 配置

`src/hivla/configs/lerobot_act_aloha.yaml`（ACT，双臂 ALOHA 仿真，transfer-cube）：

```yaml
model:
  family: lerobot
  checkpoint: checkpoints/act_aloha_sim_transfer_cube_human   # local dir or hub repo id
  action_horizon: 100      # <= the policy's chunk_size; cut server-side

runtime:
  device: cuda

policy:
  action_dim: 14
  control_mode: position
  cameras: [top]

native_kwargs:             # all optional
  task: "transfer the cube"   # default instruction when a request has none
  robot_type: aloha
  revision: main              # hub revision
```

## 观测与动作

观测的键是 LeRobot 的 feature 名，和 checkpoint 的 `config.json` 一致：

| 请求里的键 | 例子（ACT ALOHA） | 说明 |
|---|---|---|
| `observation.state` | `(14,)` float32 | float64 会转成 float32 |
| `observation.images.<cam>` | `(480, 640, 3)` uint8 | 每个相机一个键 |
| `task` | `"transfer the cube"` | 可选；没有时用 `native_kwargs.task` |

发送 hiRobot 字段名（`joint_pos`、`images`、`instruction`，`hivla.robot` 和 hiAgent 就是这样发的）的调用方，
需要改顶层的键名：`policy.rename_inputs: {joint_pos: observation.state, instruction: task}`。`rename_inputs`
只改顶层的键，所以这个系列目前还不会把嵌套的 `images: {cam: ...}` 映射成 `observation.images.<cam>`；
暂时由调用方发平铺的键。

返回的是形状为 `(action_horizon, action_dim)` 的 `joint_targets`，单位和 policy 训练时一致。

## 已验证

`scripts/smoke_lerobot_act.py` 在 GPU 上加载 `src/hivla/configs/lerobot_act_aloha.yaml`，发送合成的 ALOHA 观测，检查每段
chunk 都是 `(action_horizon, 14)`，进程内、HTTP（`--http`）和 WebSocket（`--ws`）都测，JSON 和 msgpack
（`--wire`）都测。还没有用 hiVLA 测过闭环成功率。

## 常见问题

- `action_horizon` 大于 policy 的 `chunk_size` 时，第一次请求就报错（`predicts N steps < action_horizon`）。
- 不使用 LeRobot 的 `select_action`（从内部队列里取一步）；hiVLA 总是返回整段 chunk，逐步执行由调用方负责。
