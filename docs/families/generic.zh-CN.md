# generic：torch / ONNX / TensorRT checkpoint

[English](./generic.md) | 简体中文

系列：`generic`（不设 `model.family` 时的默认值）。它部署只是一个网络的 checkpoint：数组进，数组出。
前后处理由 hiVLA 按配置来做：改键名，用你的 `norm_stats` 归一化输入、反归一化输出。

自己导出的模型用它。自带处理流程的 checkpoint（LeRobot、openpi、GR00T 等）应该用对应的具名系列，结果才和
原始发布一致。

## 环境

hiVLA 加上对应后端的 extra：

```bash
pip install -e ".[server,client,torch]"      # or onnx / tensorrt
```

`echo` 后端不需要 extra 也不需要 checkpoint：它返回全零的 chunk，用来试服务端和客户端
（`src/hivla/configs/echo_example.yaml`）。

## 配置

`src/hivla/configs/example.yaml` 列出了所有选项。和 `generic` 有关的部分：

```yaml
model:
  family: generic
  checkpoint: ${CKPT_DIR}/policy.onnx
  backend: auto             # auto | echo | torch | onnx | tensorrt
  action_horizon: 16

policy:
  action_dim: 7
  control_mode: position
  cameras: [head]
  norm_stats: ${CKPT_DIR}/norm_stats.json
  rename_inputs: {joint_pos: state}
  rename_outputs: {action: joint_targets}
```

`backend: auto` 按 checkpoint 选：`.engine` / `.plan` 是 TensorRT，`.onnx` 是 ONNX，没有 checkpoint 是
`echo`，其他都是 torch。checkpoint 是目录时，看里面的文件扩展名。

| `model.backend` | 加载什么 | 设备 |
|---|---|---|
| `echo` | 什么都不加载；返回全零 `(action_horizon, action_dim)` | 无 |
| `torch` | 用 `torch.save` 存的整个 `nn.Module` / TorchScript（`torch.load`） | `runtime.device`、`runtime.precision` |
| `onnx` | 一个 ONNX Runtime 会话 | `runtime.device`（`cuda` 时加上 CUDA provider） |
| `tensorrt` | 序列化的 TensorRT engine | CUDA |

`remote`（转发给另一台 hiVLA 服务）不能从 YAML 配置，因为它需要 `endpoint`；要在 Python 里构建（见
[后端](../api.zh-CN.md#后端)）。

## hiVLA 做了什么

```
observation -> ConcatState (state_layout) -> rename_inputs -> Normalize
            -> backend -> Unnormalize -> rename_outputs -> GripperToOpening -> chunk
```

`norm_stats` 把一个键映射到它的统计量：`mean` / `std`（z-score）、`q01` / `q99`（分位数，映射到 [-1, 1]）
或 `min` / `max`（映射到 [-1, 1]）。模式按出现的字段推断。统计量比数据宽时，截到数据的宽度。

torch 模块直接返回一个张量时，它就是 `joint_targets`。否则后端的输出要包含形状为
`(action_horizon, action_dim)` 的 `joint_targets`（或改名成它的键），或其他 chunk 字段（`ee_targets`、
`gripper`）。chunk 发出前会按 hiRobot 的 `ActionChunk` 规则检查。

## 常见问题

- checkpoint 的输入名要和 `rename_inputs` 之后的键一致。先运行 `hivla inspect --config ...`；它解析配置和
  后端，不加载权重。
- 只有 `generic` 使用 `norm_stats`。具名系列忽略它，因为模型自己做归一化。
