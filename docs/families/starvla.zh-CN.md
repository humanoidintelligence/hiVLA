# StarVLA

[English](./starvla.md) | 简体中文

系列：`starvla`。它部署 [StarVLA](https://github.com/starVLA/starVLA) 的 `baseframework` checkpoint（QwenGR00T、
QwenOFT 等）。StarVLA 的 `predict_action` 返回归一化后的动作，把状态编码、图像缩放和反归一化留给每个评测的
客户端（`examples/<benchmark>/eval_files/model2*_interface.py`）。hiVLA 把这些客户端的工作搬到服务端，所以机器人
只发普通的 `joint_pos`、`images` 和 `instruction`。对 RoboCasa GR-1，它和 StarVLA 的
`model2robocasa_interface.py` 完全一致。

## 环境

按 StarVLA 的说明配好环境（验证过的那个：Python 3.10、torch 2.6.0+cu124、transformers 4.57、flash-attn 2.7.4），
再加上 hiVLA 的服务依赖。锁住 `typing_extensions`，免得它们把环境里的版本升级掉：

```bash
pip install -e <hiVLA>"[server,client]" "typing_extensions==4.15.0"
```

后端还需要 OpenCV（`cv2`），StarVLA 环境里已经有了。

## 配置

`src/hivla/configs/starvla_qwengr00t_gr1.yaml`（Qwen3-VL-GR00T，RoboCasa GR-1，9 万步）：

```yaml
model:
  family: starvla
  checkpoint: ${HIVLA_MODELS}/Qwen3-VL-GR00T-Robocasa-gr1/checkpoints/steps_90000_pytorch_model.pt
  action_horizon: 16

native_kwargs:
  base_vlm: ${HIVLA_MODELS}/Qwen3-VL-4B-Instruct-Action
  unnorm_key: gr1
  state_groups: [7, 7, 6, 6, 3]
  state_encoding: sincos
  image_size: [224, 224]
  bf16: true

policy:
  action_dim: 29
  control_mode: position
  cameras: [ego_view]
```

`checkpoint` 是 `.pt` 文件。它的 `config.yaml` 和 `dataset_statistics.json` 在上两级目录，StarVLA 的
`read_mode_config` 就在那里找。

| `native_kwargs` 键 | 默认值 | 含义 |
|---|---|---|
| `base_vlm` | 取自 `config.yaml` | VLM 的路径；checkpoint 里存的是相对训练目录的路径，所以要设置 |
| `unnorm_key` | 唯一的那个键 | 用 `dataset_statistics.json` 里哪一项反归一化 |
| `state_groups` | 无（不用状态） | `joint_pos` 拆成的各组大小 |
| `state_encoding` | `none` | `sincos`：每组变成 `[sin(g), cos(g)]`（29 -> 58） |
| `image_size` | `[224, 224]` | OpenCV `INTER_AREA` 缩放的 `[w, h]` |
| `bf16` | `true` | 权重转成 bfloat16，和 StarVLA 的评测服务一样 |
| `compile` | 关 | DiT 动作头的 `torch.compile` 模式 |
| `task` | 无 | 默认指令 |

## 观测与动作

| 观测字段 | 送到 | 说明 |
|---|---|---|
| `images[cam]` | `image` 列表 | 按 `policy.cameras` 的顺序，缩放到 `image_size` |
| `joint_pos` | `state (1, state_dim)` | 拆成 `state_groups` 后编码 |
| `instruction` | `lang` | 必填，或用 `native_kwargs.task` |

归一化的动作先裁到 [-1, 1]，再用 `unnorm_key` 统计量的 min / max 映射回去（被 mask 的维度原样通过），作为
`joint_targets` `(action_horizon, action_dim)` 返回。gr1 是绝对弧度，关节顺序和状态一样，都是那 29 个关节。

加载时检查统计量的动作宽度等于 `policy.action_dim`，且 `action_horizon` 不超过 checkpoint 的。

## 已验证

QwenGR00T 在 RoboCasa GR-1 `PnPBottleToCabinetClose` 上，经 hiRobot 的 RoboCasa driver：50 集成功 37 集（0.74）。
StarVLA 自己的客户端跑了四次，各 50 集，为 0.58–0.74。

## 常见问题

- checkpoint 里有多份数据集统计量时需要 `unnorm_key`；报错会列出可选的键。
- 不设 `base_vlm` 时，checkpoint 里存的相对 VLM 路径会相对服务的工作目录解析，通常会失败。
