# GR00T（N1.6 / N1.7）

[English](./gr00t.md) | 简体中文

系列：`gr00t`。它用 NVIDIA [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.6 / N1.7 checkpoint 自己的
`Gr00tPolicy` 来部署，整条流水线都归它管：Eagle VLM 处理器、按 embodiment 的状态和动作归一化，以及把相对
动作解码成绝对动作。hiVLA 只在 hiRobot 的平铺观测和 GR00T 分组、带 batch 的格式之间转换。

## 环境

按 Isaac-GR00T 的说明配好环境（N1.6：Python 3.12、torch 2.7.1+cu128），再装上 Eagle backbone 需要的
`flash-attn` 和 hiVLA：

```bash
conda activate gr00tn1d6                         # your GR00T environment
pip install flash-attn --no-build-isolation
pip install -e <hiVLA>"[server,client]"
```

checkpoint 的 VLM backbone 在 Hugging Face 上需要申请权限：在
[`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) 申请访问，用 `hf auth login` 登录，再下载
到 `${HIVLA_MODELS}` 下：

```bash
hf download nvidia/GR00T-N1.6-3B --local-dir $HIVLA_MODELS/Isaac-GR00T-n1.6/checkpoints/GR00T-N1.6-3B
```

README 的[完整示例](../../README.zh-CN.md#完整示例在-fourier-gr-1-上部署-gr00t-n16)从全新 clone 开始走一遍这些步骤。

## 配置

`src/hivla/configs/gr00t_n1d6_gr1.yaml`（GR00T N1.6-3B，Fourier GR-1，embodiment `gr1`）：

```yaml
model:
  family: gr00t
  checkpoint: ${HIVLA_MODELS}/Isaac-GR00T-n1.6/checkpoints/GR00T-N1.6-3B
  action_horizon: 16       # gr1 predicts 16 steps
  n_action_steps: 8        # GR00T's RoboCasa eval executes 8 of 16

native_kwargs:
  embodiment_tag: gr1      # required
  cameras:
    ego_view: ego_view_bg_crop_pad_res256_freq20   # robot camera -> GR00T video key

policy:
  action_dim: 29
  control_mode: position
  cameras: [ego_view]
```

| `native_kwargs` 键 | 默认值 | 含义 |
|---|---|---|
| `embodiment_tag` | 必填 | GR00T embodiment，例如 `gr1` |
| `cameras` | 同名 | `{机器人相机: GR00T video 键}`；模型只有一个 video 键、机器人只发一张图时可省略 |
| `task` | 无 | 默认指令 |
| `strict` | `true` | GR00T 自己的输入 / 输出校验 |
| `compile` | 关 | DiT 动作头的 `torch.compile` 模式，例如 `max-autotune`；Eagle backbone 不编译 |

## 观测与动作

| 观测字段 | 送到 | 说明 |
|---|---|---|
| `joint_pos` | GR00T 的各个状态组 | 按 checkpoint modality 配置的顺序拆开。gr1：`left_arm 7, right_arm 7, left_hand 6, right_hand 6, waist 3` = 29 |
| `state`（dict） | 同样的组 | 代替 `joint_pos`，按名字给出各组 |
| `images[cam]` | `video[cameras[cam]]` | `(H, W, 3)` uint8；GR00T 自己缩放 |
| `instruction` | 语言键 | 必填，或用 `native_kwargs.task` |

各个动作组按动作键的顺序拼成 `joint_targets` `(action_horizon, dof)`，是绝对值、物理单位（gr1 是弧度）。gr1
的顺序和状态一样，都是那 29 个关节。

`runtime.max_batch_size > 1` 时，并发请求合成一次前向。

## 已验证

GR00T N1.6-3B，`gr1`，一块 H200，经 hiRobot 的 RoboCasa driver：

- RoboCasa GR-1 `PnPBottleToCabinetClose`：10 集成功 5 集。
- 热态每次请求约 100–130 ms；在服务压测里，`compile: max-autotune` 让前向从 211 ms 降到 62 ms（见
  [性能](../../README.zh-CN.md#性能)）。

`scripts/smoke_gr00t_n1d6.py --config gr00t_n1d6_gr1` 在进程内加载模型（`--http` / `--ws` 时放在
一个本地服务后面），发送合成观测，检查每段 chunk 的形状。

## 常见问题

- `joint_pos` 的宽度必须正好等于各状态组的总宽度（gr1 是 29）；报错会列出 GR00T 期望的各组。
- 只支持单步观测。modality 配置用了历史窗口（video、state 或 language 不止一步）的 embodiment 在加载时就被
  拒绝。
- `action_horizon` 不能超过 embodiment 的动作 horizon。
