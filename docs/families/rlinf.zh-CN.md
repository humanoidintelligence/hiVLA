# RLinf（PyTorch 版 π₀ / π₀.₅）

[English](./rlinf.md) | 简体中文

系列：`rlinf`。它部署 [RLinf](https://github.com/RLinf/RLinf) 的 checkpoint。RLinf 用 PyTorch 重新实现了 openpi
的 π₀ / π₀.₅（数值上和 JAX 参考实现对齐），并以这种形式发布它的 SFT 和 RL 微调模型：`model.safetensors` 加
`<asset_id>/norm_stats.json`，例如 `RLinf/RLinf-Pi05-LIBERO-SFT`。后端用
`rlinf.models.embodiment.openpi.get_model` 以 eval 模式构建模型，它会为具名配置装上 openpi 的输入 / 输出
transform，所以归一化、缩放到 224 和分词都在模型里完成，和 RLinf 自己的评测一样。

## 环境

RLinf 加上它的 openpi fork（Python 3.11；验证过的组合是 rlinf-openpi 0.1.1、torch 2.7.1、jax[cuda12] 0.5.3、
rlinf-transformer-openpi 4.53.2），照 RLinf 的 `requirements/install.sh` 为 openpi 模型的装法安装，再把 hiVLA
装进同一个环境。它和 `openpi` 环境要分开：两边用的 openpi 包不同。

```bash
pip install -e <RLinf> -e <hiVLA>"[server,client]"
```

需要两个环境变量，缺一个加载就失败：`RLINF_CKPT`（checkpoint 目录）和 `OPENPI_DATA_HOME`（PaliGemma
tokenizer 的缓存位置，和 openpi 共用）。

## 配置

`src/hivla/configs/rlinf_pi05_libero.yaml`（RLinf-Pi05-LIBERO-SFT，四个 LIBERO suite 共用一个模型）：

```yaml
model:
  family: rlinf
  checkpoint: ${RLINF_CKPT}
  action_horizon: 5        # the executed chunk RLinf returns (num_action_chunks)

native_kwargs:
  config: pi05_libero      # openpi train-config name (required)
  cameras:
    agentview: main_images
    wrist: wrist_images
  num_steps: 5             # flow-matching denoise steps (RLinf's LIBERO evals: 3-5)
  seed: 0
  action_layout:
    - [ee_targets, eef, 6]
    - [gripper, gripper, 1]

policy:
  action_dim: 7
  control_mode: cartesian_delta
  cameras: [agentview, wrist]
```

| `native_kwargs` 键 | 默认值 | 含义 |
|---|---|---|
| `config` | 必填 | openpi 训练配置，例如 `pi05_libero` |
| `cameras` | 无 | `{机器人相机: RLinf env-obs 键}` |
| `action_layout` | 无 | `[field, name, size]` 切片，同 [openpi](openpi.zh-CN.md#观测与动作) |
| `pi05` | `true` | π₀.₅；`false` 为 π₀ |
| `num_steps` | `5` | 去噪步数 |
| `precision` | checkpoint 的 | 参数 dtype，例如 `bf16` |
| `seed` | 无 | 初始噪声所用 CUDA generator 的种子 |
| `openpi` | 无 | 额外的 `cfg.openpi` 字段，例如 `{torch_compile: false}` |

## 观测与动作

| 观测字段 | 送到 | 说明 |
|---|---|---|
| `images[cam]` | `env_obs[cameras[cam]]` | 原样的 uint8；模型自己缩放到 224 |
| `joint_pos` | `states` | LIBERO：末端位置 3 \| 轴角 3 \| 夹爪 qpos 2 |
| `instruction` | `task_descriptions` | |

已经带 `main_images` 的观测（RLinf 自己的格式）原样传入。动作以反归一化后的 `(action_horizon, action_dim)` 返回，
按 `action_layout` 拆开：LIBERO 是 `ee_targets["eef"]` `(5, 6)` 和 [-1, 1] 内的 `gripper["gripper"]` `(5,)`。

`runtime.max_batch_size > 1` 时，并发请求合成一次前向。

## 已验证

RLinf-Pi05-LIBERO-SFT 经 hiRobot 的 LIBERO driver 跑过 LIBERO spatial suite：5 个仿真 worker 对一个服务，每批 5 个
请求（H200 上每小时 738 集）。那次运行的成功率这里还没有记录。

## 常见问题

- LIBERO 的夹爪在 [-1, 1]（-1 张开），是旧 LIBERO driver 的约定。要 hiRobot 开合度的机器人需要
  `policy.gripper_range: {gripper: [1, -1]}`。
- 开合批时，在同一个 suite 上 RLinf 每集的耗时约是 openpi 的两倍。原因还没测清楚。
