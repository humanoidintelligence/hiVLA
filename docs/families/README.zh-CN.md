# 模型系列

[English](./README.md) | 简体中文

一个系列是 hiVLA 知道怎样加载和运行的一类 checkpoint。每篇指南讲模型需要的环境、配置项、hiRobot 的观测
字段怎样送进模型、模型输出怎样变回 action chunk、已经验证过什么，以及常见问题。

## 支持的系列

| 系列 | 模型 | 上游 | 内置配置 | 有状态 | 指南 |
|---|---|---|---|---|---|
| `generic` | 任意 torch / ONNX / TensorRT checkpoint | 无 | `example` | 否 | [generic.zh-CN.md](generic.zh-CN.md) |
| `lerobot` | LeRobot v0.6.2 的单帧观测 policy（ACT） | [huggingface/lerobot](https://github.com/huggingface/lerobot) | `lerobot_act_aloha` | 否 | [lerobot.zh-CN.md](lerobot.zh-CN.md) |
| `openpi` | π₀、π₀.₅、π₀-FAST（JAX 或 PyTorch） | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `openpi_pi05_libero` | 否 | [openpi.zh-CN.md](openpi.zh-CN.md) |
| `gr00t` | GR00T N1.6 / N1.7 | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `gr00t_n1d6_gr1` | 否 | [gr00t.zh-CN.md](gr00t.zh-CN.md) |
| `starvla` | StarVLA `baseframework` checkpoint（QwenGR00T、QwenOFT 等） | [starVLA/starVLA](https://github.com/starVLA/starVLA) | `starvla_qwengr00t_gr1` | 否 | [starvla.zh-CN.md](starvla.zh-CN.md) |
| `xpolicylab` | 任意 XPolicyLab policy（`ModelTemplate` 接口） | [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | `xpolicylab_demo_g1` | 是 | [xpolicylab.zh-CN.md](xpolicylab.zh-CN.md) |
| `rlinf` | RLinf 用 PyTorch 重新实现的 π₀ / π₀.₅ | [RLinf/RLinf](https://github.com/RLinf/RLinf) | `rlinf_pi05_libero` | 否 | [rlinf.zh-CN.md](rlinf.zh-CN.md) |

**有状态**的系列在请求之间保留状态（观测窗口、policy 自己的缓冲）。服务端对它们一次只服务一个会话，并在
`/metadata` 里报告 `stateful: true`；见[架构](../../README.zh-CN.md#架构)和
[XPolicyLab 指南](xpolicylab.zh-CN.md)。

## 通用流程

1. **一个模型一个环境。** 每个系列锁定自己的 torch / transformers / jax 版本，而且互相冲突。按系列自己的
   说明装好它的代码，再把 hiVLA 装进同一个环境：`pip install -e ".[server,client]"`。
2. **不加载权重先检查配置：** `hivla inspect --config <名字>`，名字从 `hivla configs` 里选，也可以给自己的 YAML 路径。
3. **启动服务。** 在模型的环境里运行 `hivla serve --config ...`。在任何装了 hiVLA 的其他环境里，也可以在
   配置里加一节 `launch.env`，再运行 `hivla launch --config ...`：它用那个环境的 Python 启动服务，等到
   `/health` 应答为止。
4. **驱动机器人。** 在机器人的环境里，`hivla.robot.run_episode` 只需 `hivla[client]` 就能让一个 hiRobot
   `Robot` 对接服务；hiAgent 在它的 agent 循环里做同样的事。见
   [驱动 hiRobot 机器人](../../README.zh-CN.md#驱动-hirobot-机器人)。

每个系列都映射同样的观测字段（`joint_pos`、`images`、`instruction` 等），返回同样的 chunk 字段
（`joint_targets`、`ee_targets`、`gripper`、`mode`），所以换系列时调用方不用改。所有系列共用两个配置项来
适配机器人一侧：`policy.state_layout` 用其他观测字段拼出 `joint_pos`，`policy.gripper_range` 把模型的夹爪值
映射成 hiRobot 的开合度（0 闭合，1 张开）。两者见 [API 参考](../api.zh-CN.md#服务配置)。

## 添加系列

一个新系列就是一个 `FamilyAdapter`（用哪个后端、哪些 transform），通常再加一个后端类。见 API 参考里的
[模型系列](../api.zh-CN.md#模型系列)，并以这里现有的指南为模板写它的文档。
