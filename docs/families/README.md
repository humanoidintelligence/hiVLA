# Model families

English | [简体中文](./README.zh-CN.md)

A family is one kind of checkpoint hiVLA knows how to load and run. Each guide
covers the environment the model needs, the config keys, how hiRobot's
observation fields reach the model and how its output comes back as an action
chunk, what has been verified, and common pitfalls.

## Supported families

| Family | Models | Upstream | Built-in config | Stateful | Guide |
|---|---|---|---|---|---|
| `generic` | Any torch / ONNX / TensorRT checkpoint | none | `example` | no | [generic.md](generic.md) |
| `lerobot` | LeRobot v0.6.2 single-observation policies (ACT) | [huggingface/lerobot](https://github.com/huggingface/lerobot) | `lerobot_act_aloha` | no | [lerobot.md](lerobot.md) |
| `openpi` | π₀, π₀.₅, π₀-FAST (JAX or PyTorch) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `openpi_pi05_libero` | no | [openpi.md](openpi.md) |
| `gr00t` | GR00T N1.6 / N1.7 | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `gr00t_n1d6_gr1` | no | [gr00t.md](gr00t.md) |
| `starvla` | StarVLA `baseframework` checkpoints (QwenGR00T, QwenOFT, ...) | [starVLA/starVLA](https://github.com/starVLA/starVLA) | `starvla_qwengr00t_gr1` | no | [starvla.md](starvla.md) |
| `xpolicylab` | Any XPolicyLab policy (`ModelTemplate` interface) | [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | `xpolicylab_demo_g1` | yes | [xpolicylab.md](xpolicylab.md) |
| `rlinf` | π₀ / π₀.₅ in RLinf's PyTorch reimplementation | [RLinf/RLinf](https://github.com/RLinf/RLinf) | `rlinf_pi05_libero` | no | [rlinf.md](rlinf.md) |

**Stateful** families keep state between requests (an observation window, the
policy's own buffers). The server serves one session at a time for them and
reports `stateful: true` in `/metadata`; see the
[Architecture](../../README.md#architecture) and the
[XPolicyLab guide](xpolicylab.md).

## General workflow

1. **One environment per model.** Each family pins its own torch /
   transformers / jax versions, and they conflict. Install the family's own
   code by its instructions, then hiVLA into the same environment:
   `pip install -e ".[server,client]"`.
2. **Check the config without loading weights:**
   `hivla inspect --config <name>`, with a built-in name from `hivla configs`
   or the path of your own YAML.
3. **Serve it.** In the model's environment, `hivla serve --config ...`. From
   any other environment with hiVLA, add a `launch.env` section to the config
   and run `hivla launch --config ...`: it starts the server with that env's
   Python and waits until `/health` answers.
4. **Drive a robot.** From the robot's environment, `hivla.robot.run_episode`
   runs a hiRobot `Robot` against the server with only `hivla[client]`
   installed; hiAgent does the same inside its agent loop. See
   [Drive a hiRobot robot](../../README.md#drive-a-hirobot-robot).

Every family maps the same observation fields (`joint_pos`, `images`,
`instruction`, ...) and returns the same chunk fields (`joint_targets`,
`ee_targets`, `gripper`, `mode`), so the caller does not change between them.
Two config options shared by all families adjust the robot side:
`policy.state_layout` builds `joint_pos` from other observation fields, and
`policy.gripper_range` maps the model's gripper values to hiRobot's opening
(0 closed, 1 open). Both are described in the
[API reference](../api.md#serving-config).

## Adding a family

A new family is one `FamilyAdapter` (which backend, which transforms) and
usually one backend class. See [Model families](../api.md#model-families) in
the API reference, and use an existing guide here as the template for its doc.
