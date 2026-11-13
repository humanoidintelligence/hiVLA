# XPolicyLab

[English](./xpolicylab.md) | 简体中文

系列：`xpolicylab`。一个 adapter 覆盖 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) 里的所有 policy
（`XPolicyLab/policy/<name>/`，锁定的提交里有 81 个目录）。XPolicyLab 给每个 policy 同一个 `ModelTemplate` 接口：
每个 episode 调 `reset()`，每个观测调 `update_obs(obs)`，`get_action()`（不带观测参数）返回一个列表，每步一个
dict。hiVLA 在进程内加载 `XPolicyLab.policy.<name>.model.Model`，在 hiRobot 的平铺观测和 XPolicyLab 按手臂分开
的观测之间来回映射。

policy 在调用之间保留状态，所以这个系列是**有状态**的：服务端把模型绑定到最后一次 `reset` 的会话，拒绝其他
会话（HTTP 409），并在 `/metadata` 里报告 `stateful: true, observe: every_step`。调用方在一段 chunk 的第一步之后，
每执行一步前发一次 `observe`；`/metadata` 要求时，`hivla.robot.run_episode` 会自动这样做。

## 环境

每个 policy 有自己的依赖。按 XPolicyLab 的说明装好它和该 policy 的依赖，再把 hiVLA 装进同一个环境：

```bash
pip install -e <XPolicyLab> -e <hiVLA>"[server,client]"
```

`demo_policy` 只需要 numpy，返回全零；不用权重和 GPU 就能把整条链路检查一遍。用 `hivla launch` 时，内置配置在
`${XPOLICYLAB_ENV}` 指定的 conda 环境里启动它（默认 `xpolicylab`）。

### `env_cfg`

`Model.__init__` 从 `<XPolicyLab 的上一级目录>/env_cfg/` 读机器人的维度（`<type>.yml`、`robot/_robot_info.json`、
`sim/<type>.yml`）。这些文件通常来自评测仓库，所以后端在加载前按 `native_kwargs.arms` 生成它们，条目名是
`env_cfg_type`（默认 `hivla_<模型名>`）。已有同名条目且内容不同时，它报错而不覆盖；这时换一个 `env_cfg_type`。

## 配置

`src/hivla/configs/xpolicylab_demo_g1.yaml`（`demo_policy`，hiRobot 29 自由度 `unitree_g1_mujoco` 的两条 7 关节手臂，
没有夹爪）：

```yaml
model:
  family: xpolicylab
  action_horizon: 2        # demo_policy returns 2 steps

native_kwargs:
  policy_name: demo_policy
  action_type: joint
  arms:                    # model's arm order, left first
    - {joints: [15, 22], ee: left_hand, gripper: null, ee_dim: 0}
    - {joints: [22, 29], ee: right_hand, gripper: null, ee_dim: 0}
  cameras:
    cam_head: head         # model camera -> robot camera

runtime:
  device: cpu
  max_batch_size: 1
  cache: false             # required: stateful models run without cache or batching

policy:
  action_dim: 29
  control_mode: position

launch:
  env: ${XPOLICYLAB_ENV:-xpolicylab}   # for `hivla launch`
  ready_timeout: 120

server:
  host: 127.0.0.1
  port: 8012
```

`src/hivla/configs/xpolicylab_demo_ur5e_dual.yaml` 是同一个 policy，用于两条带夹爪的 6 关节 UR5e 手臂
（`ee: left/tcp`、`right/tcp`）。

| `native_kwargs` 键 | 默认值 | 含义 |
|---|---|---|
| `policy_name` | 必填 | `XPolicyLab/policy` 下的目录名 |
| `action_type` | `joint` | `joint`（手臂关节目标）或 `ee`（末端位姿） |
| `arms` | 必填 | 一或两条手臂，按模型的顺序（左臂在前） |
| `arms[i].joints` | 必填 | hiRobot `joint_pos` 里的 `[start, stop]` |
| `arms[i].ee` | 无 | 位姿和 `ee_targets` 用的 hiRobot 末端名；`action_type: ee` 时必填 |
| `arms[i].gripper` | 同 `ee` | 夹爪用的 hiRobot 末端名；没有夹爪写 `null` |
| `arms[i].ee_dim` | `1` | 模型的末端关节数；有夹爪为 `1`，没有为 `0` |
| `cameras` | 同名 | `{模型相机: 机器人相机}` |
| `env_cfg_type` | `hivla_<模型名>` | 生成的 `env_cfg` 条目名 |
| `deploy` | 无 | 额外的 `deploy_cfg` 键（`ckpt_name` 等），覆盖在 policy 的 `deploy.yml` 之上 |

## 观测与动作

一条手臂时 `{p}` 为 `""`，两条时为 `left_` / `right_`。

| hiRobot 字段 | XPolicyLab 键 | 说明 |
|---|---|---|
| `joint_pos[start:stop]` | `state/{p}arm_joint_state` | 一条手臂时还有 `state/joint_state` |
| `gripper[name]` | `state/{p}ee_joint_state` | 开合度 [0, 1] 经 `policy.gripper_range` 映射到模型的范围 |
| `ee_poses[name]` | `state/{p}ee_pose` | 转换四元数顺序 |
| `images[机器人相机]` | `vision/<模型相机>/color` | |
| `instruction` | `instruction` | |

| XPolicyLab 动作 | hiRobot chunk 字段 | 说明 |
|---|---|---|
| `{p}arm_joint_state` | `joint_targets` 里这些手臂的列 | `action_type: joint`，`mode: position`。`arms` 以外的关节（G1 的腿、腰）保持 episode 第一帧的 `joint_pos` |
| `{p}ee_pose` | `ee_targets[ee]` | `action_type: ee`，`mode: cartesian` |
| `{p}ee_joint_state` | `gripper[gripper]` | 设置了 `gripper_range` 时映射成开合度 |

位姿：XPolicyLab 是 `[x, y, z, qw, qx, qy, qz]`，hiRobot 是 `[x, y, z, qx, qy, qz, qw]`；两个方向都会转换。

## 已验证

`demo_policy`，从 hiRobot 的环境（那里没装 XPolicyLab）用 `hivla launch` 启动：

- hiRobot 的 `unitree_g1_mujoco`，用 `hivla.robot.run_episode` 跑 100 步，HTTP 和
  WebSocket 各一次（各 50 个 chunk、50 次 observe；`act` 平均 HTTP 13.0 ms、WebSocket 7.8 ms）。最终关节位置和
  直接把同样的目标发给 hiRobot 相差不到 8e-10 rad。腿和腰保持 episode 第一帧的姿态。
- `ur5e_dual`（`xpolicylab_demo_ur5e_dual`）：chunk 的形状，HTTP 和 WebSocket，JSON 和 msgpack；第二个会话得到 409。

还没有部署过带真实权重的 policy。

## 常见问题

- 有状态的模型在启动时拒绝 `runtime.cache: true` 或 `max_batch_size > 1`。
- WebSocket 连接断开后，下一条连接是新会话。`act` 之前要重新调用 `reset`。
- policy 返回的步数少于 `action_horizon` 会报错；把 `action_horizon` 设成 `get_action` 返回的步数。
- 还不支持多关节的手（有夹爪时 `ee_dim > 1`）。
