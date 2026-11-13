# XPolicyLab

English | [简体中文](./xpolicylab.zh-CN.md)

Family: `xpolicylab`. One adapter for every policy in
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) (`XPolicyLab/policy/<name>/`,
81 directories at the pinned commit). XPolicyLab gives each policy the same
`ModelTemplate` interface: `reset()` per episode, `update_obs(obs)` for each
observation and `get_action()` (no observation argument), which returns a list
with one dict per step. hiVLA loads `XPolicyLab.policy.<name>.model.Model`
in-process and maps hiRobot's flat observation to XPolicyLab's per-arm one and
back.

The policy keeps its state between calls, so this family is **stateful**: the
server binds the model to the session of the last `reset`, refuses other
sessions (HTTP 409), and reports `stateful: true, observe: every_step` in
`/metadata`. A caller sends one `observe` before each executed step after the
first of a chunk; `hivla.robot.run_episode` does this when `/metadata` asks
for it.

## Environment

Each policy has its own dependencies. Install XPolicyLab and the policy's
requirements by its instructions, then hiVLA into the same environment:

```bash
pip install -e <XPolicyLab> -e <hiVLA>"[server,client]"
```

`demo_policy` needs only numpy and returns zeros; it checks the whole path
without weights or a GPU. The built-in configs start it in the conda env named
by `${XPOLICYLAB_ENV}` (default `xpolicylab`) when you use `hivla launch`.

### `env_cfg`

`Model.__init__` reads the robot's dimensions from `<parent of XPolicyLab>/env_cfg/`
(`<type>.yml`, `robot/_robot_info.json`, `sim/<type>.yml`). Those files normally
come from a benchmark repo, so the backend writes them from `native_kwargs.arms`
before loading, under `env_cfg_type` (default `hivla_<model name>`). It refuses
to overwrite an entry that already holds something else; set another
`env_cfg_type` in that case.

## Configuration

`src/hivla/configs/xpolicylab_demo_g1.yaml` (`demo_policy` on both 7-joint arms of hiRobot's
29-DoF `unitree_g1_mujoco`, no grippers):

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

`src/hivla/configs/xpolicylab_demo_ur5e_dual.yaml` is the same policy for two 6-joint
UR5e arms with grippers (`ee: left/tcp`, `right/tcp`).

| `native_kwargs` key | Default | Meaning |
|---|---|---|
| `policy_name` | required | directory under `XPolicyLab/policy` |
| `action_type` | `joint` | `joint` (arm joint targets) or `ee` (end-effector poses) |
| `arms` | required | one or two arms, in the model's order (left first) |
| `arms[i].joints` | required | `[start, stop]` into hiRobot's `joint_pos` |
| `arms[i].ee` | none | hiRobot end-effector name for poses and `ee_targets`; required for `action_type: ee` |
| `arms[i].gripper` | `ee` | hiRobot end-effector name for the gripper; `null` for none |
| `arms[i].ee_dim` | `1` | the model's end-effector joint count; `1` with a gripper, `0` without |
| `cameras` | same names | `{model camera: robot camera}` |
| `env_cfg_type` | `hivla_<model name>` | name of the generated `env_cfg` entry |
| `deploy` | none | extra `deploy_cfg` keys (`ckpt_name`, ...), merged over the policy's `deploy.yml` |

## Observation and action

`{p}` is `""` for one arm and `left_` / `right_` for two.

| hiRobot field | XPolicyLab key | Notes |
|---|---|---|
| `joint_pos[start:stop]` | `state/{p}arm_joint_state` | one arm also gets `state/joint_state` |
| `gripper[name]` | `state/{p}ee_joint_state` | opening [0, 1] mapped to the model's range by `policy.gripper_range` |
| `ee_poses[name]` | `state/{p}ee_pose` | quaternion order converted |
| `images[robot cam]` | `vision/<model cam>/color` | |
| `instruction` | `instruction` | |

| XPolicyLab action | hiRobot chunk field | Notes |
|---|---|---|
| `{p}arm_joint_state` | those arms' columns of `joint_targets` | `action_type: joint`, `mode: position`. Joints outside `arms` (G1 legs, waist) hold the episode's first `joint_pos` |
| `{p}ee_pose` | `ee_targets[ee]` | `action_type: ee`, `mode: cartesian` |
| `{p}ee_joint_state` | `gripper[gripper]` | mapped to the opening by `gripper_range` when it is set |

Poses: XPolicyLab is `[x, y, z, qw, qx, qy, qz]`, hiRobot is
`[x, y, z, qx, qy, qz, qw]`; both directions are converted.

## Verified

`demo_policy`, started with `hivla launch` from hiRobot's environment (no XPolicyLab there):

- hiRobot's `unitree_g1_mujoco`, 100 steps
  with `hivla.robot.run_episode` over HTTP and over WebSocket (50 chunks and 50
  observes each; `act` 13.0 ms mean over HTTP, 7.8 ms over WebSocket). The final
  joint positions are within 8e-10 rad of sending the same targets to hiRobot
  directly. Legs and waist hold the episode's first pose.
- `ur5e_dual` (`xpolicylab_demo_ur5e_dual`): chunk shapes over HTTP and WebSocket,
  JSON and msgpack; a second session gets 409.

No policy with real weights has been served yet.

## Common pitfalls

- `runtime.cache: true` or `max_batch_size > 1` is refused for a stateful
  model at start.
- After a WebSocket connection drops, the next connection is a new session.
  Call `reset` again before acting.
- A policy that returns fewer steps than `action_horizon` raises; set
  `action_horizon` to what `get_action` returns.
- Hands with more than one joint (`ee_dim > 1` with a gripper) are not
  supported yet.
