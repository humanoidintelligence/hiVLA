# openpi (π₀ / π₀.₅ / π₀-FAST)

English | [简体中文](./openpi.zh-CN.md)

Family: `openpi`. It serves a Physical Intelligence checkpoint through
[openpi](https://github.com/Physical-Intelligence/openpi)'s own
`create_trained_policy` and `Policy.infer`, JAX or PyTorch. openpi owns
normalization, tokenization and the robot-specific input / output transforms
(`LiberoInputs`, `AlohaInputs`, ...) of the named train config. hiVLA does what
openpi's eval clients do before sending (`examples/libero/main.py`): it maps
hiRobot's fields onto openpi's keys, resizes the images, and splits the action
vector into chunk fields.

## Environment

openpi pins its own Python (3.11) and jax / torch versions in a `uv.lock`;
install it by its instructions, then hiVLA into the same environment:

```bash
cd openpi && uv sync && source .venv/bin/activate
pip install -e <hiVLA>"[server,client]"
```

Checkpoints are resolved through openpi's cache, so set `OPENPI_DATA_HOME`;
`src/hivla/configs/openpi_pi05_libero.yaml` reads the checkpoint from it. openpi
downloads `gs://openpi-assets/...` checkpoints into it on first use.

## Configuration

`src/hivla/configs/openpi_pi05_libero.yaml` (π₀.₅-LIBERO, JAX):

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

| `native_kwargs` key | Default | Meaning |
|---|---|---|
| `config` | required | openpi train config, e.g. `pi05_libero`, `pi0_aloha_sim` |
| `cameras` | required | `{robot camera: openpi image key}` |
| `state_key` | `observation/state` | where `joint_pos` goes |
| `image_size` | `[224, 224]` | `[h, w]` for openpi's `resize_with_pad` |
| `action_layout` | none | list of `[field, name, size]` slices; without it the whole vector is `joint_targets` |
| `task` | none | default prompt |
| `seed` | openpi's (0) | JAX sampling seed |
| `sample_kwargs` | none | passed to `sample_actions`, e.g. `{num_steps: 10}` |

`action_horizon` cannot exceed the train config's horizon; loading checks it.

## Observation and action

| Observation field | Goes to | Notes |
|---|---|---|
| `images[cam]` | `obs[cameras[cam]]` | resized with padding to `image_size`, uint8 |
| `joint_pos` | `obs[state_key]` | the model's state vector, flattened |
| `instruction` | `prompt` | `native_kwargs.task` otherwise |

`action_layout` cuts `actions (action_horizon, action_dim)` left to right.
`field` is `ee_targets` (kept `(T, size)`), `gripper` (`(T,)` per name) or
`joint_targets`. For LIBERO the 7-D action becomes `ee_targets["eef"]`
`(10, 6)` (delta position + delta axis-angle) and `gripper["gripper"]` `(10,)`,
with `mode: cartesian_delta`. ALOHA and DROID joint configs leave
`action_layout` out and get `joint_targets`.

For LIBERO the robot side follows openpi's eval: both images rotated 180°
before sending, and `joint_pos` holding the 8-D state (end-effector position 3
| axis-angle 3 | gripper qpos 2), not the arm joints.

## Verified

π₀.₅-LIBERO (JAX) on one H200, about 71 ms per request when warm, driven by
hiRobot's LIBERO driver over `/ws` msgpack, 500 episodes per suite:

| Suite | spatial | object | goal | 10 | mean |
|---|---|---|---|---|---|
| hiVLA | 99.0 | 98.0 | 98.2 | 93.0 | **97.05** |
| openpi reported | | | | | 96.85 |

## Common pitfalls

- `Policy.infer` takes one observation, so concurrent requests run one after
  another; `max_batch_size > 1` does not batch them. Run several servers behind
  `hivla route` instead.
- The LIBERO gripper comes back in [-1, 1] (-1 open, +1 closed), the old LIBERO
  driver's convention, not hiRobot's 0-closed / 1-open opening. A robot that
  expects the opening needs `policy.gripper_range: {gripper: [1, -1]}`.
