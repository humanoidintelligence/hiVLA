# RLinf (π₀ / π₀.₅ in PyTorch)

English | [简体中文](./rlinf.zh-CN.md)

Family: `rlinf`. It serves a checkpoint from [RLinf](https://github.com/RLinf/RLinf),
which reimplements openpi's π₀ / π₀.₅ in PyTorch (numerically aligned with the
JAX reference) and publishes its SFT and RL-finetuned models in that form:
`model.safetensors` plus `<asset_id>/norm_stats.json`, for example
`RLinf/RLinf-Pi05-LIBERO-SFT`. The backend builds the model with
`rlinf.models.embodiment.openpi.get_model` in eval mode, which installs
openpi's input / output transforms for the named config, so normalization, the
224 resize and tokenization happen inside the model, as in RLinf's own eval.

## Environment

RLinf with its openpi fork (Python 3.11; the verified set is rlinf-openpi
0.1.1, torch 2.7.1, jax[cuda12] 0.5.3, rlinf-transformer-openpi 4.53.2),
installed as RLinf's `requirements/install.sh` does for the openpi model, then
hiVLA in the same environment. Keep it separate from an `openpi` environment:
the two use different openpi packages.

```bash
pip install -e <RLinf> -e <hiVLA>"[server,client]"
```

Two environment variables are required, and loading fails if either is unset:
`RLINF_CKPT` (the checkpoint directory) and `OPENPI_DATA_HOME` (where the
PaliGemma tokenizer is cached, shared with openpi).

## Configuration

`src/hivla/configs/rlinf_pi05_libero.yaml` (RLinf-Pi05-LIBERO-SFT, one model for all four
LIBERO suites):

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

| `native_kwargs` key | Default | Meaning |
|---|---|---|
| `config` | required | openpi train config, e.g. `pi05_libero` |
| `cameras` | none | `{robot camera: RLinf env-obs key}` |
| `action_layout` | none | `[field, name, size]` slices, as in [openpi](openpi.md#observation-and-action) |
| `pi05` | `true` | π₀.₅; `false` for π₀ |
| `num_steps` | `5` | denoise steps |
| `precision` | the checkpoint's | parameter dtype, e.g. `bf16` |
| `seed` | none | seed of the CUDA generator for the initial noise |
| `openpi` | none | extra `cfg.openpi` fields, e.g. `{torch_compile: false}` |

## Observation and action

| Observation field | Goes to | Notes |
|---|---|---|
| `images[cam]` | `env_obs[cameras[cam]]` | uint8 as captured; the model resizes to 224 |
| `joint_pos` | `states` | LIBERO: end-effector position 3 \| axis-angle 3 \| gripper qpos 2 |
| `instruction` | `task_descriptions` | |

An observation that already carries `main_images` (RLinf's own form) is passed
through as is. The actions come back unnormalized, `(action_horizon,
action_dim)`, and leave by `action_layout`: for LIBERO `ee_targets["eef"]`
`(5, 6)` and `gripper["gripper"]` `(5,)` in [-1, 1].

Concurrent requests are stacked into one forward pass when
`runtime.max_batch_size > 1`.

## Verified

RLinf-Pi05-LIBERO-SFT ran the LIBERO spatial suite through hiRobot's LIBERO
driver, with five simulation workers against one server batching five
requests (738 episodes per hour on an H200). A success rate from that run is
not recorded here yet.

## Common pitfalls

- The LIBERO gripper is in [-1, 1] (-1 open), the old LIBERO driver's
  convention. A robot that expects hiRobot's opening needs
  `policy.gripper_range: {gripper: [1, -1]}`.
- With batching on, RLinf ran about twice as slow per episode as openpi on the
  same suite. The cause has not been measured yet.
