# GR00T (N1.6 / N1.7)

English | [简体中文](./gr00t.zh-CN.md)

Family: `gr00t`. It serves an NVIDIA
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.6 / N1.7 checkpoint
through its own `Gr00tPolicy`, which owns the whole pipeline: the Eagle VLM
processor, per-embodiment state and action normalization, and decoding
relative actions into absolute ones. hiVLA only translates between hiRobot's
flat observation and GR00T's grouped, batched form.

## Environment

Set up Isaac-GR00T by its instructions (N1.6: Python 3.12, torch 2.7.1+cu128),
then add the Eagle backbone's `flash-attn` and hiVLA:

```bash
conda activate gr00tn1d6                         # your GR00T environment
pip install flash-attn --no-build-isolation
pip install -e <hiVLA>"[server,client]"
```

The checkpoint's VLM backbone is gated on Hugging Face: request access to
[`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B), log in
with `hf auth login`, then download it under `${HIVLA_MODELS}`:

```bash
hf download nvidia/GR00T-N1.6-3B --local-dir $HIVLA_MODELS/Isaac-GR00T-n1.6/checkpoints/GR00T-N1.6-3B
```

The README's [worked example](../../README.md#worked-example-gr00t-n16-on-the-fourier-gr-1)
goes through these steps from a fresh clone.

## Configuration

`src/hivla/configs/gr00t_n1d6_gr1.yaml` (GR00T N1.6-3B, Fourier GR-1, embodiment `gr1`):

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

| `native_kwargs` key | Default | Meaning |
|---|---|---|
| `embodiment_tag` | required | GR00T embodiment, e.g. `gr1` |
| `cameras` | identity | `{robot camera: GR00T video key}`; optional when the model has one video key and the robot sends one image |
| `task` | none | default instruction |
| `strict` | `true` | GR00T's own input / output validation |
| `compile` | off | a `torch.compile` mode for the DiT action head, e.g. `max-autotune`; the Eagle backbone stays eager |

## Observation and action

| Observation field | Goes to | Notes |
|---|---|---|
| `joint_pos` | GR00T's state groups | split in the order of the checkpoint's modality config. gr1: `left_arm 7, right_arm 7, left_hand 6, right_hand 6, waist 3` = 29 |
| `state` (dict) | the same groups | instead of `joint_pos`, named groups |
| `images[cam]` | `video[cameras[cam]]` | `(H, W, 3)` uint8; GR00T resizes |
| `instruction` | the language key | required, or `native_kwargs.task` |

The action groups are joined in action-key order into `joint_targets`
`(action_horizon, dof)`, absolute, in physical units (radians for gr1). For gr1
it is the same 29-joint order as the state.

Concurrent requests are stacked into one forward pass when
`runtime.max_batch_size > 1`.

## Verified

GR00T N1.6-3B, `gr1`, on one H200, through hiRobot's RoboCasa driver:

- RoboCasa GR-1 `PnPBottleToCabinetClose`: 5 of 10 episodes succeeded.
- About 100–130 ms per request when warm; `compile: max-autotune` brings the
  forward pass from 211 to 62 ms in the serving benchmark (see
  [Performance](../../README.md#performance)).

`scripts/smoke_gr00t_n1d6.py --config gr00t_n1d6_gr1` loads the
model in-process (or behind a local server with `--http` / `--ws`), sends
synthetic observations and checks every chunk's shape.

## Common pitfalls

- `joint_pos` must have exactly the total width of the state groups (29 for
  gr1); the error lists the groups GR00T expects.
- Only single-step observations are served. An embodiment whose modality
  config uses a history window (more than one video, state or language step)
  is refused at load.
- `action_horizon` cannot exceed the embodiment's action horizon.
