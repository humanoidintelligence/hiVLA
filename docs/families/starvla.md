# StarVLA

English | [简体中文](./starvla.zh-CN.md)

Family: `starvla`. It serves a [StarVLA](https://github.com/starVLA/starVLA)
`baseframework` checkpoint (QwenGR00T, QwenOFT, ...). StarVLA's
`predict_action` returns normalized actions and leaves state encoding, image
resizing and unnormalization to each benchmark's client
(`examples/<benchmark>/eval_files/model2*_interface.py`). hiVLA does that
client-side work on the server, so the robot sends plain `joint_pos`, `images`
and `instruction`. For RoboCasa GR-1 it reproduces StarVLA's
`model2robocasa_interface.py` exactly.

## Environment

A StarVLA environment by its instructions (the verified one: Python 3.10,
torch 2.6.0+cu124, transformers 4.57, flash-attn 2.7.4), plus hiVLA's server
dependencies. Pin `typing_extensions` so they do not upgrade the env's copy:

```bash
pip install -e <hiVLA>"[server,client]" "typing_extensions==4.15.0"
```

The backend also needs OpenCV (`cv2`), which StarVLA environments have.

## Configuration

`src/hivla/configs/starvla_qwengr00t_gr1.yaml` (Qwen3-VL-GR00T, RoboCasa GR-1, 90k steps):

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

`checkpoint` is the `.pt` file. Its `config.yaml` and
`dataset_statistics.json` sit two levels up, where StarVLA's `read_mode_config`
expects them.

| `native_kwargs` key | Default | Meaning |
|---|---|---|
| `base_vlm` | from `config.yaml` | path to the VLM; checkpoints store one relative to the training directory, so set it |
| `unnorm_key` | the only key | which entry of `dataset_statistics.json` to unnormalize with |
| `state_groups` | none (no state) | group sizes `joint_pos` is split into |
| `state_encoding` | `none` | `sincos`: each group becomes `[sin(g), cos(g)]` (29 -> 58) |
| `image_size` | `[224, 224]` | `[w, h]` for OpenCV `INTER_AREA` resizing |
| `bf16` | `true` | cast weights to bfloat16, as StarVLA's eval server does |
| `compile` | off | a `torch.compile` mode for the DiT action head |
| `task` | none | default instruction |

## Observation and action

| Observation field | Goes to | Notes |
|---|---|---|
| `images[cam]` | `image` list | in `policy.cameras` order, resized to `image_size` |
| `joint_pos` | `state (1, state_dim)` | split into `state_groups`, then encoded |
| `instruction` | `lang` | required, or `native_kwargs.task` |

The normalized actions are clipped to [-1, 1] and mapped back with the
`unnorm_key` statistics' min / max (masked dimensions pass through), returned
as `joint_targets` `(action_horizon, action_dim)`. For gr1 these are absolute
radians in the same 29-joint order as the state.

Loading checks that the statistics' action width equals `policy.action_dim`
and that `action_horizon` does not exceed the checkpoint's.

## Verified

QwenGR00T on RoboCasa GR-1 `PnPBottleToCabinetClose`, through hiRobot's
RoboCasa driver: 37 of 50 episodes succeeded (0.74). StarVLA's own client,
four runs of 50 episodes, scored 0.58–0.74.

## Common pitfalls

- A checkpoint with several dataset statistics needs `unnorm_key`; the error
  lists the keys.
- Without `base_vlm`, the relative VLM path stored in the checkpoint resolves
  against the server's working directory and usually fails.
