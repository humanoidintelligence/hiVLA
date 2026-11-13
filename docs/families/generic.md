# generic: torch / ONNX / TensorRT checkpoints

English | [简体中文](./generic.zh-CN.md)

Family: `generic` (the default when `model.family` is not set). It serves a
checkpoint that is just a network: arrays in, arrays out. hiVLA does the
pre- and post-processing itself, from the config: renaming keys, normalizing
inputs and unnormalizing outputs with your `norm_stats`.

Use it for a model you exported yourself. A checkpoint that ships its own
processing (LeRobot, openpi, GR00T, ...) should use its named family, so the
results match the original release.

## Environment

hiVLA plus the extra for the backend:

```bash
pip install -e ".[server,client,torch]"      # or onnx / tensorrt
```

The `echo` backend needs no extra and no checkpoint: it returns zero chunks,
for trying the server and the client (`src/hivla/configs/echo_example.yaml`).

## Configuration

`src/hivla/configs/example.yaml` lists every option. The parts that matter for
`generic`:

```yaml
model:
  family: generic
  checkpoint: ${CKPT_DIR}/policy.onnx
  backend: auto             # auto | echo | torch | onnx | tensorrt
  action_horizon: 16

policy:
  action_dim: 7
  control_mode: position
  cameras: [head]
  norm_stats: ${CKPT_DIR}/norm_stats.json
  rename_inputs: {joint_pos: state}
  rename_outputs: {action: joint_targets}
```

`backend: auto` picks from the checkpoint: `.engine` / `.plan` is TensorRT,
`.onnx` is ONNX, no checkpoint is `echo`, anything else is torch. For a
directory, the file extensions inside it decide.

| `model.backend` | Loads | Device |
|---|---|---|
| `echo` | nothing; returns zeros `(action_horizon, action_dim)` | none |
| `torch` | a whole `nn.Module` / TorchScript saved with `torch.save` (`torch.load`) | `runtime.device`, `runtime.precision` |
| `onnx` | an ONNX Runtime session | `runtime.device` (`cuda` adds the CUDA provider) |
| `tensorrt` | a serialized TensorRT engine | CUDA |

`remote` (forward to another hiVLA server) cannot be configured from YAML,
because it needs an `endpoint`; build it in Python (see
[Backends](../api.md#backends)).

## What hiVLA does

```
observation -> ConcatState (state_layout) -> rename_inputs -> Normalize
            -> backend -> Unnormalize -> rename_outputs -> GripperToOpening -> chunk
```

`norm_stats` maps a key to its statistics: `mean` / `std` (z-score),
`q01` / `q99` (quantile, to [-1, 1]) or `min` / `max` (to [-1, 1]). The mode
is inferred from the fields present. Statistics wider than the data are sliced
to the data's width.

A torch module that returns a bare tensor gives `joint_targets`. Otherwise the
backend's output must contain `joint_targets` (or a key renamed to it) of
shape `(action_horizon, action_dim)`, or the other chunk fields (`ee_targets`,
`gripper`). The chunk is checked against hiRobot's `ActionChunk` rules before
it is sent.

## Common pitfalls

- The checkpoint's input names must match the keys after `rename_inputs`. Run
  `hivla inspect --config ...` first; it resolves the config and the backend
  without loading weights.
- Only `generic` applies `norm_stats`. A named family ignores it, because the
  model normalizes itself.
