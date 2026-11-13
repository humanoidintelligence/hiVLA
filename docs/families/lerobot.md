# LeRobot

English | [简体中文](./lerobot.zh-CN.md)

Family: `lerobot`. It serves a [LeRobot](https://github.com/huggingface/lerobot)
**v0.6.2** policy through LeRobot's own chain, the one its
`SyncInferenceEngine` runs:

```
prepare_observation_for_inference -> preprocessor -> policy.predict_action_chunk -> postprocessor
```

The preprocessor and postprocessor are the `policy_preprocessor.json` /
`policy_postprocessor.json` pipelines saved next to the weights, so
normalization is the checkpoint's own. hiVLA only renames keys and cuts the
chunk to `action_horizon`.

## Supported checkpoints

- **Processor-format checkpoints only** (LeRobot >= 0.4: the two processor
  files next to `model.safetensors`). The original `lerobot/act_*` and
  `lerobot/diffusion_*` hub repos are legacy checkpoints, with normalization
  buffers inside the weights; convert one once in the LeRobot env:

  ```bash
  python <lerobot>/src/lerobot/processor/migrate_policy_normalization.py \
      --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
      --output-dir checkpoints/act_aloha_sim_transfer_cube_human
  ```

  Loading a legacy checkpoint raises LeRobot's `ProcessorMigrationError`,
  whose message carries this command.
- **Single-observation policies only** (`n_obs_steps: 1`, such as ACT). A
  policy with an observation history (`n_obs_steps > 1`) is refused at load,
  because `predict_action_chunk` on a fresh request has no history.

## Environment

```bash
conda create -n lerobot python=3.12 -y && conda activate lerobot
pip install -e <lerobot checkout at v0.6.2>
pip install -e ".[server,client]"              # hiVLA, from its checkout
```

## Configuration

`src/hivla/configs/lerobot_act_aloha.yaml` (ACT, bimanual ALOHA sim, transfer-cube):

```yaml
model:
  family: lerobot
  checkpoint: checkpoints/act_aloha_sim_transfer_cube_human   # local dir or hub repo id
  action_horizon: 100      # <= the policy's chunk_size; cut server-side

runtime:
  device: cuda

policy:
  action_dim: 14
  control_mode: position
  cameras: [top]

native_kwargs:             # all optional
  task: "transfer the cube"   # default instruction when a request has none
  robot_type: aloha
  revision: main              # hub revision
```

## Observation and action

The observation keys are LeRobot's feature names, as in the checkpoint's
`config.json`:

| Request key | Example (ACT ALOHA) | Notes |
|---|---|---|
| `observation.state` | `(14,)` float32 | float64 is cast to float32 |
| `observation.images.<cam>` | `(480, 640, 3)` uint8 | one key per camera |
| `task` | `"transfer the cube"` | optional; `native_kwargs.task` otherwise |

A caller that sends hiRobot's names (`joint_pos`, `images`, `instruction`, as
`hivla.robot` and hiAgent do) needs the top-level ones renamed:
`policy.rename_inputs: {joint_pos: observation.state, instruction: task}`.
`rename_inputs` only renames top-level keys, so this family does not yet map
the nested `images: {cam: ...}` dict to `observation.images.<cam>`; for now the
caller sends the flat keys.

The reply is `joint_targets` of shape `(action_horizon, action_dim)`, in the
units the policy was trained on.

## Verified

`scripts/smoke_lerobot_act.py` loads `src/hivla/configs/lerobot_act_aloha.yaml` on a GPU,
sends synthetic ALOHA observations and checks every chunk is
`(action_horizon, 14)`, in-process, over HTTP (`--http`) and over WebSocket
(`--ws`), with JSON or msgpack (`--wire`). No closed-loop success rate has been
measured with hiVLA yet.

## Common pitfalls

- `action_horizon` larger than the policy's `chunk_size` raises at the first
  request (`predicts N steps < action_horizon`).
- LeRobot's `select_action` (one step from an internal queue) is not used;
  hiVLA always returns the whole chunk and leaves stepping through it to the
  caller.
