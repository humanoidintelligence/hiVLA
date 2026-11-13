# Changelog

User-facing changes in each hiVLA release, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Before 0.1.0, any
release may change the API or the wire format incompatibly; such changes are
listed under "Removed" or "Changed" and marked **breaking**.

## [0.0.1] - 2026-11-14

First public release: a serving layer that runs a VLA checkpoint behind one
network interface. A robot-side caller sends an observation and gets back an
action chunk; everything model-specific runs on the server.

### Added

- **Model families**: `lerobot` (LeRobot v0.6.2 single-observation policies
  such as ACT), `openpi` (π₀, π₀.₅, π₀-FAST), `gr00t` (GR00T N1.6 / N1.7),
  `starvla` (`baseframework` checkpoints such as QwenGR00T and QwenOFT),
  `xpolicylab` (any XPolicyLab policy, stateful) and `rlinf` (RLinf's PyTorch
  π₀ / π₀.₅). Each runs the model through its own code and pre- /
  post-processing. A `generic` family serves plain torch / ONNX / TensorRT
  checkpoints and normalizes with the config's `norm_stats`. A new family is
  one `FamilyAdapter`.
- **Built-in configs**, usable by name (`hivla configs` lists them):
  `lerobot_act_aloha`, `openpi_pi05_libero`, `gr00t_n1d6_gr1`,
  `starvla_qwengr00t_gr1`, `xpolicylab_demo_g1`, `xpolicylab_demo_ur5e_dual`,
  `rlinf_pi05_libero`, and `example` / `echo_example` / `router_example`.
  Checkpoint paths come from environment variables (`${HIVLA_MODELS}`,
  `${OPENPI_DATA_HOME}`, `${RLINF_CKPT}`); an unset variable without a
  default is an error, not a silent wrong path.
- **Server**: `hivla serve` exposes `POST /act`, `POST /reset`,
  `POST /observe`, `GET /metadata` and `GET /health` over HTTP, and the same
  contract over WebSocket (`/ws`). Bodies are JSON (arrays base64-encoded) or
  msgpack. Object-dtype arrays are rejected, so a request cannot carry a
  pickle. An optional `server.api_key` protects the HTTP endpoints (`/ws` does
  not check it yet).
- **hiRobot fields on the wire**: observations and replies use hiRobot's
  `Observation` and `ActionChunk` field names (`joint_targets`, `mode`, `kp`,
  `kd`, `ee_targets`, `gripper`, `parts`, ...). `policy.state_layout` builds a
  model's state vector from those fields, and `policy.gripper_range` maps its
  gripper output to hiRobot's opening (0 closed, 1 open). `/metadata` reports
  `protocol_version` (3), `n_action_steps` and what the model needs from the
  robot.
- **Sessions for stateful models**: a model that keeps history between
  requests is reset at the start of each episode and can read every executed
  step (`observe: every_step` in `/metadata`). A stateful model serves one
  session at a time; a second session gets `409`.
- **Clients**: `hivla.client.VlaClient` (HTTP) and `VlaWsClient` (WebSocket)
  need only numpy and `hivla[client]`, none of the model's packages.
  `hivla.robot.run_episode` runs a served model on a hiRobot `Robot`.
- **Faster serving**: dynamic batching of concurrent requests
  (`max_batch_size`), a cache that replays the last chunk for a repeated
  observation, `torch.compile`, and `AsyncInferenceEngine`, which predicts
  the next chunk while the robot executes the current one.
- **Many GPUs behind one address**: `hivla route` load-balances `POST /act`
  across several servers (round-robin, least-outstanding or consistent-hash).
  A backend that cannot be reached gives `502` at once.
- **One environment per model**: `hivla launch` starts the server with the
  Python of the environment named in the config, from any environment.
  `hivla inspect` shows how a config resolves without loading weights.
- Verified: π₀.₅ on LIBERO, 97.05% mean over 4 suites (openpi reports
  96.85%); GR00T N1.6 on RoboCasa GR-1, 5 / 10 episodes; StarVLA QwenGR00T
  on RoboCasa GR-1, 37 / 50; RLinf Pi05-LIBERO-SFT on LIBERO spatial; LeRobot
  ACT ALOHA chunk shapes over HTTP and WebSocket; XPolicyLab `demo_policy` on
  hiRobot's MuJoCo G1. No XPolicyLab policy with real weights has been served
  yet.

[0.0.1]: https://github.com/humanoidintelligence/hiVLA/releases/tag/v0.0.1
