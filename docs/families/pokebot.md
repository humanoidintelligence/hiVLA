# Pokebot RoboAPI

English | [简体中文](./pokebot.zh-CN.md)

Family: `pokebot`. Pokebot's cloud model (`poke-aurora`) behind hiVLA's
interface, through [RoboAPI](https://developers.pokebot.com/docs/roboapi/latest),
an OpenAI-compatible HTTP API at `https://api.pokebot.com/v1`. Nothing runs
locally: no weights, no GPU. Each request sends one wrist image, the
end-effector pose and the gripper opening and gets back a block of absolute
end-effector poses.

RoboAPI ties the requests of one task into a *session*. Every request is a
*turn*, and the next turn must carry the previous turn's reply and an
execution receipt (how many steps of the block ran). So this family is
**stateful**: the server serves one hiVLA session at a time, `reset` closes
the RoboAPI session, and the first request after it opens a new one.

## Environment

The family needs `httpx` and Pillow, nothing else:

```bash
pip install -e ".[server,client,vision]"
export POKEBOT_API_KEY=sk-...    # Developer Console: https://platform.pokebot.com
```

The key is read from the environment variable named by
`native_kwargs.api_key_env` (default `POKEBOT_API_KEY`). It is never written to
the config, the log or `/metadata`.

## Configuration

`src/hivla/configs/pokebot_aurora.yaml`:

```yaml
model:
  family: pokebot
  action_horizon: 40       # poke-aurora-preview returns 40 rows per turn
  n_action_steps: 40       # rows the client runs before asking again

native_kwargs:
  base_url: https://api.pokebot.com/v1
  model: poke-aurora-preview     # an id from GET /models
  api_key_env: POKEBOT_API_KEY
  ee: eef                  # hiRobot end-effector name (pose in, ee_targets out)
  camera: wrist            # robot camera sent as the image
  image_format: jpeg       # or png
  max_pixels: 230400       # larger images are scaled down, aspect kept

runtime:
  cache: false
  warmup: false            # a warm-up call would cost balance

policy:
  control_mode: cartesian
```

| `native_kwargs` key | Default | Meaning |
|---|---|---|
| `base_url` | `https://api.pokebot.com/v1` | Keep the `/v1` suffix |
| `model` | `poke-aurora-preview` | An `id` from `GET /models`; checked at load |
| `api_key_env` | `POKEBOT_API_KEY` | Environment variable with the key |
| `ee` | required | End-effector name in `ee_poses` and `ee_targets` |
| `gripper` | `ee` | Gripper name in `gripper` |
| `camera` | required | Camera name in `images` |
| `image_format` | `jpeg` | `jpeg` or `png` |
| `jpeg_quality` | 90 | |
| `max_pixels` | 640 × 360 | Never above the schema's `max_pixels` |
| `instruction` | none | Used when the observation has none |
| `timeout` | 60 | Seconds per request |

At load the backend calls `GET /models` (the model must be listed) and
`GET /models/{model}/schema` (the state must take 8 numbers and the output
must have 8 columns).

## I/O

| hiRobot | RoboAPI |
|---|---|
| `ee_poses[ee]` `[x, y, z, qx, qy, qz, qw]` + `gripper[gripper]` (opening, 0 closed, 1 open) | `state` `[x, y, z, qx, qy, qz, qw, gripper]` |
| `images[camera]` HxWx3 RGB | one `image_url` data URL |
| `instruction` | `instruction` (non-empty, at most 4096 characters) |
| `ee_targets[ee]` (N, 7), `gripper[gripper]` (N,), `mode: cartesian` | `execute_actions` tool call, `output` (N, 8) |

The layouts match, so nothing is reordered: hiRobot's poses are already
`xyzw`. The returned quaternions are renormalized and the gripper is clipped
to [0, 1]. A block longer than `action_horizon` is cut to it.

**Frames.** RoboAPI expects the pose of the flange centre (not the fingertip
or the TCP) in its own calibrated task frame, and returns targets in that
frame. The robot's driver (or a hiRobot body config) must report and accept
poses in that frame; see [Before a real robot](#before-a-real-robot). At the
calibrated Home the pose is `[-0.13, -0.46, 0.49, 0, 1, 0, 0]`.

**Image.** One wrist camera, the full frame with its fisheye distortion: do
not crop, mirror, rotate or undistort it. Both fingers should be in view, with
a thin strip of the gripper palm along the bottom edge. Capture the image
together with the pose.

**Rate.** Each output row is an absolute target that RoboAPI's own runtime
dispatches at 30 Hz. Run the block at the same rate.

## Before a real robot

hiVLA only relays requests; it does not calibrate the robot. The steps below
follow RoboAPI's
[hardware](https://developers.pokebot.com/docs/roboapi/latest/hardware.md),
[UR setup](https://developers.pokebot.com/docs/roboapi/latest/ur-mock.md) and
[calibration](https://developers.pokebot.com/docs/roboapi/latest/calibration.md)
pages; check them for the current version.

1. **Hardware.** RoboAPI lists UR5 (CB3), UR5e and UR7e, and Franka FR3; a
   Robotiq 2F-85 with a parallel locking kit; and a wrist camera on its
   standard mount: an Insta360 Ace Pro 2 with the Ultra Wide Lens, or a DJI
   Osmo Action 6 with the FOV Boost Lens, as a USB camera at MJPG 1280 × 720,
   30 FPS. The fisheye view is part of the model input, so another camera
   (a RealSense, a plain webcam) is outside what the model was trained on.
2. **Tool frame.** On the UR pendant, set the TCP to the flange centre with
   zero position offset, and set the real payload. A TCP offset changes the
   pose every other program on that arm reads, so note the old one first.
3. **RoboAPI runtime.** Install it on the computer wired to the arm
   (Linux x86-64, Python 3.10–3.12; the user needs the `video` and `dialout`
   groups):

   ```bash
   uv pip install 'roboapi[ur-cb3,gripper]==0.4.5'   # UR5 CB3
   uv pip install 'roboapi[gripper]==0.4.5'          # UR5e / UR7e
   roboapi init     # arm generation, controller IP, gripper, camera, key
   roboapi http     # local console on http://127.0.0.1:8080/
   ```

   Only one program can control the arm at a time: stop hiRobot (or any other
   controller) while RoboAPI holds it.
4. **Home + A/B/C calibration** (console: Settings → Devices → Coordinate
   calibration). Guide the arm by hand and capture each pose once it settles,
   with the same tool orientation, gripper opening and contact point
   throughout:
   * **Home:** gripper vertical, fingers down, 30–50 cm above a levelled
     surface, with a clear path back.
   * **A:** the contact point touching the surface.
   * **B:** along the surface toward camera left; A → B is task +X.
   * **C:** another surface point off the line AB. The arm then returns to
     Home on its own and the gripper may move; stay at the stop button.

   Save, enable control, and run the direction check (5 cm and 30° forward,
   left and up from Home). Task +Z is the surface normal toward Home. The
   result is stored on that computer, by default in
   `~/.config/roboapi/calibration.json`, and must be redone when the tool, the
   camera mount, Home or the task axes change.
5. **Frame on the hiRobot side.** Report `ee_poses[ee]` as the flange pose in
   that task frame, and map `ee_targets[ee]` back to the robot before sending
   them. A raw base-frame TCP pose gives targets in the wrong place.

## Receipts

The turn after a block reports how much of it ran:

* `extra["executed_steps"]` in the observation, when the client sends it;
* otherwise `min(n_action_steps, rows)`.

`status` is `completed` when every row of the block ran, else `interrupted`.
On `reset` (or server shutdown) the session is closed with the last block's
receipt, counted the same way.

## Failures

* **No automatic retry.** RoboAPI warns that a retried request must never make
  the robot run the same block twice.
* **Timeout or dropped connection.** The backend asks the server for the turn
  (`GET /sessions/{id}/turns/{turn}`) and uses the stored response if the
  server finished it. Otherwise the request fails.
* **Any other failure** (HTTP error, a reply for another turn, a malformed
  `output`) closes the RoboAPI session and fails the request. The next request
  opens a new session. Take a fresh observation before asking again.

| HTTP | Meaning |
|---|---|
| 400 | Bad JSON, data URL or state |
| 401 | Bad API key |
| 403 | No access to the model, or no account |
| 404 | Wrong `base_url`, or a model id not in `GET /models` |
| 413 | Request too large: lower `max_pixels` or use JPEG |
| 429 | Rate, concurrency or balance limit |
| 5xx | Service failure |

The error message carries RoboAPI's `X-Request-ID`; quote it to Pokebot
support (never the key or the image).

## Verified

2026-11-14 against `https://api.pokebot.com/v1` with `poke-aurora-preview`
(served by `poke-aurora-gauss-1030`):

* A turn from a 640 × 360 image and the Home pose returned a (40, 8) block in
  1.7 s on the server. The first row equals the state sent, so the targets
  are absolute poses.
* Session create, turn lookup and close with a receipt work as described
  above.

Not verified: a real robot, including the calibration and frame steps in
[Before a real robot](#before-a-real-robot).

## Pitfalls

* **Not behind `hivla route`.** The router does not forward `/reset`, so a
  stateful model behind it would mix episodes.
* **Every request costs balance.** Warm-up is off, and `hivla inspect` does
  not call the API.
* **Pose in the wrong frame.** Raw controller poses (or the Home constants at
  other poses) give targets in the wrong place. Send the calibrated pose.
* **TCP offset left on.** A gripper TCP on the pendant makes the reported pose
  the fingertip, not the flange. Set it to zero before calibrating.