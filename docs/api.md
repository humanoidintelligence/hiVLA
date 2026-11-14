# hiVLA API reference

English | [简体中文](./api.zh-CN.md)

This page lists the interfaces hiVLA exposes: the HTTP / WebSocket endpoints a
client calls, the wire format, the serving config, the CLI, and the Python API
for building an engine and adding a model family or backend. Names under
[Top-level package](#top-level-package) are importable as `from hivla import ...`.

hiVLA serves `observation → action chunk`. Preprocessing (rename, normalize,
pad) and action decoding run on the server, so a client sends a semi-raw
observation and gets back a decoded chunk. `hivla.client` has ready-made
clients (see [Clients](#clients-hivlaclient)); hiAgent uses them to run a
served model on a robot.

## Contents

- [Top-level package](#top-level-package)
- [Quick start](#quick-start)
- [Network API](#network-api)
- [Wire format](#wire-format)
- [Clients](#clients-hivlaclient)
- [Router](#router)
- [Serving config](#serving-config)
- [CLI](#cli)
- [ModelSpec](#modelspec)
- [Building an engine](#building-an-engine)
- [InferenceEngine and AsyncInferenceEngine](#inferenceengine-and-asyncinferenceengine)
- [VlaPolicy](#vlapolicy)
- [Transforms](#transforms)
- [Model families](#model-families)
- [Backends](#backends)
- [Embedding the server](#embedding-the-server)

## Top-level package

| Name | Kind | Module | Purpose |
|---|---|---|---|
| `ModelSpec` | pydantic model | `hivla.core.spec` | Everything needed to build and serve one model |
| `load_config` | function | `hivla` | Read a YAML config, expanding `~` and environment variables |
| `build_engine`, `build_engine_from_spec` | functions | `hivla.core.build` | Config / spec → `InferenceEngine` |
| `InferenceEngine`, `AsyncInferenceEngine`, `ObservationCache` | classes | `hivla.serving.engine` | Serving lifecycle, caching, batching |
| `VlaPolicy` | class | `hivla.core.policy` | Input transforms → backend → output transforms |
| `FamilyAdapter`, `get_adapter`, `available_families` | class / functions | `hivla.families` | Per-model-family extension point |
| `ModelBackend`, `EchoBackend`, `build_backend`, `infer_backend` | class / functions | `hivla.runtimes` | Tensor-in / tensor-out model runners |
| `__version__` | str | `hivla` | Package version |

Other public modules: `hivla.client`, `hivla.protocol`, `hivla.core.transforms`,
`hivla.serving.server`, `hivla.serving.router`, `hivla.serving.router`. The package imports names on
first use, so `import hivla.client` loads only numpy and `hivla.protocol`
(no pydantic, backends or engine).

## Quick start

```bash
pip install -e '.[server,client]'
hivla inspect --config echo_example       # resolve the config, load no weights
hivla serve   --config echo_example       # HTTP + WebSocket on :8000
```

`src/hivla/configs/echo_example.yaml` uses the echo backend, which needs no checkpoint and
returns a zero chunk of shape `(action_horizon, action_dim)`. Any config without
`model.checkpoint` (or with `model.backend: echo`) does the same.
`src/hivla/configs/example.yaml` lists every option; it names a checkpoint, so `auto`
picks `torch` for it.

From Python, with [`hivla.client`](#clients-hivlaclient):

```python
from hivla.client import VlaWsClient
client = VlaWsClient("http://localhost:8000")
```

To drive a robot with it, use hiAgent: `hiagent run --policy vla --config ...`
with `vla.endpoint` pointing at this server.

## Network API

`hivla serve` mounts one engine behind every endpoint below. Each request is
answered by `InferenceEngine.infer`, which runs off the event loop.

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | no | `{"status": "ok"}` |
| `GET` | `/metadata` | no | The model's metadata (see below) |
| `POST` | `/act` | yes, if `api_key` is set | One encoded observation in, one encoded action chunk out |
| `POST` | `/reset` | yes, if `api_key` is set | Start an episode: reset the model, drop cached chunks, bind a stateful model to the session |
| `POST` | `/observe` | yes, if `api_key` is set | Hand one encoded observation to a stateful model; no chunk back |
| `WS` | `/ws` | no | `act` / `observe` / `reset` / `metadata` over one long-lived connection |

**Authentication.** With `server.api_key` set, `/act`, `/reset` and `/observe`
require the header `Authorization: Api-Key <key>` and answer `401` otherwise. `/ws` does not check
the key yet, so do not expose the port beyond a trusted network.

**Metadata.** Built from the spec:

```json
{"protocol_version": 3, "model": "example-vla", "family": "generic",
 "action_horizon": 16, "n_action_steps": 16, "action_dim": 7,
 "control_mode": "position", "cameras": ["head", "left_wrist", "right_wrist"],
 "state_layout": null, "joint_names": null, "gripper": "raw",
 "stateful": false, "observe": "chunk"}
```

`state_layout`, `joint_names` and `gripper` say what the model needs from the
robot (see the wire format below), so a client can check before it runs.

`n_action_steps` is how many steps of each chunk the model expects a client to
execute before asking again. It is `model.n_action_steps` from the config, or
`action_horizon` when that is not set.

`stateful` says whether the model keeps state between requests. `observe` is
`chunk` when the model reads only the observation sent with `act`, and
`every_step` when it also wants one `observe` per executed step of a chunk.

**Sessions.** Over HTTP a session is the `X-Hivla-Session` header (each
`hivla.client` picks a random id); over `/ws` each connection is one session.
A stateless model ignores sessions. A stateful model needs a `reset` before its
first request, serves the session of the last `reset`, and refuses requests
from any other session (`409` over HTTP, an `error` reply over `/ws`). A new
`reset` hands it to the new session. A stateful model runs without the
observation cache and without batching (`runtime.cache: false`,
`max_batch_size: 1`).

### `POST /act`

- Body: an encoded observation (see [Wire format](#wire-format)).
- `Content-Type: application/json` (default): JSON body, arrays base64-encoded,
  JSON reply.
- `Content-Type: application/msgpack`: msgpack body with raw array bytes; the
  reply is msgpack too.
- `?timings=1` adds `server_timings_ms: {decode, infer, encode}` to the reply.
- Status codes: `200` with the chunk, `400` for a malformed or unsafe payload
  (for example an `object`-dtype array), `401` for a bad API key, `409` when a
  stateful model belongs to another session or has not been reset.

### `POST /reset` and `POST /observe`

- `/reset` takes no body and answers `{"ok": true}`.
- `/observe` takes an encoded observation, JSON or msgpack as for `/act`, and
  answers `{"ok": true}`, or `409` as for `/act`.
- Both read the session from `X-Hivla-Session`.

### `WS /ws`

1. On connect the server sends a JSON text frame
   `{"type": "metadata", "metadata": {...}}`.
2. The client sends `{"type": "act", "observation": {...}}`. It may add
   `"timings": true`.
3. The server replies `{"type": "action", "chunk": {...}}`, or
   `{"type": "error", "detail": "..."}` for a bad message or a session error.
   The connection stays open.
4. `{"type": "reset"}` starts an episode and is answered `{"type": "reset"}`;
   `{"type": "observe", "observation": {...}}` is answered `{"type": "observed"}`.

`{"type": "metadata"}` may also be sent at any time. Text frames are JSON;
binary frames are msgpack and get msgpack replies. The server refuses
permessage-deflate because camera frames barely compress.

## Wire format

`hivla.protocol`. The fields follow hiRobot's `Observation` and `ActionChunk`
(units, quaternion order `[x, y, z, w]`, gripper opening, control modes), so a
reply can be turned into a `hirobot.ActionChunk` field for field. hiVLA does not
import hiRobot; `tests/test_hirobot_contract.py` checks the two agree when
hiRobot is installed. `/metadata` reports `protocol_version` (currently 3).

**Arrays** are encoded as:

```json
{"__ndarray__": "<base64>", "dtype": "float32", "shape": [16, 7]}
```

On the msgpack wire, `__ndarray__` holds raw bytes instead of base64. Only
bool / int / uint / float dtypes are accepted; `object` dtype is refused in both
directions, and pickle is never used.

**Observation** (request). Any keys are accepted. Encoded arrays are decoded at
any depth; scalars and strings pass through. A client sends the populated
subset of hiRobot's `Observation`:

| Key | Value |
|---|---|
| `joint_pos`, `joint_vel`, `joint_torque` | array `(N,)` |
| `base_orientation` | array `(4,)`, `[x, y, z, w]` |
| `base_angular_velocity`, `base_linear_acceleration`, `base_position`, `base_linear_velocity` | array `(3,)` |
| `ee_poses` | `{name: array (7,)}`, `[x, y, z, qx, qy, qz, qw]` |
| `gripper` | `{name: float}`, opening in [0, 1] (0 closed, 1 open) |
| `images` | `{camera: array HxWx3 uint8}` |
| `depth` | `{camera: array HxW}` |
| `image_timestamps` | `{camera: float}` |
| `instruction` | string |
| `extra` | `{key: array, scalar or nested dict}` |

What the model reads depends on its family, `rename_inputs` and
`state_layout`. `state_layout` (config `policy` section) concatenates
structured fields into `joint_pos` before the family sees it, for example
`[[ee_poses, eef], [gripper, eef]]`; a missing field is an error.

**Action chunk** (reply). The keys are hiRobot's `ActionChunk` fields; only
the ones that are set are sent:

| Key | Value |
|---|---|
| `mode` | string: `position`, `velocity`, `torque`, `pd`, `cartesian`, `cartesian_delta`, `base_velocity`, `motion_latent`, `base_pose` |
| `joint_targets` | array `(horizon, n)` |
| `kp`, `kd` | array `(n,)`, shared by every step |
| `joint_names` | list of `n` names, only in the non-joint modes (targets for a subset of joints) |
| `ee_targets` | `{name: array (horizon, pose_dim)}` |
| `gripper` | `{name: array (horizon,)}` |
| `base_velocity`, `base_target` | array `(horizon, 3)` |
| `latent` | array `(horizon, D)` |
| `extra` | `{key: array, scalar or nested dict}` |
| `parts` | `{part: chunk}`, one chunk per part of a robot assembly, may nest; a chunk with `parts` carries no other command field |
| `server_timings_ms` | only when timings were requested |

The server checks the chunk before it sends it: a known `mode`, per-step
arrays with the same leading horizon (equal to `action_horizon` when set),
`joint_names` rules, and the `parts` rules at every depth. A chunk that fails
raises on the server. With `gripper_range: {name: [closed, open]}` in the
config, gripper commands are mapped from the model's range to an opening in
[0, 1] and `/metadata` reports `gripper: opening` (otherwise `raw`).

| Function | Description |
|---|---|
| `encode_array(arr, binary=False)` / `decode_array(obj)` | One array |
| `encode_observation(obs, binary=False)` / `decode_observation(payload)` | Observation dict |
| `encode_action_chunk(chunk, binary=False)` / `decode_action_chunk(payload)` | Action-chunk dict |
| `pack(payload)` / `unpack(data)` | msgpack bytes |
| `encode_tree(value, binary=False)` / `decode_tree(value)` | Arrays at any depth of a dict / list |
| `is_encoded_array(obj)` | Whether `obj` is an encoded array |
| `PROTOCOL_VERSION`, `CHUNK_FIELDS`, `CONTROL_MODES` | Wire version, chunk keys, accepted modes |
| `MSGPACK_MEDIA_TYPE` | `"application/msgpack"` |

**Errors during inference.** If the model itself raises, `/act` answers `500`
and a WebSocket connection is closed with code `1011`. Only malformed payloads
get a `400` or an `error` message.

## Clients (`hivla.client`)

`pip install 'hivla[client]'`. The module imports only numpy and
`hivla.protocol`; `httpx` / `websockets` / `msgpack` load on first use.

```python
from hivla.client import VlaWsClient

client = VlaWsClient("http://localhost:8000")         # or VlaClient(...) for HTTP
meta = client.reset()                                 # /metadata: n_action_steps, ...
chunk = client.act({"joint_pos": q, "images": {"head": img}, "instruction": "pick"})
chunk["joint_targets"]                                # (action_horizon, action_dim)
client.close()
```

| Name | Description |
|---|---|
| `VlaClient(endpoint, timeout=10.0, api_key=None, wire="json", timings=False)` | One HTTP request per call (`POST /act`, `/reset`, `/observe`, `GET /metadata`) on a pooled connection. `session` is its random session id, sent as `X-Hivla-Session` |
| `VlaWsClient(endpoint, timeout=10.0, api_key=None, wire="msgpack", timings=False)` | One long-lived `/ws` connection, which is its session; a reconnect is a new session. `endpoint` is the base URL, `http(s)://` or `ws(s)://`; the client appends `/ws` itself |
| `act(observation) -> dict` | Send a plain observation dict (numpy arrays, nested `images` / `ee_poses`, `instruction`), return the decoded chunk dict (`mode`, `joint_targets`, `kp`, `kd`, `ee_targets`, `gripper`; keys the server sent) |
| `reset() -> dict` | Start an episode: fetch the server metadata into `metadata`, tell the server to reset, and return the metadata. A server that is down leaves it `{}` and the error shows on the next `act`; a server that is up but refuses the reset raises |
| `observe(observation)` | Send one observation without asking for a chunk, for a model with `observe: every_step` |
| `metadata` | The last metadata; `VlaWsClient` fills it from the connect handshake |
| `close()` | Release the connection. Idempotent |
| `SERVER_TIMINGS_KEY` | `"server_timings_ms"`: with `timings=True` the chunk carries the server's own stage times (ms) under this key |
| `SessionLost` | A `RuntimeError` subclass. `VlaWsClient` raises it after a dropped connection to a stateful model, see below |

Both clients serialize calls with a lock, so a prefetch thread and the control
loop can share one. `VlaWsClient` never offers permessage-deflate and sets no
message size limit. After a timeout or a dropped connection it discards the
connection, so a late reply is never paired with the next request; the next
call reconnects. A server `error` reply raises `RuntimeError`; HTTP errors
raise `httpx.HTTPStatusError`.

The clients only move data: caching a chunk, replaying it and cutting it to
`n_action_steps` is the caller's job. Call `reset()` before each episode; a
stateful model refuses requests until it has been reset.

A `VlaWsClient` connection is its session, so a reconnect starts a new one. When
the connection of a started episode drops and the model is stateful, the next
`act` or `observe` reconnects, sends `reset` on the new connection and raises
`SessionLost` without sending its own request. The model has forgotten the
episode so far: start the episode over (or at least take a fresh observation).
Calls after that work normally. With a stateless model the client just
reconnects. `hivla.robot.run_episode` lets `SessionLost` propagate like any
other error, after telling the robot to `hold()`.

## Router

`hivla route --config router.yaml` starts a gateway that load-balances
`POST /act` across several `hivla serve` backends (for example one per GPU). It
holds no model. Clients point at the router exactly as at one server. Needs
`pip install 'hivla[server,client]'`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "backends": [...], "policy": "..."}` |
| `POST` | `/act` | Forwarded to one backend. The body and `?timings=` go through unchanged |

`/ws`, `/metadata`, `/reset` and `/observe` are not routed, so serve stateful
models without the router and connect to a backend directly for those.
A backend that cannot be reached gives `502`; a backend error status (for
example `400`) is passed through.

```yaml
router:
  backends:                 # base URLs; ${VAR}, ~ and ${VAR:-default} are expanded
    - http://127.0.0.1:8001
    - http://127.0.0.1:8002
  policy: round_robin       # round_robin | least_outstanding | consistent_hash
  host: 0.0.0.0
  port: 8000
  timeout: 30.0             # seconds per forwarded request
  # api_key: secret         # required on the router's /act and forwarded upstream
```

| Policy | Behavior |
|---|---|
| `round_robin` | Take turns; the body is not decoded |
| `least_outstanding` | Pick the backend with the fewest in-flight requests |
| `consistent_hash` | Hash the observation content so a repeated observation reaches the same backend and hits its cache. Decodes the body; a malformed body gets `400` |

Python: `hivla.serving.router.create_router_app(backends, policy="round_robin",
api_key=None, timeout=30.0)` returns the FastAPI app, and `run_router(...)`
(same arguments plus `host`, `port`) runs it. `hivla.serving.router.make_policy(name)`
builds a policy.

## Serving config

A YAML file read by `hivla serve` / `hivla inspect` through `load_config`. In
string values, `~` expands to the home directory, `$VAR` / `${VAR}` to an
environment variable, and `${VAR:-default}` to the variable or the default. An
unset variable with no default raises `KeyError`. Examples are in `src/hivla/configs/`.

```yaml
model:
  name: example-vla
  family: generic           # generic | lerobot | openpi | gr00t | starvla | rlinf
  checkpoint: ${CKPT_DIR}/example.pt
  backend: auto             # generic family only: auto | echo | torch | onnx | tensorrt | remote
  action_horizon: 16
  n_action_steps: 8         # steps a client executes per chunk; default action_horizon

runtime:
  device: cuda              # cuda | cpu
  precision: fp16           # fp32 | fp16 | bf16
  max_batch_size: 1         # > 1 enables dynamic batching
  batch_timeout: 0.002      # seconds to collect a batch
  cache: false              # replay the last chunk for an identical observation
  cache_ttl: 0.5
  warmup: true              # one dummy forward pass at start

policy:
  action_dim: 7
  control_mode: position    # `mode` attached to chunks that do not set one
  cameras: [head, left_wrist]
  norm_stats: path/to/norm_stats.json   # or an inline dict; generic family only
  rename_inputs: {joint_pos: state}     # observation key -> model key
  rename_outputs: {action: joint_targets}
  state_layout: [[joint_pos], [gripper, eef]]   # observation fields -> joint_pos, in order
  gripper_range: {eef: [1, -1]}       # model's value when [closed, open] -> opening [0, 1]
  joint_names: [joint_1, joint_2]     # training joint order, reported in /metadata

native_kwargs: {}           # top level: family-specific loader arguments

server:
  transport: http           # informational; /act and /ws are always mounted
  host: 0.0.0.0
  port: 8000
  api_key: secret           # optional
  extensions: []            # "module:attr" callables (app, engine) -> None
```

Each field maps to a `ModelSpec` field of the same name. Top-level keys other
than `model` / `runtime` / `policy` / `server` (such as `native_kwargs`) are kept
on the spec as extras, and the family reads what it needs.

`hivla serve` currently binds to `--host` / `--port`, which default to `0.0.0.0`
and `8000`, so `server.host` / `server.port` in the config have no effect on it.
Pass the flags to change the address. `hivla route` does honor `router.host` /
`router.port` when no flags are given.

`norm_stats` maps a key to its statistics. The mode is `zscore` (`mean`, `std`),
`quantile` (`q01`, `q99`, mapped to `[-1, 1]`) or `minmax` (`min`, `max`, mapped
to `[-1, 1]`). When `mode` is omitted it is inferred from which fields are
present:

```json
{"state":  {"mean": [...], "std": [...]},
 "action": {"q01": [...], "q99": [...], "mode": "quantile"}}
```

## CLI

Installed as `hivla`. `--config` takes a file path or the name of a built-in
config in `src/hivla/configs/` (`gr00t_n1d6_gr1`); an existing file wins over a
built-in of the same name.

| Command | Description |
|---|---|
| `hivla --version` | Print the version |
| `hivla configs` | List the built-in configs |
| `hivla inspect --config PATH` | Print the resolved metadata, the backend that `auto` resolves to (`family:<name>` for non-generic families) and the available families. Loads no weights |
| `hivla serve --config PATH [--host H] [--port P]` | Build the engine, load the model, serve HTTP + WebSocket (blocking) |
| `hivla launch --config PATH [--env ENV] [--host H] [--port P] [--ready-timeout S] [--log FILE]` | Run `hivla serve` with the Python of the config's `launch.env` (a conda env name, prefix or interpreter) and wait for `/health` |
| `hivla route --config PATH [--host H] [--port P]` | Run the [router](#router) from the config's `router:` section |

## ModelSpec

`hivla.core.spec.ModelSpec`, a pydantic model with extra fields allowed. The single
description of one servable model.

| Group | Field | Default |
|---|---|---|
| model | `name` | `"vla"` |
| | `family` | `"generic"` |
| | `checkpoint` | `None` |
| | `backend` | `"auto"` |
| | `action_horizon` | `16` |
| | `n_action_steps` | `None` (advertised as `action_horizon`); must be 1..`action_horizon` |
| runtime | `device` | `"cuda"` |
| | `precision` | `"fp32"` |
| | `max_batch_size` | `1` |
| | `batch_timeout` | `0.002` |
| | `cache` | `False` |
| | `cache_ttl` | `0.5` |
| | `warmup` | `True` |
| policy | `action_dim` | `7` |
| | `control_mode` | `"position"` |
| | `cameras` | `[]` (list, or `{name: [h, w]}`) |
| | `norm_stats` | `None` (path or dict) |
| | `rename_inputs`, `rename_outputs` | `{}` |
| | `state_layout`, `gripper_range`, `joint_names` | `None` |
| server | `transport` | `"http"` |
| | `host` | `"0.0.0.0"` |
| | `port` | `8000` |
| | `api_key` | `None` |

| Method | Description |
|---|---|
| `ModelSpec.from_config(config)` | Accepts the nested YAML shape or a flat dict. Top-level keys override the nested sections |
| `load_norm_stats() -> Optional[Dict[str, NormStats]]` | Resolve `norm_stats` from a JSON path or an inline dict |
| `metadata() -> dict` | The `/metadata` payload |

Extra keys (such as `native_kwargs`) are available as `spec.model_extra`.

## Building an engine

```python
from hivla import load_config, build_engine
from hivla.serving.server import serve

engine = build_engine(load_config("example"))
serve(engine, host="0.0.0.0", port=8000)
```

| Function | Description |
|---|---|
| `load_config(path) -> dict` | Read YAML (a path or a built-in config name) with `~` / environment-variable expansion |
| `build_engine(config) -> InferenceEngine` | `ModelSpec.from_config(config)`, then `build_engine_from_spec` |
| `build_engine_from_spec(spec) -> InferenceEngine` | Look up the family adapter, build its backend and transforms, wrap them in a `VlaPolicy`, add an `ObservationCache` if `spec.cache` is set |

The engine is returned unstarted. `serve()` starts it; call `engine.start()`
yourself when using the engine in-process.

## InferenceEngine and AsyncInferenceEngine

`hivla.serving.engine`.

### `InferenceEngine(policy, cache=None, max_batch_size=1, batch_timeout=0.002, warmup=True)`

| Member | Description |
|---|---|
| `start()` | Load the backend, run `backend.warmup()` if `warmup` is set, and start the batch worker if `max_batch_size > 1`. Idempotent |
| `infer(observation, session=None) -> dict` | Decoded observation dict → action-chunk dict. Raises `RuntimeError` before `start()`, and `SessionError` when a stateful model belongs to another session. Thread-safe; with batching on, concurrent calls are grouped into one `VlaPolicy.infer_batch` call |
| `reset(session=None)` | Reset the model and clear the cache; a stateful model is bound to `session` |
| `observe(observation, session=None)` | Hand one observation to a stateful model. A stateless model ignores it |
| `stateful` | Whether the model keeps state between requests. A stateful model refuses a cache or `max_batch_size > 1` |
| `stop()` | Stop batching and unload the backend |
| `metadata` | The policy's metadata dict |

Batching waits up to `batch_timeout` seconds, or until `max_batch_size` requests
are pending. An exception in a batch is raised in every request of that batch.

### `ObservationCache(max_size=16, ttl=0.5)`

LRU cache from observation content (shape, dtype, CRC32 of every array) to its
chunk. `InferenceEngine` returns a cached chunk for an identical observation seen
within `ttl` seconds. Methods: `get(obs)`, `put(obs, chunk)`, `clear()`, `len()`.

### `AsyncInferenceEngine(policy)`

Runs inference on a background thread so it overlaps with chunk execution.
Library-level only; the server does not use it.

| Method | Description |
|---|---|
| `start()` | Load the backend and start the worker. Does not run warmup |
| `submit(observation)` | Hand over the newest observation (non-blocking). Superseded observations are dropped. Raises `RuntimeError` before `start()` |
| `get_action() -> Optional[dict]` | The most recently completed chunk, or `None` if none is ready. It keeps returning the same chunk until a newer one finishes |
| `stop()` | Stop the worker and unload the backend |
| `metadata` | The policy's metadata dict |

## VlaPolicy

`hivla.core.policy.VlaPolicy(backend, input_transforms=(), output_transforms=(),
action_horizon=None, default_mode="position", metadata=None)`

The pipeline `observation → input_transforms → backend.predict →
output_transforms → chunk`.

| Method | Description |
|---|---|
| `infer(observation) -> dict` | Run the pipeline on one observation |
| `infer_batch(observations) -> list` | Transform each input, call `backend.predict_batch` once, decode each output |
| `metadata` | The `metadata` argument, plus `action_horizon` if given |

Each chunk gets `mode = default_mode` if the output transforms did not set one.
It is then validated: `joint_targets` must be 2-D, each `ee_targets` entry 2-D,
each `gripper` entry 1-D, and, when `action_horizon` is set, their leading axis
must equal it. A failure raises `ValueError`.

## Transforms

`hivla.core.transforms`. Each transform is a frozen dataclass with
`__call__(dict) -> dict`. It returns a new dict and skips keys that are absent.
`compose(transforms)` chains a sequence into one callable.

| Transform | Arguments | Effect |
|---|---|---|
| `RenameKeys` | `mapping: {src: dst}` | Rename keys |
| `SelectKeys` | `keys` | Keep only the listed keys |
| `Normalize` | `stats: {key: NormStats}`, `mode=None`, `eps=1e-6` | Normalize each key with its stats. `mode` overrides every stat's own mode |
| `Unnormalize` | same | Inverse of `Normalize` |
| `Slice` | `key`, `dim` | Keep the first `dim` entries of the last axis |
| `PadToDim` | `key`, `dim`, `value=0.0` | Pad (or truncate) the last axis to `dim` |
| `SinCos` | `key` | Replace the last axis with `[sin(x), cos(x)]` (doubles its width) |
| `AsFloat` | `key`, `scale=1.0`, `offset=0.0` | `float32(x) * scale + offset` |
| `ResizeImages` | `key`, `size` | Resize one `HxWxC` image or a `{name: image}` dict to `size x size`. Needs the `vision` extra (Pillow) |

`NormStats(mean, std, q01, q99, min, max, mode)` holds one key's statistics.
`NormStats.from_dict(d)` builds it from JSON. `resolve_mode()` returns `mode`, or
infers it from the fields present (`zscore` by default). Statistics wider than the
data are sliced to the data's last-axis width.

Only the `generic` family applies `norm_stats`. Every family applies
`rename_inputs` before the backend and `rename_outputs` after it.
`build_engine` adds two transforms around every family: `ConcatState`
(`state_layout`) first on the input side, and `GripperToOpening`
(`gripper_range`) last on the output side.

## Model families

`hivla.families`. A family adapter decides how one family of models is loaded
(the backend) and how its I/O is wrapped (the transforms). Adding a model family
means registering an adapter. You do not subclass `VlaPolicy`. Each family's
environment, config keys and field mapping are in its
[guide](families/README.md).

| Family | Loads through | Notes |
|---|---|---|
| `generic` | `build_backend(spec.backend, ...)` | hiVLA owns normalization: `rename_inputs` → `Normalize` → backend → `Unnormalize` → `rename_outputs` |
| `lerobot` | LeRobot v0.6.2 checkpoint and its pre/post processors | Observation keys are lerobot feature names (`observation.state`, `observation.images.<cam>`) |
| `openpi` | openpi `create_trained_policy` (π0 / π0.5 / π0-FAST, JAX or PyTorch) | Requires `native_kwargs.config` and `native_kwargs.cameras` |
| `gr00t` | Isaac GR00T N1.6 / N1.7 `Gr00tPolicy` | Requires `native_kwargs.embodiment_tag`. Splits `joint_pos` into GR00T state groups and joins the action groups into `joint_targets` |
| `starvla` | StarVLA `baseframework` checkpoint | State encoding, image resizing and unnormalization run on the server |
| `rlinf` | RLinf's openpi PyTorch reimplementation | Requires `native_kwargs.config` |

Non-generic families ignore `model.backend` and `norm_stats` and take loader
options from the top-level `native_kwargs`. The per-family example configs in `src/hivla/configs/` show the
keys each one expects.

```python
from hivla.families import FamilyAdapter, register_family

class MyFamily(FamilyAdapter):
    name = "my_family"
    def build_backend(self, spec):
        return MyBackend(spec.checkpoint)
    def build_transforms(self, spec):
        return [], []           # (input_transforms, output_transforms)

register_family(MyFamily())
```

| Function / class | Description |
|---|---|
| `FamilyAdapter` | Abstract: `name`, `build_backend(spec)`, `build_transforms(spec) -> (inputs, outputs)` |
| `register_family(adapter)` | Register an adapter instance under `adapter.name`. An existing name is replaced |
| `get_adapter(family)` | Raises `ValueError` listing available families |
| `available_families()` | Names in registration order |
| `rename_transforms(spec)` | `(inputs, outputs)` with only the spec's `rename_inputs` / `rename_outputs`, for a family whose model normalizes by itself |
| `GenericFamily` | The `generic` adapter |
| `NativeFamily(name, module, factory, call_method="infer", default_kwargs=None)` | Adapter for a package whose factory `module.factory(checkpoint=..., **native_kwargs)` returns a policy. Inference calls the policy itself if it is callable, otherwise its `call_method`. A non-dict result is taken as `joint_targets` |

## Backends

`hivla.runtimes` runs one checkpoint format; each family's own backend is in
`hivla.families.<name>`. A backend maps a dict of already-preprocessed inputs to a dict
of raw outputs. Heavy dependencies are imported in `load()`, and a missing one
raises an `ImportError` naming the extra to install.

### `ModelBackend`

| Method | Description |
|---|---|
| `load()` | Abstract. Load weights or open a connection |
| `predict(inputs) -> dict` | Abstract. One forward pass |
| `predict_batch(batch) -> list` | Defaults to calling `predict` on each item; override to stack into one pass |
| `warmup()` | One dummy forward pass after `load()`. No-op by default |
| `unload()` | Release resources. No-op by default |

### Built-in backends (generic family)

| `model.backend` | Class | Extra | Arguments taken from the spec |
|---|---|---|---|
| `echo` | `EchoBackend(action_dim=7, action_horizon=16)` | none | `action_dim`, `action_horizon`. Returns zeros `(horizon, action_dim)` as `joint_targets` |
| `torch` | `TorchBackend(checkpoint, device="cuda", precision="fp32", module_factory=None)` | `torch` | `checkpoint`, `device`, `precision` |
| `onnx` | `OnnxBackend(checkpoint, device="cpu")` | `onnx` | `checkpoint` |
| `tensorrt` | `TensorrtBackend(checkpoint, device="cuda")` | `tensorrt` | `checkpoint`, `device` |
| `remote` | `RemoteBackend(endpoint, timeout=30.0, api_key=None, wire="json")` | `client` | Proxies to another server's `/act` |

`build_backend(name, cfg)` builds one by name from a dict. With `name="auto"`
it calls `infer_backend(checkpoint)`, which returns `tensorrt` for
`.engine` / `.plan`, `onnx` for `.onnx`, `echo` when there is no checkpoint, and
`torch` otherwise. For a directory, the file extensions inside it decide.

`GenericFamily` passes only `checkpoint`, `action_dim`, `action_horizon`, `device`
and `precision` to `build_backend`. So `backend: remote` cannot be configured
from YAML (it needs an `endpoint`), and `TorchBackend`'s `module_factory` is not
reachable from YAML either. Construct those backends in Python and wrap them in a
`VlaPolicy` yourself.

## Embedding the server

`hivla.serving.server`:

| Function | Description |
|---|---|
| `create_app(engine, api_key=None, enable_ws=True, extensions=None)` | Build the FastAPI app. Does not start the engine |
| `serve(engine, host="0.0.0.0", port=8000, api_key=None, enable_ws=True, extensions=None)` | `engine.start()`, then run uvicorn with permessage-deflate disabled (blocking) |
| `load_extensions(paths)` | Resolve `"module:attr"` strings to callables |

An extension is a callable `(app, engine) -> None` that runs after the core
routes are mounted, so it can add its own endpoints.

A custom backend served in-process:

```python
from hivla import InferenceEngine, VlaPolicy
from hivla.runtimes.base import ModelBackend
from hivla.serving.server import serve
from hivla.core.transforms import RenameKeys

class MyBackend(ModelBackend):
    def load(self): ...
    def predict(self, inputs):
        return {"joint_targets": ...}            # (horizon, dof)

policy = VlaPolicy(MyBackend(),
                   input_transforms=[RenameKeys({"joint_pos": "state"})],
                   action_horizon=16,
                   metadata={"model": "mine", "action_dim": 7})
serve(InferenceEngine(policy), port=8000)
```
