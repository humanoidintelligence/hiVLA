"""Pokebot RoboAPI backend: the cloud ``poke-aurora`` model behind hiVLA's interface.

RoboAPI (https://developers.pokebot.com/docs/roboapi/latest) is an
OpenAI-compatible HTTP API. One task is one *session*; every request is one
*turn* that returns a block of end-effector poses, and the next turn must carry
the previous assistant message and an execution receipt. So this backend is
``stateful``: :meth:`reset` ends the session, the next :meth:`predict` opens a
new one.

What it translates (names from ``native_kwargs``):

* observation ``ee_poses[ee]`` (``[x, y, z, qx, qy, qz, qw]``) and
  ``gripper[gripper]`` (opening in [0, 1]) -> ``state`` (8 numbers, the same
  layout); ``images[camera]`` -> one PNG / JPEG data URL; ``instruction`` as is.
* the ``execute_actions`` tool call's ``output`` (N x 8, absolute poses in the
  same frame) -> ``ee_targets[ee]`` (N, 7) and ``gripper[gripper]`` (N,),
  ``mode: cartesian``. Quaternions are renormalized.

Receipts: the turn after a chunk reports ``executed_steps`` from the
observation's ``extra["executed_steps"]`` when the client sends it, else
``min(n_action_steps, rows)``; ``status`` is ``completed`` when every row ran.

Failures: no automatic retry. When a request times out or the connection
drops, the backend looks the turn up (``GET /sessions/{id}/turns/{turn}``) and
uses the stored response if the server finished it, so one block is never
fetched (and executed) twice. Any other failure closes the session and raises;
the next request opens a new one.

The API key is read from the environment (``api_key_env``, default
``POKEBOT_API_KEY``); it is never logged or reported in ``/metadata``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from hivla.core.spec import ModelSpec
from hivla.families.base import FamilyAdapter, rename_transforms
from hivla.runtimes.base import ModelBackend

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.pokebot.com/v1"
DEFAULT_MODEL = "poke-aurora-preview"
STATE_DIM = 8
TOOL_NAME = "execute_actions"
IMAGE_FORMATS = ("jpeg", "png")


class PokebotError(RuntimeError):
    """RoboAPI returned an error or a response that breaks the session contract."""


class PokebotBackend(ModelBackend):
    """Call RoboAPI for each action chunk, one session per episode.

    Args:
        spec: The :class:`~hivla.core.spec.ModelSpec`. ``action_horizon`` is the
            chunk length (longer blocks are cut to it; the receipt then reports
            fewer steps). ``spec.extra["native_kwargs"]`` holds:

            * ``base_url`` (default ``https://api.pokebot.com/v1``), ``model``
              (default ``poke-aurora-preview``, an ``id`` from ``GET /models``).
            * ``api_key_env``: environment variable with the key
              (default ``POKEBOT_API_KEY``).
            * ``ee`` (required): hiRobot end-effector name for the pose in and
              ``ee_targets`` out; ``gripper``: gripper name (default ``ee``).
            * ``camera`` (required): the robot camera sent as the image.
            * ``image_format``: ``jpeg`` (default) or ``png``; ``jpeg_quality``
              (default 90); ``max_pixels``: images larger than this are scaled
              down, keeping the aspect ratio (default 640 x 360, as RoboAPI's
              own runtime sends).
            * ``instruction``: used when the observation has none.
            * ``timeout``: seconds per request (default 60).
        transport: An ``httpx`` transport, for tests.
    """

    stateful = True
    observe_mode = "chunk"

    def __init__(self, spec: ModelSpec, transport: Any = None) -> None:
        self._spec = spec
        kw: Dict[str, Any] = dict((spec.model_extra or {}).get("native_kwargs", {}) or {})
        self._base_url = str(kw.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._model = str(kw.get("model") or DEFAULT_MODEL)
        self._api_key_env = str(kw.get("api_key_env") or "POKEBOT_API_KEY")
        self._ee = kw.get("ee")
        if not self._ee:
            raise ValueError("family 'pokebot' needs native_kwargs.ee (end-effector name)")
        self._gripper = str(kw.get("gripper") or self._ee)
        self._camera = kw.get("camera")
        if not self._camera:
            raise ValueError("family 'pokebot' needs native_kwargs.camera (robot camera name)")
        self._image_format = str(kw.get("image_format", "jpeg")).lower()
        if self._image_format not in IMAGE_FORMATS:
            raise ValueError(f"pokebot image_format must be one of {IMAGE_FORMATS}, "
                             f"got {self._image_format!r}")
        self._jpeg_quality = int(kw.get("jpeg_quality", 90))
        self._max_pixels = int(kw.get("max_pixels", 640 * 360))
        self._instruction = kw.get("instruction")
        self._timeout = float(kw.get("timeout", 60.0))
        self._transport = transport

        self._client: Any = None
        self._served_model: Optional[str] = None
        self._max_chars: Optional[int] = None
        self._lock = threading.Lock()
        self._session: Optional[str] = None
        self._next_turn = 1
        # The last turn's assistant message and row count, until its receipt is sent.
        self._pending: Optional[Dict[str, Any]] = None
        self._delivered = 0

    # --- lifecycle ----------------------------------------------------------

    def load(self) -> None:
        import httpx

        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(f"family 'pokebot' reads the API key from ${self._api_key_env}, "
                               "which is not set")
        self._client = httpx.Client(
            base_url=self._base_url, timeout=self._timeout, transport=self._transport,
            headers={"Authorization": f"Bearer {key}"})
        ids = [m.get("id") for m in self._get("/models").get("data", [])]
        if self._model not in ids:
            raise PokebotError(f"model {self._model!r} is not available to this key; "
                               f"GET /models lists {ids}")
        self._check_schema(self._get(f"/models/{self._model}/schema"))

    def _check_schema(self, schema: Dict[str, Any]) -> None:
        self._served_model = schema.get("model")
        state = (schema.get("inputs", {}).get("properties", {}).get("state", {}))
        n = state.get("maxItems")
        if n is not None and int(n) != STATE_DIM:
            raise PokebotError(f"model {self._model!r} takes a {n}-number state; "
                               f"this backend sends {STATE_DIM}")
        cols = schema.get("session", {}).get("output", {}).get("columns")
        if cols is not None and int(cols) != STATE_DIM:
            raise PokebotError(f"model {self._model!r} returns {cols} columns, "
                               f"expected {STATE_DIM}")
        if schema.get("session", {}).get("supported") is False:
            raise PokebotError(f"model {self._model!r} does not support sessions")
        image = schema.get("image", {})
        if image.get("max_pixels"):
            self._max_pixels = min(self._max_pixels, int(image["max_pixels"]))
        self._max_chars = schema.get("text", {}).get("max_chars")
        log.info("pokebot: model %s is served by %s", self._model, self._served_model)

    def warmup(self) -> None:
        # A real request costs balance and opens a session; nothing to warm locally.
        pass

    def reset(self) -> None:
        with self._lock:
            self._close_session("stopped")

    def unload(self) -> None:
        with self._lock:
            self._close_session("stopped")
            if self._client is not None:
                self._client.close()
                self._client = None

    def metadata(self) -> Dict[str, Any]:
        # Read once when the policy is built, before load(): only static fields.
        return {"pokebot": {"model": self._model, "base_url": self._base_url, "ee": self._ee,
                            "gripper": self._gripper, "camera": self._camera}}

    # --- inference ----------------------------------------------------------

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user = self._user_message(inputs)
        executed = _executed_steps(inputs)
        with self._lock:
            if self._client is None:
                raise RuntimeError("PokebotBackend.load() must be called first")
            if self._session is None:
                self._open_session()
            messages: List[Dict[str, Any]] = []
            if self._pending is not None:
                messages += [self._pending, self._receipt(executed)]
            messages.append(user)
            turn = self._next_turn
            body = {"model": self._model,
                    "metadata": {"session_id": self._session, "turn": str(turn)},
                    "messages": messages}
            try:
                resp = self._chat(body, turn)
                rows = self._parse(resp, turn)
            except Exception:
                # RoboAPI ends the run on a failed turn; start fresh next time.
                self._close_session("failed")
                raise
            self._pending = resp["choices"][0]["message"]
            self._delivered = len(rows)
            self._next_turn = int(resp["metadata"]["next_turn"])
        return self.to_chunk(rows)

    def _chat(self, body: Dict[str, Any], turn: int) -> Dict[str, Any]:
        import httpx

        try:
            r = self._client.post("/chat/completions", json=body)
        except httpx.TransportError as exc:
            # The server may have finished the turn; use its stored response rather
            # than sending the turn again.
            stored = self._stored_turn(turn)
            if stored is None:
                raise PokebotError(f"turn {turn} of session {self._session} failed "
                                   f"({type(exc).__name__}) and the server has no result") from exc
            log.warning("pokebot: turn %d recovered from the server after %s",
                        turn, type(exc).__name__)
            return stored
        return _json_or_raise(r, "POST /chat/completions")

    def _stored_turn(self, turn: int) -> Optional[Dict[str, Any]]:
        try:
            r = self._client.get(f"/sessions/{self._session}/turns/{turn}")
        except Exception:  # noqa: BLE001 - lookup is best effort
            return None
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "succeeded" or not isinstance(data.get("response"), dict):
            return None
        return data["response"]

    def _parse(self, resp: Dict[str, Any], turn: int) -> np.ndarray:
        meta = resp.get("metadata") or {}
        if meta.get("session_id") != self._session or str(meta.get("turn")) != str(turn):
            raise PokebotError(f"response is for session {meta.get('session_id')!r} turn "
                               f"{meta.get('turn')!r}, sent {self._session!r} turn {turn}")
        if "next_turn" not in meta:
            raise PokebotError("response metadata has no next_turn")
        try:
            calls = resp["choices"][0]["message"]["tool_calls"] or []
        except (KeyError, IndexError, TypeError) as exc:
            raise PokebotError("response has no choices[0].message.tool_calls") from exc
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != TOOL_NAME:
            raise PokebotError(f"expected one {TOOL_NAME!r} tool call, got "
                               f"{[c.get('function', {}).get('name') for c in calls]}")
        try:
            output = json.loads(calls[0]["function"]["arguments"])["output"]
            rows = np.asarray(output, dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise PokebotError(f"{TOOL_NAME} arguments have no numeric output") from exc
        if rows.ndim != 2 or rows.shape[1] != STATE_DIM or rows.shape[0] == 0:
            raise PokebotError(f"{TOOL_NAME} output has shape {rows.shape}, "
                               f"expected (N, {STATE_DIM})")
        if not np.isfinite(rows).all():
            raise PokebotError(f"{TOOL_NAME} output has non-finite values")
        shape = meta.get("action_shape")
        if shape is not None and list(json.loads(shape)) != list(rows.shape):
            raise PokebotError(f"metadata.action_shape {shape} != output shape "
                               f"{list(rows.shape)}")
        return rows

    def to_chunk(self, rows: np.ndarray) -> Dict[str, Any]:
        """RoboAPI's (N, 8) block -> a hiRobot-shaped chunk, cut to ``action_horizon``."""
        horizon = self._spec.action_horizon
        if rows.shape[0] < horizon:
            raise PokebotError(f"model returned {rows.shape[0]} rows < action_horizon {horizon}")
        rows = rows[:horizon]
        poses = rows[:, :7].copy()
        norm = np.linalg.norm(poses[:, 3:7], axis=1, keepdims=True)
        if (norm < 1e-6).any():
            raise PokebotError("model returned a zero quaternion")
        poses[:, 3:7] /= norm
        return {"mode": "cartesian",
                "ee_targets": {self._ee: poses.astype(np.float32)},
                "gripper": {self._gripper: np.clip(rows[:, 7], 0.0, 1.0).astype(np.float32)}}

    # --- observation --------------------------------------------------------

    def _user_message(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        instruction = obs.get("instruction") or self._instruction
        if not instruction or not str(instruction).strip():
            raise ValueError("pokebot needs a non-empty instruction (observation "
                             "'instruction' or native_kwargs.instruction)")
        instruction = str(instruction)
        if self._max_chars and len(instruction) > int(self._max_chars):
            raise ValueError(f"instruction has {len(instruction)} chars, the model takes "
                             f"at most {self._max_chars}")
        text = json.dumps({"instruction": instruction, "state": self.to_state(obs)})
        return {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": self._image_url(obs)}}]}

    def to_state(self, obs: Dict[str, Any]) -> List[float]:
        """``ee_poses[ee]`` + ``gripper[gripper]`` -> RoboAPI's 8-number state."""
        poses = obs.get("ee_poses") or {}
        if self._ee not in poses:
            raise ValueError(f"pokebot needs ee_poses[{self._ee!r}] (got {sorted(poses)})")
        pose = np.asarray(poses[self._ee], dtype=np.float64).reshape(-1)
        if pose.shape[0] != 7:
            raise ValueError(f"ee_poses[{self._ee!r}] has {pose.shape[0]} values, expected 7 "
                             "[x, y, z, qx, qy, qz, qw]")
        grippers = obs.get("gripper") or {}
        if self._gripper not in grippers:
            raise ValueError(f"pokebot needs gripper[{self._gripper!r}] (got {sorted(grippers)})")
        opening = float(np.asarray(grippers[self._gripper]).reshape(-1)[0])
        state = [*pose.tolist(), opening]
        if not all(math.isfinite(v) for v in state):
            raise ValueError("pokebot state has non-finite values")
        if np.linalg.norm(pose[3:7]) < 1e-6:
            raise ValueError("pokebot state has a zero quaternion")
        return state

    def _image_url(self, obs: Dict[str, Any]) -> str:
        from PIL import Image

        images = obs.get("images") or {}
        if self._camera not in images:
            raise ValueError(f"pokebot needs images[{self._camera!r}] (got {sorted(images)})")
        img = np.asarray(images[self._camera])
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"images[{self._camera!r}] has shape {img.shape}, expected HxWx3 RGB")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img)
        w, h = pil.size
        if w * h > self._max_pixels:
            scale = math.sqrt(self._max_pixels / (w * h))
            pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        buf = io.BytesIO()
        if self._image_format == "jpeg":
            pil.save(buf, format="JPEG", quality=self._jpeg_quality)
        else:
            pil.save(buf, format="PNG")
        return f"data:image/{self._image_format};base64," + base64.b64encode(buf.getvalue()).decode()

    # --- session ------------------------------------------------------------

    def _open_session(self) -> None:
        data = self._post("/sessions", {"model": self._model}, "POST /sessions")
        sid = data.get("id")
        if not sid or data.get("status", "open") != "open":
            raise PokebotError(f"POST /sessions returned no open session: {data.get('status')!r}")
        self._session = str(sid)
        self._next_turn = int(data.get("next_turn") or 1)
        self._pending = None
        self._delivered = 0

    def _receipt(self, executed: Optional[int]) -> Dict[str, Any]:
        if executed is None:
            n = self._spec.n_action_steps or self._spec.action_horizon
            executed = min(int(n), self._delivered)
        executed = max(0, min(int(executed), self._delivered))
        status = "completed" if executed == self._delivered else "interrupted"
        call_id = self._pending["tool_calls"][0]["id"]
        return {"role": "tool", "tool_call_id": call_id,
                "content": json.dumps({"executed_steps": executed, "status": status})}

    def _close_session(self, reason: str) -> None:
        """Close the session, sending the last turn's receipt. Best effort: the
        robot stopping never depends on it."""
        if self._session is None or self._client is None:
            self._session = None
            return
        messages: List[Dict[str, Any]] = []
        if self._pending is not None:
            # Whether the last block ran is unknown here; report what was asked for.
            messages = [self._pending, self._receipt(None)]
        try:
            self._post(f"/sessions/{self._session}/close",
                       {"reason": reason, "messages": messages}, "POST /sessions/{id}/close")
        except Exception as exc:  # noqa: BLE001
            log.warning("pokebot: closing session %s failed: %s", self._session, exc)
        self._session = None
        self._pending = None
        self._delivered = 0

    # --- http ---------------------------------------------------------------

    def _get(self, path: str) -> Dict[str, Any]:
        return _json_or_raise(self._client.get(path), f"GET {path}")

    def _post(self, path: str, body: Dict[str, Any], what: str) -> Dict[str, Any]:
        return _json_or_raise(self._client.post(path, json=body), what)


def _json_or_raise(r: Any, what: str) -> Dict[str, Any]:
    if r.status_code >= 400:
        try:
            err = r.json().get("error", {})
            msg = err.get("message") if isinstance(err, dict) else str(err)
        except Exception:  # noqa: BLE001
            msg = r.text[:200]
        rid = r.headers.get("x-request-id")
        raise PokebotError(f"{what} -> HTTP {r.status_code}: {msg}"
                           + (f" (X-Request-ID {rid})" if rid else ""))
    try:
        return r.json()
    except ValueError as exc:
        raise PokebotError(f"{what} returned non-JSON (wrong base_url?)") from exc


def _executed_steps(obs: Dict[str, Any]) -> Optional[int]:
    extra = obs.get("extra") or {}
    value = extra.get("executed_steps") if isinstance(extra, dict) else None
    if value is None:
        return None
    return int(np.asarray(value).reshape(-1)[0])


class PokebotFamily(FamilyAdapter):
    """Serve Pokebot's cloud RoboAPI model (``poke-aurora``) as a hiVLA backend.

    The model reads one wrist image, the end-effector pose and the gripper
    opening, and returns absolute end-effector poses in its task frame; see
    :class:`PokebotBackend`.
    """

    name = "pokebot"

    def build_backend(self, spec: ModelSpec) -> ModelBackend:
        return PokebotBackend(spec)

    def build_transforms(self, spec: ModelSpec):
        return rename_transforms(spec)
