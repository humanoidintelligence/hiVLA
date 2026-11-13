"""Wire format (protocol) for observations and action chunks.

The server (:mod:`hivla.serving.server`) and the clients (:mod:`hivla.client`, which
hiAgent uses) share this module, so both sides always agree. Arrays are
JSON-safe dicts of base64 data + dtype + shape. We deliberately do **not** use
pickle and refuse ``object``-dtype arrays, matching GR00T n1.7's hardening
against deserializing untrusted payloads.

A binary variant of the same envelope is available for image-heavy traffic: with
``binary=True`` the ``__ndarray__`` field holds raw bytes instead of base64, and
the payload is packed with msgpack (:func:`pack` / :func:`unpack`, media type
:data:`MSGPACK_MEDIA_TYPE`). This skips the base64 + JSON round trip (~9 ms for
one 480x640 RGB frame). msgpack carries no code objects, so the dtype checks
below are still the only gate. :func:`decode_array` accepts both forms.
"""

from __future__ import annotations

import base64
from typing import Any, Dict

import numpy as np

MSGPACK_MEDIA_TYPE = "application/msgpack"

# dtype kinds we are willing to (de)serialize; 'O' (object) is excluded on
# purpose as it is the vector for arbitrary-code-execution via pickle payloads.
_SAFE_KINDS = set("biufb")


def encode_array(arr: np.ndarray, binary: bool = False) -> Dict[str, Any]:
    """Encode an ndarray as a wire-safe dict (data + dtype + shape).

    ``binary=False`` base64-encodes the data (JSON wire); ``binary=True`` keeps
    it as raw bytes for the msgpack wire.
    """
    arr = np.asarray(arr)
    if arr.dtype.kind == "O":
        raise ValueError("refusing to serialize object-dtype array (unsafe)")
    data = np.ascontiguousarray(arr).tobytes()
    return {
        "__ndarray__": data if binary else base64.b64encode(data).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def decode_array(obj: Dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_array` (base64 or raw bytes), rejecting unsafe dtypes."""
    dtype = np.dtype(obj["dtype"])
    if dtype.kind == "O":
        raise ValueError("refusing to deserialize object-dtype array (unsafe)")
    if dtype.kind not in _SAFE_KINDS:
        raise ValueError(f"refusing to deserialize dtype {dtype!r}")
    raw = obj["__ndarray__"]
    data = raw if isinstance(raw, (bytes, bytearray, memoryview)) else base64.b64decode(raw)
    return np.frombuffer(data, dtype=dtype).reshape(obj["shape"])


def pack(payload: Any) -> bytes:
    """Pack a (binary-encoded) payload as msgpack bytes."""
    return _msgpack().packb(payload, use_bin_type=True)


def unpack(data: bytes) -> Any:
    """Unpack msgpack bytes; malformed input raises ``ValueError``."""
    return _msgpack().unpackb(data, raw=False)


def _msgpack():
    try:
        import msgpack
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "the msgpack wire format needs msgpack; install the 'client' or 'server' "
            "extra: pip install 'hivla[client]'."
        ) from exc
    return msgpack


def is_encoded_array(obj: Any) -> bool:
    """True if ``obj`` looks like an :func:`encode_array` payload."""
    return isinstance(obj, dict) and "__ndarray__" in obj


def encode_tree(value: Any, binary: bool = False) -> Any:
    """Encode every ndarray inside nested dicts / lists; leave other leaves as they are.

    Used for the free-form parts of the wire (``Observation.extra``,
    ``ActionChunk.extra``) and for the fixed fields, so one rule covers them all.
    numpy scalars become Python scalars so JSON can carry them.
    """
    if isinstance(value, np.ndarray):
        return encode_array(value, binary)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: encode_tree(v, binary) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_tree(v, binary) for v in value]
    return value


def decode_tree(value: Any) -> Any:
    """Inverse of :func:`encode_tree`: encoded arrays at any depth become ndarrays."""
    if is_encoded_array(value):
        return decode_array(value)
    if isinstance(value, dict):
        return {k: decode_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_tree(v) for v in value]
    return value


def decode_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Decode an observation request payload into a dict of arrays/strings.

    Encoded arrays become ndarrays at any depth (``joint_pos``, ``images[cam]``,
    ``ee_poses[name]``, arrays inside ``extra``); strings and scalars
    (``instruction``, ``gripper[name]``) pass through.
    """
    return {key: decode_tree(val) for key, val in payload.items()}


def encode_observation(obs: Dict[str, Any], binary: bool = False) -> Dict[str, Any]:
    """Encode a decoded observation dict into a wire payload.

    Inverse of :func:`decode_observation`. Used by the clients and by the
    remote backend to proxy an observation to another hiVLA server unchanged.
    """
    return {key: encode_tree(val, binary) for key, val in obs.items()}


#: Version of the observation / action-chunk wire format, reported in
#: ``/metadata`` as ``protocol_version``. Bump it when a field changes meaning
#: or a client would misread a reply; adding an optional field does not.
#: 2: the chunk carries every hiRobot ``ActionChunk`` field, ``parts`` included.
#: 3: sessions. ``reset`` / ``observe`` requests; ``/metadata`` has ``stateful``
#: and ``observe``; a stateful model refuses requests from other sessions.
PROTOCOL_VERSION = 3

#: HTTP header carrying the caller's session id on ``/act``, ``/reset`` and
#: ``/observe``. Over ``/ws`` each connection is one session.
SESSION_HEADER = "X-Hivla-Session"

#: hiRobot's ``ControlMode`` values (``hirobot.core.types``, 0.0.2): what a
#: chunk's ``mode`` may be. ``motion_clip`` is left out on purpose: hiRobot sends
#: a ``MotionClip`` as one ``Action``, never inside an ``ActionChunk``.
CONTROL_MODES = ("position", "velocity", "torque", "pd", "cartesian", "cartesian_delta",
                 "base_velocity", "motion_latent", "base_pose")
#: Modes whose ``joint_targets`` has one column per joint of the robot (no
#: ``joint_names``); the other modes may carry targets for a subset of joints.
JOINT_MODES = ("position", "velocity", "torque", "pd")

#: Chunk fields that are one array each (leading axis = horizon, except the
#: per-joint gains ``kp`` / ``kd`` shared by every step).
CHUNK_ARRAY_FIELDS = ("joint_targets", "kp", "kd", "base_velocity", "base_target", "latent")
#: Chunk fields that are ``name -> array`` maps (end-effector or gripper name).
CHUNK_MAP_FIELDS = ("ee_targets", "gripper")
#: Every key an action chunk may carry, as hiRobot's ``ActionChunk`` names them.
#: ``parts`` maps a part name to a chunk of the same shape (it may nest).
CHUNK_FIELDS = ("mode", *CHUNK_ARRAY_FIELDS, *CHUNK_MAP_FIELDS, "joint_names", "extra", "parts")


def encode_action_chunk(chunk: Dict[str, Any], binary: bool = False) -> Dict[str, Any]:
    """Encode an action-chunk dict for the ``/act`` response.

    The keys are hiRobot's ``ActionChunk`` fields (:data:`CHUNK_FIELDS`):
    ``mode`` is a string, ``joint_names`` a list of strings, the
    :data:`CHUNK_ARRAY_FIELDS` arrays, the :data:`CHUNK_MAP_FIELDS` name ->
    array maps, ``extra`` a free-form tree, and ``parts`` part name -> chunk,
    encoded the same way at every depth. Only keys that are set are emitted, so
    a model that outputs only ``joint_targets`` sends what it sent before.
    Keys outside :data:`CHUNK_FIELDS` are dropped.
    """
    out: Dict[str, Any] = {}
    if chunk.get("mode") is not None:
        out["mode"] = str(getattr(chunk["mode"], "value", chunk["mode"]))
    for key in CHUNK_ARRAY_FIELDS:
        val = chunk.get(key)
        if val is not None:
            out[key] = encode_array(np.asarray(val), binary)
    for key in CHUNK_MAP_FIELDS:
        val = chunk.get(key)
        if val:
            out[key] = {k: encode_array(np.asarray(v), binary) for k, v in val.items()}
    if chunk.get("joint_names") is not None:
        out["joint_names"] = [str(n) for n in chunk["joint_names"]]
    if chunk.get("extra"):
        out["extra"] = encode_tree(chunk["extra"], binary)
    if chunk.get("parts"):
        out["parts"] = {name: encode_action_chunk(sub, binary) for name, sub in chunk["parts"].items()}
    return out


def decode_action_chunk(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of :func:`encode_action_chunk`: payload -> decoded chunk dict."""
    chunk: Dict[str, Any] = {}
    if payload.get("mode") is not None:
        chunk["mode"] = payload["mode"]
    for key in CHUNK_ARRAY_FIELDS:
        val = payload.get(key)
        if val is not None:
            chunk[key] = decode_array(val)
    for key in CHUNK_MAP_FIELDS:
        val = payload.get(key)
        if val:
            chunk[key] = {k: decode_array(v) for k, v in val.items()}
    if payload.get("joint_names") is not None:
        chunk["joint_names"] = list(payload["joint_names"])
    if payload.get("extra"):
        chunk["extra"] = decode_tree(payload["extra"])
    if payload.get("parts"):
        chunk["parts"] = {name: decode_action_chunk(sub) for name, sub in payload["parts"].items()}
    return chunk
