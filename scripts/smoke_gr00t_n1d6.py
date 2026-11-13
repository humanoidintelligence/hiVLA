"""Closed-loop smoke test for a VLA policy served through hiVLA.

Builds the engine from ``--config`` (default the built-in ``gr00t_n1d6_gr1``,
family ``gr00t``; ``starvla_qwengr00t_gr1`` and
``openpi_pi05_libero`` work the same way), feeds a synthetic
observation (one 256x256 image per ``policy.cameras`` entry, a ``--state-dim``
state, an instruction) and checks every action array in the chunk
(``joint_targets`` or ``ee_targets`` + ``gripper``) has the declared horizon,
adds up to ``policy.action_dim`` and is finite. Runs on a GPU machine in the
model's env.

Without flags the engine is called in-process. ``--http`` serves it with the
real FastAPI app on localhost and sends each request through ``POST /act``;
``--ws`` uses one ``/ws`` connection. ``--wire`` picks JSON or msgpack.
``--compile MODE`` compiles the DiT action head; ``--batch N`` also times one
``infer_batch`` of N observations.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from hivla import build_engine, load_config
from smoke_lerobot_act import _serve_in_thread, ws_act, ws_connect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gr00t_n1d6_gr1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--http", action="store_true", help="go through POST /act on localhost")
    parser.add_argument("--ws", action="store_true", help="go through WS /ws on localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wire", choices=["json", "msgpack"], default="json")
    parser.add_argument("--compile", default=None,
                        help="torch.compile mode for the DiT head (e.g. max-autotune)")
    parser.add_argument("--batch", type=int, default=0,
                        help="also time infer_batch with this many observations (in-process)")
    parser.add_argument("--state-dim", type=int, default=0,
                        help="joint_pos size (default: policy.action_dim; LIBERO: 8)")
    parser.add_argument("--instruction", default="pick up the apple and place it on the plate")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("runtime", {})["device"] = args.device
    if args.compile:
        config.setdefault("native_kwargs", {})["compile"] = args.compile
    engine = build_engine(config)
    t0 = time.perf_counter()
    engine.start()
    print(f"loaded on {args.device} in {time.perf_counter() - t0:.1f}s")

    infer, server, ws = engine.infer, None, None
    if args.http:
        from hivla.runtimes.remote import RemoteBackend

        server = _serve_in_thread(engine, args.port)
        infer = RemoteBackend(f"http://127.0.0.1:{args.port}", wire=args.wire).predict
        print(f"serving on :{args.port}, requests via POST /act ({args.wire})")
    elif args.ws:
        server = _serve_in_thread(engine, args.port)
        ws = ws_connect(f"ws://127.0.0.1:{args.port}/ws", max_size=None)
        meta = json.loads(ws.recv())
        assert meta["type"] == "metadata", meta
        print(f"ws metadata: {meta['metadata']}")
        infer = ws_act(ws, args.wire)
        print(f"serving on :{args.port}, requests via WS /ws ({args.wire})")

    rng = np.random.default_rng(0)
    horizon = config["model"]["action_horizon"]
    dof = config["policy"]["action_dim"]
    cameras = config["policy"].get("cameras") or ["ego_view"]
    state_dim = args.state_dim or dof

    def make_obs():
        return _obs(rng, state_dim, cameras, args.instruction)

    latencies = []
    for i in range(args.iters):
        obs = make_obs()
        t = time.perf_counter()
        chunk = infer(obs)
        latencies.append(time.perf_counter() - t)
        actions = _check_chunk(chunk, horizon, dof)
        if i == 0:
            parts = {k: np.shape(v) for k, v in _parts(chunk)}
            print(f"chunk {parts} mode={chunk['mode']} "
                  f"range=[{actions.min():.3f}, {actions.max():.3f}]")
    if ws is not None:
        ws.close()
    if server is not None:
        server.should_exit = True
    warm = latencies[1:] or latencies
    print(f"latency: first {latencies[0] * 1e3:.1f} ms, "
          f"warm median {np.median(warm) * 1e3:.1f} ms over {len(warm)} iters")

    if args.batch > 1:
        batch = [make_obs() for _ in range(args.batch)]
        times = []
        for _ in range(4):
            t = time.perf_counter()
            chunks = engine._policy.infer_batch(batch)
            times.append(time.perf_counter() - t)
        assert len(chunks) == args.batch
        for c in chunks:
            _check_chunk(c, horizon, dof)
        b = np.median(times[1:])
        print(f"batch {args.batch}: median {b * 1e3:.1f} ms per call, "
              f"{b * 1e3 / args.batch:.1f} ms per observation")
    engine.stop()
    print("SMOKE OK")


def _obs(rng, state_dim, cameras, instruction):
    return {
        "joint_pos": (rng.standard_normal(state_dim) * 0.1).astype(np.float32),
        "images": {c: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for c in cameras},
        "instruction": instruction,
    }


def _parts(chunk):
    """``(label, array)`` for every action array in a chunk."""
    if chunk.get("joint_targets") is not None:
        yield "joint_targets", np.asarray(chunk["joint_targets"])
    for key in ("ee_targets", "gripper"):
        for name, val in (chunk.get(key) or {}).items():
            yield f"{key}.{name}", np.asarray(val)


def _check_chunk(chunk, horizon, dof):
    """Assert every part has the declared horizon, dims add up to ``dof``, all finite."""
    parts = list(_parts(chunk))
    assert parts, f"chunk has no actions: {sorted(chunk)}"
    width = 0
    for label, arr in parts:
        assert arr.shape[0] == horizon, (label, arr.shape)
        assert np.isfinite(arr).all(), f"non-finite {label}"
        width += arr.shape[1] if arr.ndim == 2 else 1
    assert width == dof, f"action dims {width} != policy.action_dim {dof}"
    return np.concatenate([a.reshape(horizon, -1) for _, a in parts], axis=1)


if __name__ == "__main__":
    main()
