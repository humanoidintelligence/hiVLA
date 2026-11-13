"""Closed-loop smoke test: serve a real LeRobot ACT checkpoint through hiVLA.

Builds the engine from the built-in ``lerobot_act_aloha`` config (family ``lerobot``),
feeds a synthetic ALOHA observation (one 480x640 camera + 14-D state) and checks
the returned chunk is ``(action_horizon, 14)``. Runs on a GPU machine in the
``lerobot`` env; the checkpoint must already be in the processor format.

With ``--http`` the engine is served by the real FastAPI app on localhost and
every request goes ``RemoteBackend -> POST /act -> engine``; ``--ws`` sends the
same requests over one ``/ws`` connection instead. ``--wire`` picks JSON or
msgpack for either, covering the wire format as well as the model.
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np

from hivla import build_engine, load_config


def _serve_in_thread(engine, port: int):
    """Run ``create_app(engine)`` under uvicorn in a daemon thread; return the server."""
    import httpx
    import uvicorn

    from hivla.serving.server import UVICORN_WS_KWARGS, create_app

    server = uvicorn.Server(
        uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="warning",
                       **UVICORN_WS_KWARGS)
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return server
        except httpx.TransportError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server did not come up on port {port}")


def ws_connect(url: str, **kwargs):
    """Open a sync ``websockets`` client connection (hiRobot has no WS client yet)."""
    from websockets.sync.client import connect

    return connect(url, **kwargs)


def ws_act(ws, wire: str):
    """Return a blocking ``obs -> chunk`` that round-trips one ``act`` over ``ws``."""
    from hivla.protocol import decode_action_chunk, encode_observation, pack, unpack

    binary = wire == "msgpack"

    def act(obs):
        msg = {"type": "act", "observation": encode_observation(obs, binary)}
        ws.send(pack(msg) if binary else json.dumps(msg))
        frame = ws.recv()
        resp = unpack(frame) if binary else json.loads(frame)
        if resp.get("type") != "action":
            raise RuntimeError(f"ws error: {resp}")
        return decode_action_chunk(resp["chunk"])

    return act


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lerobot_act_aloha")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--http", action="store_true", help="go through POST /act on localhost")
    parser.add_argument("--ws", action="store_true", help="go through WS /ws on localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wire", choices=["json", "msgpack"], default="json",
                        help="wire format for --http / --ws")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("runtime", {})["device"] = args.device
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
        assert json.loads(ws.recv())["type"] == "metadata"
        infer = ws_act(ws, args.wire)
        print(f"serving on :{args.port}, requests via WS /ws ({args.wire})")

    rng = np.random.default_rng(0)
    horizon = config["model"]["action_horizon"]
    latencies = []
    for i in range(args.iters):
        obs = {
            "observation.images.top": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            "observation.state": rng.standard_normal(14).astype(np.float32),
        }
        t = time.perf_counter()
        chunk = infer(obs)
        latencies.append(time.perf_counter() - t)
        jt = chunk["joint_targets"]
        assert jt.shape == (horizon, 14), jt.shape
        assert np.isfinite(jt).all(), "non-finite actions"
        if i == 0:
            print(f"chunk {jt.shape} {jt.dtype} mode={chunk['mode']} "
                  f"range=[{jt.min():.3f}, {jt.max():.3f}]")
    if ws is not None:
        ws.close()
    if server is not None:
        server.should_exit = True
    engine.stop()
    warm = latencies[1:] or latencies
    print(f"latency: first {latencies[0] * 1e3:.1f} ms, "
          f"warm median {np.median(warm) * 1e3:.1f} ms over {len(warm)} iters")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
