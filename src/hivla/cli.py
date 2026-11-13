"""Command-line entry point for hiVLA."""

from __future__ import annotations

import argparse
import json
import sys

from hivla import __version__


_CONFIG_HELP = "Serving config: a file path or a built-in name (see `hivla configs`)."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hivla",
        description="Model deployment and inference serving for humanoid VLA models.",
    )
    parser.add_argument("--version", action="version", version=f"hivla {__version__}")

    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Serve a model for inference over the network.")
    serve.add_argument("--config", required=True, help=_CONFIG_HELP)
    # None so server.host / server.port in the config apply (defaults 0.0.0.0:8000).
    serve.add_argument("--host", default=None, help="Host to bind (overrides server.host).")
    serve.add_argument("--port", type=int, default=None,
                       help="Port to bind (overrides server.port).")

    launch = sub.add_parser(
        "launch", help="Serve a model in its own env (config 'launch.env'), wait until ready.")
    launch.add_argument("--config", required=True, help=_CONFIG_HELP)
    launch.add_argument("--env", default=None,
                        help="Conda env name, env prefix or python path (overrides launch.env).")
    launch.add_argument("--host", default=None, help="Host to bind (overrides server.host).")
    launch.add_argument("--port", type=int, default=None,
                        help="Port to bind (overrides server.port).")
    launch.add_argument("--ready-timeout", type=float, default=None,
                        help="Seconds to wait for /health (overrides launch.ready_timeout).")
    launch.add_argument("--log", default=None,
                        help="Worker log file (default: a temp file, path printed).")

    inspect = sub.add_parser(
        "inspect", help="Resolve a config and print the manifest without loading weights."
    )
    inspect.add_argument("--config", required=True, help=_CONFIG_HELP)

    route = sub.add_parser(
        "route", help="Load-balance /act across a pool of hivla serve backends."
    )
    route.add_argument("--config", required=True, help="Router config: a path or a built-in name.")
    route.add_argument("--host", default=None, help="Host to bind.")
    route.add_argument("--port", type=int, default=None, help="Port to bind.")

    sub.add_parser("configs", help="List the built-in configs (names usable as --config).")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        from hivla import build_engine, load_config
        from hivla.serving.server import load_extensions, serve

        config = load_config(args.config)
        engine = build_engine(config)
        spec = config.get("server", {})
        host = args.host or spec.get("host", "0.0.0.0")
        port = args.port or spec.get("port", 8000)
        extensions = load_extensions(spec.get("extensions", []))
        serve(engine, host=host, port=port, api_key=spec.get("api_key"),
              extensions=extensions)
        return 0

    if args.command == "launch":
        from hivla.serving.launch import LaunchError, launch

        try:
            worker = launch(args.config, env=args.env, host=args.host, port=args.port,
                            ready_timeout=args.ready_timeout, log=args.log or True)
        except LaunchError as exc:
            print(f"hivla launch: {exc}", file=sys.stderr)
            return 1
        print(f"ready: {worker.url} (pid {worker.pid}, log {worker.log_path})", flush=True)
        try:
            return worker.process.wait() or 0
        except KeyboardInterrupt:
            return 0
        finally:
            worker.stop()

    if args.command == "inspect":
        from hivla import load_config
        from hivla.runtimes import infer_backend
        from hivla.families import available_families
        from hivla.core.spec import ModelSpec

        spec = ModelSpec.from_config(load_config(args.config))
        if spec.family == "generic":
            backend = infer_backend(spec.checkpoint) if spec.backend == "auto" else spec.backend
        else:
            backend = f"family:{spec.family}"  # the family adapter builds its own backend
        report = {
            "metadata": spec.metadata(),
            "backend": spec.backend,
            "backend_resolved": backend,
            "family": spec.family,
            "transport": spec.transport,
            "available_families": available_families(),
        }
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "route":
        from hivla import load_config
        from hivla.serving.router import run_router

        config = load_config(args.config)
        spec = config.get("router", {})
        backends = spec.get("backends")
        if not backends:
            print("router config needs a non-empty 'router.backends' list", file=sys.stderr)
            return 1
        host = args.host or spec.get("host", "0.0.0.0")
        port = args.port or spec.get("port", 8000)
        run_router(
            backends,
            host=host,
            port=port,
            policy=spec.get("policy", "round_robin"),
            api_key=spec.get("api_key") or None,
            timeout=float(spec.get("timeout", 30.0)),
        )
        return 0

    if args.command == "configs":
        from hivla.core.config import builtin_configs, config_root

        print(f"# {config_root()}")
        for name in builtin_configs():
            print(name)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
