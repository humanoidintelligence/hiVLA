#!/usr/bin/env python3
"""Download one large file over N parallel HTTP Range connections, resumably.

The mirrors reachable from the dev box throttle per connection (ModelScope ~0.3 MB/s
each, hf-mirror's xet bridge stalls), so a 7.5 GB checkpoint needs many streams.
Each piece goes to ``<out>.part/<index>``; a rerun keeps finished bytes and only
fetches what is missing. With ``--sha256`` the joined file is verified before it
replaces ``<out>``.

    python scripts/fetch_parallel.py URL OUT [--size N] [--conns 32] [--sha256 HEX]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PIECE = 64 << 20  # 64 MiB: small enough that a dropped connection loses little


def _size(url: str) -> int:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = resp.headers.get("Content-Range", "").rpartition("/")[2]
        return int(total)


def _fetch(url: str, path: str, start: int, end: int, done: list, lock: threading.Lock) -> None:
    """Fill ``path`` with bytes [start, end], resuming from its current length."""
    want = end - start + 1
    for attempt in range(50):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= want:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(path, "ab") as f:
                if resp.status != 206:
                    raise IOError(f"expected 206, got {resp.status}")
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    with lock:
                        done[0] += len(chunk)
        except Exception as exc:  # noqa: BLE001 - dropped/stalled connections: resume
            time.sleep(min(30, 2 + attempt))
            last = exc
    raise SystemExit(f"{path}: gave up after 50 attempts ({last!r})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("url")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=0, help="bytes (default: probe)")
    parser.add_argument("--conns", type=int, default=32)
    parser.add_argument("--sha256", default=None)
    args = parser.parse_args()

    size = args.size or _size(args.url)
    parts = f"{args.out}.part"
    os.makedirs(parts, exist_ok=True)
    pieces = [(i, s, min(s + PIECE, size) - 1) for i, s in enumerate(range(0, size, PIECE))]
    paths = [os.path.join(parts, f"{i:05d}") for i, _, _ in pieces]
    done = [sum(os.path.getsize(p) for p in paths if os.path.exists(p))]
    lock = threading.Lock()
    stop = threading.Event()

    def report() -> None:
        t0, d0 = time.time(), done[0]
        while not stop.wait(30):
            rate = (done[0] - d0) / (time.time() - t0) / 1e6
            print(f"{done[0] / 1e9:.2f}/{size / 1e9:.2f} GB, {rate:.1f} MB/s", flush=True)

    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(args.conns) as pool:
        futures = [pool.submit(_fetch, args.url, p, s, e, done, lock)
                   for (_, s, e), p in zip(pieces, paths)]
        for fut in futures:
            fut.result()
    stop.set()

    tmp = f"{args.out}.joined"
    digest = hashlib.sha256()
    with open(tmp, "wb") as out:
        for path in paths:
            with open(path, "rb") as f:
                while chunk := f.read(16 << 20):
                    digest.update(chunk)
                    out.write(chunk)
    if os.path.getsize(tmp) != size:
        sys.exit(f"joined size {os.path.getsize(tmp)} != {size}")
    if args.sha256 and digest.hexdigest() != args.sha256:
        sys.exit(f"sha256 mismatch: {digest.hexdigest()} != {args.sha256} (pieces kept)")
    os.replace(tmp, args.out)
    for path in paths:
        os.remove(path)
    os.rmdir(parts)
    print(f"{args.out}: {size} bytes, sha256 {digest.hexdigest()}")


if __name__ == "__main__":
    main()
