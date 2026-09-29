#!/usr/bin/env python3
"""Targeted probe for the agent upstream-proxy connection/fd leak (halogen-flash stalls).

Finding
-------
`agent/app/api/routes/proxy.py` used to build a fresh `ServerProxy` (and its own
`httpx.AsyncClient`) for **every** proxied request and never called `aclose()`.
Halogen-flash serves HTTP/1.1 keep-alive, so every leaked client squats an open
socket against the single loopback API listener inside the agent container. Over
time the agent runs out of descriptors / the upstream accept queue saturates, and a
new inference dispatch either never reaches the LLM (agent stalls to
`UPSTREAM_IDLE_TIMEOUT_SECONDS`) or cannot relay the response back upstream.

The fix makes the agent reuse one shared, bounded `ServerProxy`/`AsyncClient`
(`PROXY_MAX_CONNECTIONS` / `PROXY_MAX_KEEPALIVE_CONNECTIONS` /
`PROXY_KEEPALIVE_EXPIRY_SECONDS`) and closes it at shutdown.

Why this probe measures the AGENT process, not the public endpoint
----------------------------------------------------------------
The halogen-flash API/engine port is bound to 127.0.0.1 inside the agent container
and is never published (see `recipes/halogen-flash/README.md`). The leaked sockets
live in the *agent* process toward that loopback port, so they are only observable
by reading the agent host's `/proc/<pid>/fd` and `ss`. Run this script **on the
agent host** (e.g. `core@10.100.2.111`) or inside the agent container.

Discriminator (why we measure AFTER settle, not the burst peak)
--------------------------------------------------------------
With bounded concurrency C, both a leak and a fixed pool show only ~C sockets open
*at any instant during* the burst. The difference is what remains afterwards:
  * FIXED  -> connections are reused and, after `keepalive_expiry`, idle ones are
              closed, so the sustained count stays at or below the keepalive pool
              bound (PROXY_MAX_KEEPALIVE_CONNECTIONS).
  * LEAK   -> every request leaked its own never-closed client, so the sustained
              count tracks the cumulative request count (>> the pool bound).
The test is only conclusive when `--requests` exceeds the pool bound; otherwise a
leak would also fit under the bound and look like a pass.

Exit codes
----------
  0  PASS  - pooled/bounded behavior: after settle, sockets held <= pool bound.
  1  FAIL  - leak: after settle, agent still holds ~one socket per request,
             well above the keepalive pool bound.
  2  INCONCLUSIVE - trigger not exercised or signal too weak (e.g. too few
             requests vs the pool bound, no /proc or ss). NOT a pass.
  3  SETUP / transport error (agent unreachable, PID/port not found, no ok reqs).

Usage
-----
  # On the agent host, against the local agent HTTP API:
  python3 proxy_connection_leak_probe.py \
      --agent-url http://127.0.0.1:8080 \
      --server-id <uuid> --path v1/models \
      --requests 150 --concurrency 4 --pool-bound 50

  # Explicit PID/port:
  python3 proxy_connection_leak_probe.py --agent-pid 12345 --api-port 8091 \
      --agent-url http://127.0.0.1:8080 --server-id <uuid>

  # Validate the classifier with controlled scenarios (no real agent):
  python3 proxy_connection_leak_probe.py --self-test
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable

try:
    import httpx
except ImportError:  # pragma: no cover - httpx ships with the agent
    httpx = None  # type: ignore

PASS, FAIL, INCONCLUSIVE, SETUP_ERROR = 0, 1, 2, 3

# Tolerance so a couple of unrelated sockets (health checks, monitor, in-flight at
# sample time) do not flip a clearly-bounded pool into a false FAIL.
FD_TOLERANCE = 3


def log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def _resolve_pid(agent_pid: int | None) -> int | None:
    if agent_pid is not None:
        return agent_pid
    for pattern in ("uvicorn app.main", "app.main:app"):
        try:
            out = subprocess.check_output(
                ["pgrep", "-af", pattern], text=True, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.SubprocessError):
            out = ""
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if len(lines) == 1:
            return int(lines[0].split()[0])
        if lines:
            # Prefer the shortest command (the uvicorn entrypoint, not a shell).
            lines.sort(key=len)
            return int(lines[0].split()[0])
    log("could not auto-resolve agent PID; pass --agent-pid")
    return None


def _connections_to_port(pid: int, port: int) -> int | None:
    """Count sockets owned by `pid` connected to 127.0.0.1:`port` via `ss`.

    Returns the count, or None if `ss` is unavailable/errors OR if the output has
    no `pid=` attribution. The latter happens when `ss` is run as a non-root user:
    it still lists sockets but hides other processes' owners, so a 0 here is not a
    real measurement. In that case the caller falls back to /proc.
    """
    try:
        out = subprocess.check_output(
            ["ss", "-tanp", f"( dst = 127.0.0.1:{port} )"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if "pid=" not in out:
        # No process attribution available (non-root); not a trustworthy 0.
        return None
    count = 0
    for line in out.splitlines():
        if "ESTAB" not in line and "CLOSE-WAIT" not in line:
            continue
        pid_m = re.search(r"pid=(\d+)", line)
        if pid_m and int(pid_m.group(1)) == pid:
            count += 1
    return count


def _socket_fds_in_process(pid: int) -> int | None:
    """Total socket fds in the process (fallback when `ss` yields nothing).

    Less precise (counts all sockets, not just to the port) but still tracks
    per-request growth.
    """
    path = f"/proc/{pid}/fd"
    try:
        fds = os.listdir(path)
    except OSError:
        return None
    n = 0
    for fd in fds:
        try:
            link = os.readlink(os.path.join(path, fd))
        except OSError:
            continue
        if link.startswith("socket:["):
            n += 1
    return n


def count_agent_sockets(pid: int, port: int) -> int | None:
    """Prefer the precise per-port `ss` count; fall back to process-wide socket fds.

    `ss -p` only shows other users' pids when run as root. If the precise count is
    available (non-None) we use it; otherwise we fall back to counting the agent's
    socket fds from /proc, which still tracks per-request growth but is not
    port-specific.
    """
    precise = _connections_to_port(pid, port)
    if precise is not None:
        return precise
    return _socket_fds_in_process(pid)


def discover_api_port(agent_url: str, server_id: str) -> int | None:
    """Ask the agent for the live server's API port via /servers/status/<id>."""
    if httpx is None:
        return None
    try:
        r = httpx.get(f"{agent_url}/servers/status/{server_id}", timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            for key in ("port", "api_port"):
                if isinstance(data.get(key), int):
                    return data[key]
    except (httpx.HTTPError, ValueError) as exc:
        log(f"port discovery failed: {exc}")
    return None


def drive_requests(
    agent_url: str,
    server_id: str,
    path: str,
    n: int,
    concurrency: int,
    sampler: Callable[[], int | None],
) -> tuple[int, int, int | None]:
    """Fire N requests through the agent proxy route.

    Returns (ok, errors, peak). `peak` is the max socket count observed by a
    background poller during the burst (diagnostic; the decision uses after-settle).
    """
    if httpx is None:
        raise RuntimeError("httpx required to drive requests")
    url = f"{agent_url}/proxy/{server_id}/{path}"
    method = "GET" if path.rstrip("/").endswith("models") else "POST"
    body = (
        None
        if method == "GET"
        else {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 1,
        }
    )

    peak: list[int] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            val = sampler()
            if val is not None:
                peak.append(val)
            time.sleep(0.05)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    counters = {"ok": 0, "errors": 0}

    async def _run() -> None:
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0)
        ) as client:

            async def one() -> None:
                async with sem:
                    try:
                        resp = await client.request(method, url, json=body)
                        counters["ok" if resp.status_code < 500 else "errors"] += 1
                    except httpx.HTTPError:
                        counters["errors"] += 1

            await asyncio.gather(*(one() for _ in range(n)))

    try:
        asyncio.run(_run())
    finally:
        stop.set()
        poller.join(timeout=2)

    peak_val = max(peak) if peak else None
    return counters["ok"], counters["errors"], peak_val


def classify(
    before: int | None,
    after: int | None,
    n_requests: int,
    pool_bound: int,
) -> tuple[int, str]:
    """Decide leak vs pooled from the post-settle sustained socket count."""
    if before is None or after is None:
        return INCONCLUSIVE, "could not sample agent sockets (no /proc or ss)"

    delta = after - before
    log(
        f"samples: before={before} after_settle={after} (delta={delta}) "
        f"requests={n_requests} pool_bound={pool_bound}"
    )

    # The test only separates leak from pool when a leak would exceed the bound.
    if n_requests <= pool_bound + FD_TOLERANCE:
        return (
            INCONCLUSIVE,
            (
                f"requests ({n_requests}) not greater than pool bound "
                f"({pool_bound}); a leak would also fit under the bound. Raise "
                f"--requests well above --pool-bound for a conclusive result."
            ),
        )

    if delta <= pool_bound + FD_TOLERANCE:
        return (
            PASS,
            (
                f"pooled: after settle the agent holds {delta} socket(s), within "
                f"the keepalive pool bound ({pool_bound}). Connections are reused/closed."
            ),
        )

    if delta >= n_requests - FD_TOLERANCE:
        return (
            FAIL,
            (
                f"leak: after settle the agent still holds {delta} of {n_requests} "
                f"request sockets (~one per request, >> pool bound {pool_bound}). "
                f"Per-request clients are not being pooled or closed."
            ),
        )

    # Between the bound and the request count: still a leak (exceeds the pool).
    return (
        FAIL,
        (
            f"leak: sustained sockets ({delta}) exceed the keepalive pool bound "
            f"({pool_bound}) after settle; connections are not being released."
        ),
    )


def self_test() -> int:
    """Validate the classifier against controlled scenarios (no real agent)."""
    # (before, after, n_requests, pool_bound, expected_code, label)
    cases = [
        (10, 148, 150, 50, FAIL, "leak: ~one socket per request held"),
        (10, 120, 150, 50, FAIL, "leak: held > bound (between branch)"),
        (10, 64, 150, 50, FAIL, "leak: just above bound (54 > 50+3)"),
        (10, 63, 150, 50, PASS, "pooled: exactly at bound+tol (53)"),
        (10, 40, 150, 50, PASS, "pooled: settles under bound"),
        (10, 8, 150, 50, PASS, "pooled: settles to near baseline"),
        (10, 148, 40, 50, INCONCLUSIVE, "inconclusive: requests <= bound"),
        (None, 40, 150, 50, INCONCLUSIVE, "inconclusive: no before sample"),
        (10, None, 150, 50, INCONCLUSIVE, "inconclusive: no after sample"),
    ]
    failures = 0
    for before, after, n, bound, expected, label in cases:
        code, reason = classify(before, after, n, bound)
        ok = code == expected
        if not ok:
            failures += 1
        print(
            f"  [{'ok' if ok else 'MISMATCH'}] {label}: got {code} expected {expected} ({reason})"
        )
    if failures:
        print(f"self-test FAILED: {failures} mismatches")
        return 1
    print("self-test PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--agent-url", default="http://127.0.0.1:8080", help="Agent base URL"
    )
    ap.add_argument("--server-id", help="Live server UUID on the agent")
    ap.add_argument("--path", default="v1/models", help="Proxy path (GET-supported)")
    ap.add_argument(
        "--requests",
        type=int,
        default=150,
        help="Proxied requests (must exceed pool bound)",
    )
    ap.add_argument("--concurrency", type=int, default=4, help="Concurrent requests")
    ap.add_argument(
        "--agent-pid", type=int, help="Agent process PID (else auto-detect)"
    )
    ap.add_argument(
        "--api-port", type=int, help="Halogen-flash API port (else auto-discover)"
    )
    ap.add_argument(
        "--pool-bound", type=int, default=50, help="PROXY_MAX_KEEPALIVE_CONNECTIONS"
    )
    ap.add_argument(
        "--settle-seconds",
        type=float,
        default=35.0,
        help="Wait > keepalive_expiry so pooled idle sockets drop; leak stays.",
    )
    ap.add_argument(
        "--self-test", action="store_true", help="Run classifier self-test and exit"
    )
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if httpx is None:
        log("httpx not installed; run inside the agent env")
        return SETUP_ERROR
    if not args.server_id:
        log("--server-id is required to drive proxy requests")
        return SETUP_ERROR

    pid = _resolve_pid(args.agent_pid)
    if pid is None:
        return SETUP_ERROR
    log(f"agent pid = {pid}")

    api_port = args.api_port or discover_api_port(args.agent_url, args.server_id)
    if api_port is None:
        log("could not determine the halogen-flash API port (pass --api-port)")
        return SETUP_ERROR
    log(f"halogen-flash API port = {api_port}")

    before = count_agent_sockets(pid, api_port)
    if before is None:
        log("no socket visibility (need ss or /proc/<pid>/fd access; run as root)")
        return INCONCLUSIVE
    log(f"baseline sockets to :{api_port} = {before}")

    try:
        ok, errors, peak = drive_requests(
            args.agent_url,
            args.server_id,
            args.path,
            args.requests,
            args.concurrency,
            lambda: count_agent_sockets(pid, api_port),
        )
    except (httpx.HTTPError, RuntimeError, OSError) as exc:
        log(f"request driver failed: {exc}")
        return SETUP_ERROR
    log(f"requests: ok={ok} errors={errors} (burst peak observed={peak})")
    if ok == 0:
        log("no successful requests; cannot judge the leak")
        return SETUP_ERROR

    log(f"settling {args.settle_seconds}s to observe keepalive expiry...")
    time.sleep(args.settle_seconds)
    after = count_agent_sockets(pid, api_port)

    code, reason = classify(before, after, args.requests, args.pool_bound)
    label = {
        PASS: "PASS",
        FAIL: "FAIL",
        INCONCLUSIVE: "INCONCLUSIVE",
        SETUP_ERROR: "SETUP_ERROR",
    }[code]
    log(f"RESULT: {label} - {reason}")
    return code


if __name__ == "__main__":
    sys.exit(main())
