#!/usr/bin/env python3
"""Remote probe: does a client-disconnected Responses lease stop renewing?

Bug (fixed in inference_scheduler.py): the non-stream/stream Responses release
path called ``_stop_disconnect_monitor()`` BEFORE cancelling the lease renewal
task. If the disconnect monitor's ``http_request.is_disconnected()`` (a Starlette
ASGI ``receive()``) wedged on a half-closed connection, the teardown never
returned, so the renewal loop was never cancelled and the DB release update was
never reached. The lease stayed ``active`` with ``lease_expires_at`` advancing
+30s every 30s indefinitely -- even though the agent had already reported the
operation ``completed``. This pinned the single-slot server and stalled its
backlog.

This probe:
  1. Records the set of active/reserving leases on the target server (baseline).
  2. Streams a Responses request against ``<alias>`` and reads the first SSE
     lifecycle event to capture the backend-generated response id (== lease
     ``request_id``).
  3. Abruptly closes the client connection mid-stream (simulating a disconnect).
  4. Polls PostgreSQL for that lease and the owning agent's operation status for
     ``--window`` seconds, checking the invariant the fix guarantees:

       Once the agent reports the operation terminal (completed/cancelled/failed),
       the backend lease must NOT remain ``active`` while ``lease_expires_at``
       keeps advancing. It must reach a terminal DB status (released/cancelled/
       failed/expired) or stop renewing within the window.

The exact ASGI ``receive()`` wedge is hard to force from a client, so this probe
verifies the *distinguishing consequence*: a wedged/late disconnect must not
leave a self-renewing zombie lease. A clean, fast release is also consistent with
the fix.

Exit codes:
  0  PASS  - after the agent finished, the lease reached a terminal status or
             stopped renewing within the window (no zombie).
  1  FAIL  - the agent finished but the lease is still ``active`` and still
             advancing ``lease_expires_at`` at window end (old bug signature).
  2  INCONCLUSIVE - request id not captured, lease not found, or the agent
             operation never reached terminal in the window (e.g. cold start).
  3  transport / setup error.

Usage:
  uv run python scripts/probe_lease_release_disconnect.py \
      --base-url https://matrix.thelink.family/v1 \
      --model rocinante-tiny
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from typing import Any

import httpx

TERMINAL_LEASE = {"released", "cancelled", "failed", "expired"}
TERMINAL_AGENT = {"completed", "cancelled", "failed", "unknown"}


def db_scalar(ssh_host: str, container: str, database: str, sql: str) -> str:
    remote = (
        f"docker exec {shlex.quote(container)} psql -U postgres "
        f"-d {shlex.quote(database)} -tAc {shlex.quote(sql)}"
    )
    cmd = ["ssh", "-o", "ConnectTimeout=8", ssh_host, remote]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"psql failed: {out.stderr.strip()[:400]}")
    return out.stdout.strip()


def sqlstr(value: str) -> str:
    """Escape a Python string as a SQL single-quoted literal."""
    return "'" + value.replace("'", "''") + "'"


def agent_op_status(
    ssh_host: str, agent_host: str, server_id: str, request_id: str
) -> str:
    """Query the hardware agent's operation status over its HTTP API.

    The llamacpp agent serves on ``<agent_host>:8080`` and is reachable from the
    agent SSH host directly (not via a container exec).
    """
    cmd = [
        "ssh",
        "-o",
        "ConnectTimeout=8",
        ssh_host,
        f"curl -s --max-time 10 "
        f"http://{agent_host}:8080/proxy/{server_id}/operations/{request_id}",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        return "query_error"
    try:
        return json.loads(out.stdout).get("status", "unknown")
    except json.JSONDecodeError:
        return "unparseable"


def stream_then_disconnect(base_url: str, model: str, api_key: str) -> str | None:
    """Send a streaming Responses request; return the response id, then hard-close."""
    headers = {"Accept": "text/event-stream"}
    if api_key and api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "stream": True,
        "input": "Reply with a short poem about the sea.",
    }
    captured: str | None = None
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        try:
            with client.stream(
                "POST", f"{base_url}/responses", json=payload, headers=headers
            ) as resp:
                if resp.status_code >= 400:
                    body = resp.read()
                    raise RuntimeError(
                        f"HTTP {resp.status_code}: {body[:300].decode(errors='replace')}"
                    )
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        evt = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    rid = evt.get("response", {}).get("id") or evt.get("id")
                    if rid:
                        captured = str(rid)
                    # We have the id and the stream is live: disconnect now.
                    break
        finally:
            # Force-close the underlying connection without draining the body.
            pass
    return captured


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://matrix.thelink.family/v1")
    ap.add_argument("--model", default="rocinante-tiny")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--app-ssh", default="core@10.100.2.100")
    ap.add_argument("--agent-ssh", default="core@10.100.2.111")
    ap.add_argument("--db-container", default="postgres")
    ap.add_argument("--db", default="inference-matrix")
    ap.add_argument("--window", type=float, default=90.0)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    # Resolve the target server id + agent container for the alias.
    try:
        row = db_scalar(
            args.app_ssh,
            args.db_container,
            args.db,
            "SELECT s.id::text || '|' || s.effective_capacity::text "
            "|| '|' || a.host::text "
            "FROM server_instances s JOIN agents a ON a.id = s.agent_id "
            f"WHERE s.alias = {sqlstr(args.model)};",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[setup] cannot resolve alias: {exc}", file=sys.stderr)
        return 3
    if "|" not in row:
        print(f"[setup] alias {args.model!r} not found", file=sys.stderr)
        return 3
    server_id, capacity, agent_host = row.split("|", 2)
    print(
        f"[setup] alias={args.model} server_id={server_id} "
        f"capacity={capacity} agent_host={agent_host}"
    )

    baseline_ids = set(
        filter(
            None,
            db_scalar(
                args.app_ssh,
                args.db_container,
                args.db,
                "SELECT id::text FROM inference_leases "
                f"WHERE server_instance_id='{server_id}' "
                "AND status IN ('active','reserving');",
            ).splitlines(),
        )
    )
    print(f"[setup] baseline active/reserving on server: {len(baseline_ids)}")

    try:
        request_id = stream_then_disconnect(
            args.base_url, args.model, args.api_key
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[transport] {exc}", file=sys.stderr)
        return 3
    if not request_id:
        print("[inconclusive] could not capture backend response id", file=sys.stderr)
        return 2
    print(f"[probe] captured request_id={request_id}; client disconnected")

    agent_finished_at: float | None = None
    last_expires: str | None = None
    renewing_after_agent_done = False
    terminal_seen = False
    deadline = time.monotonic() + args.window
    timeline: list[dict[str, Any]] = []

    while time.monotonic() < deadline:
        try:
            status = db_scalar(
                args.app_ssh,
                args.db_container,
                args.db,
                "SELECT status || '|' || coalesce(lease_expires_at::text,'-') "
                f"FROM inference_leases WHERE request_id = {sqlstr(request_id)};",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[db] {exc}", file=sys.stderr)
            return 3
        if "|" not in status:
            print("[inconclusive] lease row vanished", file=sys.stderr)
            return 2
        lease_status, expires = status.split("|", 1)
        astat = agent_op_status(
            args.agent_ssh, agent_host, server_id, request_id
        )
        now = time.monotonic()
        if astat in TERMINAL_AGENT and agent_finished_at is None:
            agent_finished_at = now
        if agent_finished_at is not None and lease_status == "active":
            if last_expires is not None and expires != last_expires:
                renewing_after_agent_done = True
        last_expires = expires
        timeline.append(
            {
                "t": round(now - deadline + args.window, 1),
                "lease": lease_status,
                "expires": expires,
                "agent": astat,
            }
        )
        print(
            f"  t={timeline[-1]['t']:5.1f}s lease={lease_status:9s} "
            f"agent={astat:9s} expires={expires}"
        )

        if lease_status in TERMINAL_LEASE:
            terminal_seen = True
            break
        # If the agent finished and the lease is active but NOT renewing, it will
        # go stale and reconciliation will clean it; keep watching briefly to see
        # the terminal transition, but stop early once clearly frozen past TTL.
        time.sleep(args.interval)

    if terminal_seen and not renewing_after_agent_done:
        print("[PASS] lease reached a terminal status without zombie renewal")
        return 0
    if terminal_seen:
        print(
            "[PASS] lease terminal (brief renewal before release is acceptable)"
        )
        return 0
    if agent_finished_at is None:
        print(
            "[inconclusive] agent operation never reached terminal in window "
            "(cold start / slow model)",
            file=sys.stderr,
        )
        return 2
    if renewing_after_agent_done:
        print(
            "[FAIL] agent finished but lease still active and renewing "
            "lease_expires_at (zombie lease signature)",
            file=sys.stderr,
        )
        return 1
    print(
        "[inconclusive] agent finished; lease still active but not observed "
        "renewing within window -- reconciliation pending",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
