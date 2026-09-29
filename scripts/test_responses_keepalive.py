#!/usr/bin/env python3
"""Remote regression probe for C3: SSE keepalive must not truncate the stream.

Bug (fixed in responses/router.py): the idle keepalive used
``asyncio.wait_for`` on the pending upstream ``anext()``, which *cancels and
closes* the upstream async generator on the first gap longer than
``SSE_KEEPALIVE_INTERVAL_SECONDS`` (15s). The router then treated the
resulting ``StopAsyncIteration`` as a normal end-of-stream and still emitted
``response.completed`` -- so a response whose prefill stalled >15s lost all
of its output while looking successful.

This script streams a large-prompt (long-prefill) request against a deployed
backend and inspects the SSE frames:

  * If at least one keepalive comment (": ...") fired, the >15s idle gap was
    exercised. In that case the stream MUST still reach a terminal
    response.completed / response.incomplete with non-empty assistant text.
    Empty output after a keepalive == C3 reproduced == FAIL.
  * If no keepalive fired, the idle window was never reached on this hardware
    (model responded too fast / prompt too short). The bug cannot be
    exercised remotely -> INCONCLUSIVE (exit 2), NOT a pass.

Exit codes:
  0  PASS - keepalive exercised and stream completed with real output
  1  FAIL - keepalive exercised but stream truncated (C3 signature)
  2  INCONCLUSIVE - no idle gap reached (raise --filler-repeat / --max-tokens)
  3  connection / HTTP / configuration error

Usage:
  uv run python scripts/test_responses_keepalive.py \
      --base-url https://matrix.thelink.family/v1 \
      --model rocinante

  # Force a longer prefill stall if the first run is inconclusive:
  uv run python scripts/test_responses_keepalive.py --filler-repeat 400
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
from typing import Any

import httpx

TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}


def _build_request(model: str, max_tokens: int, filler_repeat: int) -> dict[str, Any]:
    filler = (
        "The quick brown fox jumps over the lazy dog while the rain falls softly. "
    ) * filler_repeat
    prompt = (
        f"{filler}\n\n"
        "After that long passage, reply with exactly one short sentence "
        "summarizing that foxes can run."
    )
    return {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_output_tokens": max_tokens,
        "store": False,
    }


def classify(events: list[str], text_len: int, keepalive_count: int) -> tuple[int, str]:
    if not events:
        return 3, "no SSE events received at all"
    if "response.failed" in events:
        return 3, "response.failed terminal event (see --verbose payload)"
    terminal = {"response.completed", "response.incomplete"} & set(events)
    if keepalive_count == 0:
        # Idle gap never reached -> C3 not exercised remotely.
        return (
            2,
            (
                "no keepalive comment observed; the >15s idle gap was not reached. "
                "Increase --filler-repeat (or --max-tokens) to force a longer prefill."
            ),
        )
    if not terminal:
        return 1, (
            f"keepalive fired ({keepalive_count}x) but no terminal "
            f"completed/incomplete event -> stream truncated mid-flight"
        )
    if text_len == 0:
        return 1, (
            f"keepalive fired ({keepalive_count}x) and stream terminated but "
            f"zero output_text.delta chars arrived -> C3 truncation reproduced"
        )
    return 0, (
        f"keepalive fired ({keepalive_count}x) and {text_len} output chars "
        f"arrived after the idle gap with a clean terminal event"
    )


def run(args: argparse.Namespace) -> int:
    url = args.base_url.rstrip("/") + "/responses"
    body = _build_request(args.model, args.max_tokens, args.filler_repeat)
    headers = {"Accept": "text/event-stream"}
    if args.api_key and args.api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {args.api_key}"

    events: list[str] = []
    text_len = 0
    reasoning_len = 0
    keepalive_count = 0
    max_gap = 0.0
    last_byte_ts = time.monotonic()
    saw_done = False
    verbose_payloads: list[str] = []

    ctx = ssl.create_default_context()
    if args.insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    timeout = httpx.Timeout(args.timeout, connect=30.0, read=args.read_timeout)
    try:
        with (
            httpx.Client(verify=ctx, timeout=timeout) as client,
            client.stream("POST", url, json=body, headers=headers) as resp,
        ):
            if resp.status_code >= 400:
                detail = resp.read().decode("utf-8", "replace")[:800]
                print(
                    f"[ERROR] HTTP {resp.status_code} from {url}\n{detail}",
                    file=sys.stderr,
                )
                return 3
            for line in resp.iter_lines():
                now = time.monotonic()
                max_gap = max(max_gap, now - last_byte_ts)
                last_byte_ts = now
                if not line:
                    continue
                if line.startswith(":"):
                    keepalive_count += 1
                    if args.verbose:
                        print(f"  keepalive: {line!r}")
                    continue
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    etype = obj.get("type", "")
                    if etype == "response.output_text.delta":
                        text_len += len(obj.get("delta", ""))
                    elif etype in (
                        "response.reasoning.delta",
                        "response.reasoning_summary_text.delta",
                    ):
                        reasoning_len += len(obj.get("delta", ""))
                    if args.verbose and etype not in events[:2]:
                        verbose_payloads.append(data)
    except httpx.HTTPError as exc:
        print(f"[ERROR] transport error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    code, msg = classify(events, text_len, keepalive_count)
    label = {0: "PASS", 1: "FAIL", 2: "INCONCLUSIVE", 3: "ERROR"}[code]
    print(f"[{label}] {msg}")
    print(
        f"    events={len(events)} keepalives={keepalive_count} "
        f"output_chars={text_len} reasoning_chars={reasoning_len} "
        f"max_inter_line_gap={max_gap:.1f}s saw_done={saw_done}"
    )
    print(f"    terminal={sorted(set(events) & TERMINAL_EVENTS)}")
    if args.verbose and verbose_payloads:
        print("    sample payloads:")
        for p in verbose_payloads[:5]:
            print(f"      {p[:200]}")
    if code == 2:
        print(
            "    HINT: prefill finished under the 15s keepalive interval; "
            "re-run with a larger --filler-repeat to force the idle gap.",
            file=sys.stderr,
        )
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, add_help=True)
    ap.add_argument(
        "--base-url",
        default="https://matrix.thelink.family/v1",
        help="OpenAI-compatible base URL ending in /v1",
    )
    ap.add_argument("--model", default="rocinante", help="model/alias to hit")
    ap.add_argument(
        "--api-key",
        default="none",
        help="bearer token; 'none' to omit (matrix auth is disabled)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=48,
        help="small output so prefill dominates the wall time",
    )
    ap.add_argument(
        "--filler-repeat",
        type=int,
        default=250,
        help="repeat count for the filler sentence (raise to force a longer stall)",
    )
    ap.add_argument(
        "--read-timeout",
        type=float,
        default=60.0,
        help="per-read client timeout; must exceed the 15s keepalive interval",
    )
    ap.add_argument("--timeout", type=float, default=300.0, help="overall request cap")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification")
    ap.add_argument("--verbose", action="store_true", help="dump keepalive + samples")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
