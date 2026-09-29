#!/usr/bin/env python3
"""Remote regression probe for Responses client-visible SSE idle gaps.

Bug (fixed in responses/router.py): the idle keepalive used
``asyncio.wait_for`` on the pending upstream ``anext()``, which *cancels and
closes* the upstream async generator on the first gap longer than
``SSE_KEEPALIVE_INTERVAL_SECONDS`` (15s). The router then treated the
resulting ``StopAsyncIteration`` as a normal end-of-stream and still emitted
``response.completed`` -- so a body read stalled >15s could lose its output
while looking successful. Header waits and discarded upstream lines are also
possible sources of client-visible silence.

This script streams a large-prompt (long-prefill) request against a deployed
backend and inspects the SSE frames:

  * A client-visible gap beyond the 15s keepalive interval plus 3s of network
    and scheduling tolerance after initial lifecycle events is a FAIL, even
    if the response eventually succeeds. This probe cannot distinguish backend
    silence from buffering in an intermediary.
  * After a keepalive, the stream must reach a terminal event with non-empty
    assistant output following the comment (C3 truncation regression).
  * No comments and no long client-visible gap is INCONCLUSIVE.

Exit codes:
  0  PASS - keepalive observed and output followed it with a terminal event
  1  FAIL - client-visible silence or truncated output after a keepalive
  2  INCONCLUSIVE - no keepalive or long client-visible gap observed
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
import random
import ssl
import sys
import time
import uuid
from typing import Any

import httpx

TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}
KEEPALIVE_SECONDS = 15.0
MAX_VISIBLE_IDLE_SECONDS = KEEPALIVE_SECONDS + 3.0

_VOCAB = (  # noqa: SIM905
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
    "yankee zulu harbor meadow canyon lantern harvest quiet silver thunder willow"
).split()


def _build_request(model: str, max_tokens: int, filler_repeat: int) -> dict[str, Any]:
    # Novel content per run avoids reusing a long cached prefix. A unique
    # nonce + randomized sentences encourage a longer prefill, but cannot
    # guarantee that the client sees an idle window on every run.
    nonce = uuid.uuid4().hex
    rnd = random.Random(nonce)
    lines = []
    for i in range(filler_repeat):
        words = " ".join(rnd.choice(_VOCAB) for _ in range(12))
        lines.append(f"{nonce}-{i}: {words}")
    filler = "\n".join(lines)
    prompt = (
        "Read this encoded log of random words and ignore its meaning. "
        f"{filler}\n\n"
        "After that long passage, reply with exactly one short sentence "
        "summarizing that foxes can run."
    )
    return {
        "model": model,
        "input": [{"type": "message", "role": "user", "content": prompt}],
        "stream": True,
        "max_output_tokens": max_tokens,
        "store": False,
    }


def classify(
    events: list[str],
    text_after_keepalive: int,
    keepalive_count: int,
    max_active_gap: float,
) -> tuple[int, str]:
    if not events:
        return 3, "no SSE events received at all"
    terminal = {"response.completed", "response.incomplete"} & set(events)
    if max_active_gap > MAX_VISIBLE_IDLE_SECONDS:
        return 1, (
            f"client-visible silence of {max_active_gap:.1f}s after initial events "
            f"exceeded {MAX_VISIBLE_IDLE_SECONDS:g}s (15s keepalive + 3s tolerance); "
            "backend vs intermediary buffering "
            "cannot be determined by this probe"
        )
    if "response.failed" in events:
        return 3, "response.failed terminal event (see --verbose payload)"
    if not terminal:
        return 1, "stream ended without a completed/incomplete terminal event"
    if keepalive_count == 0:
        return (
            2,
            (
                "no keepalive or long client-visible gap observed; idle handling "
                "was not exercised (increase --filler-repeat to try a longer prefill)"
            ),
        )
    if text_after_keepalive == 0:
        return 1, (
            f"keepalive fired ({keepalive_count}x) and stream terminated but "
            "no output_text.delta chars followed it -> possible truncation"
        )
    return 0, (
        f"keepalive fired ({keepalive_count}x) and {text_after_keepalive} output "
        "chars followed it with a clean terminal event"
    )


def run(args: argparse.Namespace) -> int:
    url = args.base_url.rstrip("/") + "/responses"
    body = _build_request(args.model, args.max_tokens, args.filler_repeat)
    headers = {"Accept": "text/event-stream"}
    if args.api_key and args.api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {args.api_key}"

    events: list[str] = []
    text_len = 0
    text_after_keepalive = 0
    reasoning_len = 0
    keepalive_count = 0
    max_gap = 0.0
    max_active_gap = 0.0
    saw_initial_events = False
    started_at = time.monotonic()
    last_byte_ts = started_at
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
                if now - started_at >= args.timeout:
                    print(
                        f"[ERROR] request exceeded {args.timeout:g}s overall timeout",
                        file=sys.stderr,
                    )
                    return 3
                gap = now - last_byte_ts
                max_gap = max(max_gap, gap)
                if saw_initial_events:
                    max_active_gap = max(max_active_gap, gap)
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
                    if events[-1] in {"response.created", "response.in_progress"}:
                        saw_initial_events = True
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
                        length = len(obj.get("delta", ""))
                        text_len += length
                        if keepalive_count:
                            text_after_keepalive += length
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

    code, msg = classify(events, text_after_keepalive, keepalive_count, max_active_gap)
    label = {0: "PASS", 1: "FAIL", 2: "INCONCLUSIVE", 3: "ERROR"}[code]
    print(f"[{label}] {msg}")
    print(
        f"    events={len(events)} keepalives={keepalive_count} "
        f"output_chars={text_len} reasoning_chars={reasoning_len} "
        f"max_inter_line_gap={max_gap:.1f}s "
        f"max_gap_after_initial_events={max_active_gap:.1f}s saw_done={saw_done}"
    )
    print(f"    terminal={sorted(set(events) & TERMINAL_EVENTS)}")
    if args.verbose and verbose_payloads:
        print("    sample payloads:")
        for p in verbose_payloads[:5]:
            print(f"      {p[:200]}")
    if code == 2:
        print(
            "    HINT: no client-visible idle was measured; re-run with a "
            "larger --filler-repeat to exercise keepalives.",
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
        help="number of randomized filler lines (raise to encourage a longer stall)",
    )
    ap.add_argument(
        "--read-timeout",
        type=float,
        default=60.0,
        help="per-read client timeout; must exceed the 15s keepalive interval",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="overall request cap, checked as SSE lines arrive",
    )
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification")
    ap.add_argument("--verbose", action="store_true", help="dump keepalive + samples")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
