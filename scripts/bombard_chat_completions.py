#!/usr/bin/env python3
"""Soak/bombard probe for the production OpenAI-compatible chat endpoint.

What it does
------------
Fires a continuously growing, multi-turn conversation at ``POST
/v1/chat/completions`` -- optionally several conversations in parallel -- and
keeps going until *one turn stalls*, then stops the whole run and reports which
turn broke and why.

Each turn appends the assistant's reply to the running history and sends a fresh
user message, so the prompt (and the prefill / context pressure on the serving
llama.cpp instance) grows monotonically. This is designed to surface the failure
modes that only appear under sustained, context-heavy load:

  * a turn that never produces its first token (cold start that never becomes
    ready, or a dispatch that never reaches the LLM),
  * a stall mid-stream: tokens were flowing and then stop for longer than
    ``--inter-token-stall`` (upstream wedged, lease stuck, agent unreachable),
  * a stream that ends without ``[DONE]`` (truncated / dropped connection),
  * an HTTP 5xx or transport error surfaced as a stall.

Keep-alive comments (``: keep-alive``) are intentionally NOT treated as
progress -- the backend emits them every ~15s while genuinely idle upstream, so
only real content deltas reset the inter-token clock.

Stall thresholds
----------------
  * ``--first-token-timeout``: budget before the first content delta of a turn.
    Generous, because it must cover a cold start plus a long prefill of the
    (growing) context.
  * ``--inter-token-stall``: once a turn has produced at least one token, the
    maximum tolerable gap between consecutive content deltas.
  * ``--turn-timeout``: hard wall-clock cap on a single turn.

Exit codes
----------
  0  NO STALL -- ran to ``--max-turns``/``--max-duration`` with every turn OK.
  1  STALL    -- a turn stalled/errored/truncated. This is the intended
                 stopping condition of the bombardment.
  3  SETUP    -- the very first request could not reach the endpoint at all
                 (connection refused / DNS / bad URL), so nothing was exercised.

Usage
-----
  # Default: 4 parallel growing chats against production until one stalls.
  uv run python scripts/bombard_chat_completions.py

  # Single ongoing chat, tighter stall, cap the whole run at 10 minutes.
  uv run python scripts/bombard_chat_completions.py \
      --concurrency 1 --inter-token-stall 20 --max-duration 600

  # Point at a local backend and a specific model alias.
  uv run python scripts/bombard_chat_completions.py \
      --base-url http://localhost:8000/v1 --model rocinante

  # Dry run: show the resolved request for turn 1 and exit.
  uv run python scripts/bombard_chat_completions.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

_VOCAB = (  # noqa: SIM905
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
    "yankee zulu harbor meadow canyon lantern harvest quiet silver thunder willow"
).split()

# A neutral, always-answerable thread so the model never refuses and we keep
# getting real tokens back to measure against.
_TURN_PROMPTS = (
    (
        "Count with me. Say the next number after the highest one I have heard so far, "
        "starting at zero. Reply with just the number."
    ),
    "Now the next number.",
    "And the next.",
)


@dataclass
class TurnResult:
    conv_id: int
    turn: int
    ok: bool = False
    stalled: bool = False
    reason: str = ""
    status: int | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    content_chars: int = 0
    content: str = ""
    request_id: str = ""
    started_at: float = field(default_factory=time.monotonic)


class StallSignal(Exception):
    """Raised internally to unwind a stalled turn out of the streaming loop."""


def _user_message(turn: int, rnd: random.Random, grow_tokens: int) -> str:
    base = _TURN_PROMPTS[min(turn, len(_TURN_PROMPTS) - 1)]
    if grow_tokens <= 0:
        return base
    # A block of novel filler per turn forces the context (and prefill) to grow
    # monotonically instead of hitting a cached prefix every time.
    filler = " ".join(rnd.choice(_VOCAB) for _ in range(grow_tokens))
    return f"{base}\nContext notes (ignore for the answer): {filler}"


def _build_payload(
    args: argparse.Namespace,
    history: list[dict[str, str]],
    turn: int,
    rnd: random.Random,
) -> dict[str, Any]:
    messages = list(history)
    messages.append(
        {"role": "user", "content": _user_message(turn, rnd, args.grow_tokens)}
    )
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    return payload


def _extract_delta(obj: dict[str, Any]) -> str:
    try:
        choices = obj.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        return content if isinstance(content, str) else ""
    except (AttributeError, IndexError, TypeError):
        return ""


async def _run_turn(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    conv_id: int,
    turn: int,
    args: argparse.Namespace,
) -> TurnResult:
    res = TurnResult(conv_id=conv_id, turn=turn)
    started = time.monotonic()
    # Stamp a unique, backend-parseable inference request id so this turn can be
    # grepped in backend logs (backend_inference_stream_open/close, lease
    # acquire/release) even when the client never sees response headers.
    request_id = f"chatcmpl-{uuid.uuid4()}"
    res.request_id = request_id
    turn_headers = {**headers, "X-Inference-Request-ID": request_id}
    try:
        async with client.stream(
            "POST", url, json=payload, headers=turn_headers
        ) as resp:
            res.status = resp.status_code
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")[:600]
                res.stalled = True
                res.reason = f"HTTP {resp.status_code}: {body}"
                return res
            # Absolute deadline for the next *content* token. Only real content
            # deltas advance it; keep-alive comments and role-only chunks do NOT,
            # so a stream that only pings keep-alives forever is correctly a stall.
            content_deadline = started + args.first_token_timeout
            line_iter = resp.aiter_lines().__aiter__()
            while True:
                now = time.monotonic()
                turn_remaining = args.turn_timeout - (now - started)
                if turn_remaining <= 0:
                    res.stalled = True
                    res.reason = f"turn exceeded {args.turn_timeout:g}s wall clock"
                    return res
                wait = min(content_deadline - now, turn_remaining)
                if wait <= 0:
                    # Already past the content deadline before even waiting.
                    if res.first_token_at is None:
                        res.stalled = True
                        res.reason = (
                            f"no first content token within {args.first_token_timeout:g}s "
                            "(cold start never became ready or dispatch never reached LLM)"
                        )
                    else:
                        gap = now - res.last_token_at
                        res.stalled = True
                        res.reason = (
                            f"inter-token gap {gap:.1f}s > "
                            f"{args.inter_token_stall:g}s after tokens started"
                        )
                    return res
                try:
                    line = await asyncio.wait_for(line_iter.__anext__(), timeout=wait)
                except asyncio.TimeoutError:
                    if res.first_token_at is None:
                        res.stalled = True
                        res.reason = (
                            f"no first content token within {args.first_token_timeout:g}s "
                            "(cold start never became ready or dispatch never reached LLM)"
                        )
                    else:
                        gap = time.monotonic() - res.last_token_at
                        res.stalled = True
                        res.reason = (
                            f"inter-token gap {gap:.1f}s > "
                            f"{args.inter_token_stall:g}s after tokens started"
                        )
                    return res
                except StopAsyncIteration:
                    # Stream closed without [DONE].
                    if res.content_chars > 0:
                        res.stalled = True
                        res.reason = "stream closed without [DONE] (truncated)"
                    else:
                        res.stalled = True
                        res.reason = "stream closed with no data and no [DONE]"
                    return res
                if not line:
                    continue
                if line.startswith(":"):
                    # keep-alive comment: upstream idle, NOT progress.
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    if res.content_chars == 0:
                        res.stalled = True
                        res.reason = "stream ended [DONE] with no content tokens"
                    else:
                        res.ok = True
                    return res
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    res.stalled = True
                    res.reason = f"stream error payload: {obj['error']}"
                    return res
                piece = _extract_delta(obj)
                if piece:
                    now = time.monotonic()
                    if res.first_token_at is None:
                        res.first_token_at = now
                    res.last_token_at = now
                    # Advance the content budget for the *next* token.
                    content_deadline = now + args.inter_token_stall
                    res.content += piece
                    res.content_chars += len(piece)
    except httpx.HTTPError as exc:
        res.stalled = True
        res.reason = f"transport error: {type(exc).__name__}: {exc}"
        return res


async def _conversation(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    conv_id: int,
    args: argparse.Namespace,
    stop: asyncio.Event,
) -> TurnResult | None:
    """Drive one growing multi-turn chat until it stalls or global stop."""
    history: list[dict[str, str]] = []
    rnd = random.Random(hash((conv_id, args.seed)) & 0xFFFFFFFF)
    turn = 0
    while not stop.is_set():
        if args.max_turns and turn >= args.max_turns:
            return None
        payload = _build_payload(args, history, turn, rnd)
        res = await _run_turn(client, url, headers, payload, conv_id, turn, args)
        if res.stalled:
            return res
        if not res.ok:
            return res
        # Grow the ongoing conversation with the real assistant reply.
        history.append({"role": "user", "content": payload["messages"][-1]["content"]})
        history.append({"role": "assistant", "content": res.content})
        turn += 1
        if args.heartbeat:
            ft = (
                f"{res.first_token_at - res.started_at:.1f}s"
                if res.first_token_at
                else "n/a"
            )
            dur = (res.last_token_at or time.monotonic()) - res.started_at
            print(
                f"[conv {conv_id}] turn {turn} ok: {res.content_chars} chars "
                f"first_token={ft} dur={dur:.1f}s ctx_msgs={len(history)}",
                flush=True,
            )
        if args.turn_delay:
            await asyncio.sleep(args.turn_delay)
    return None


async def _async_main(args: argparse.Namespace) -> int:
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Accept": "text/event-stream"}
    if args.api_key and args.api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {args.api_key}"

    if args.dry_run:
        rnd = random.Random(args.seed)
        print(json.dumps(_build_payload(args, [], 0, rnd), indent=2))
        return 0

    # Health/first-connection probe: distinguish a totally unreachable endpoint
    # (setup error) from a stall under load.
    print(f"[bombard] probing {url} (model={args.model}) ...", flush=True)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(args.probe_timeout, connect=args.connect_timeout),
            verify=not args.insecure,
        ) as probe:
            r = await probe.post(
                url,
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "max_tokens": 1,
                },
                headers=headers,
            )
            if r.status_code >= 500:
                print(
                    f"[SETUP] endpoint returned HTTP {r.status_code} before load: "
                    f"{r.text[:400]}",
                    file=sys.stderr,
                )
                return 3
            print(f"[bombard] probe OK (HTTP {r.status_code})", flush=True)
    except httpx.HTTPError as exc:
        print(
            f"[SETUP] cannot reach {url}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3

    stop = asyncio.Event()
    wall_start = time.monotonic()
    results: asyncio.Queue[TurnResult] = asyncio.Queue()

    async def worker(conv_id: int) -> None:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(args.turn_timeout, connect=args.connect_timeout),
                limits=httpx.Limits(
                    max_connections=args.concurrency + 2,
                    max_keepalive_connections=args.concurrency + 2,
                ),
                verify=not args.insecure,
            ) as client:
                stalled = await _conversation(client, url, headers, conv_id, args, stop)
                if stalled is not None:
                    await results.put(stalled)
                    stop.set()
        except httpx.HTTPError as exc:
            await results.put(
                TurnResult(
                    conv_id=conv_id,
                    turn=-1,
                    stalled=True,
                    reason=f"worker transport error: {type(exc).__name__}: {exc}",
                )
            )
            stop.set()

    print(
        f"[bombard] url={url} model={args.model} concurrency={args.concurrency} "
        f"max_turns={args.max_turns or 'unlimited'} max_duration={args.max_duration:g}s "
        f"first_token_timeout={args.first_token_timeout:g}s "
        f"inter_token_stall={args.inter_token_stall:g}s",
        flush=True,
    )

    tasks = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]

    async def watchdog() -> None:
        while not stop.is_set():
            if time.monotonic() - wall_start >= args.max_duration:
                stop.set()
                return
            await asyncio.sleep(0.5)

    wd = asyncio.create_task(watchdog())

    # Wait for the first stall or for all workers to finish normally.
    _done, pending = await asyncio.wait(
        [*tasks, wd], return_when=asyncio.FIRST_COMPLETED
    )
    # Give any just-enqueued stall a moment to land in the queue.
    await asyncio.sleep(0.05)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    stalled: TurnResult | None = None
    while not results.empty():
        cand = results.get_nowait()
        if cand.stalled:
            stalled = cand
            break

    elapsed = time.monotonic() - wall_start
    if stalled is not None:
        print(
            f"[STALL] conv={stalled.conv_id} turn={stalled.turn} "
            f"status={stalled.status} after {elapsed:.1f}s\n"
            f"        request_id={stalled.request_id}\n"
            f"        reason: {stalled.reason}",
            file=sys.stderr,
        )
        if args.verbose and stalled.content:
            print(
                f"        partial content: {stalled.content[:300]!r}", file=sys.stderr
            )
        return 1

    print(
        f"[NO STALL] completed {elapsed:.1f}s across {args.concurrency} chat(s) "
        f"without a stall (max_turns={args.max_turns or 'unlimited'}, "
        f"max_duration={args.max_duration:g}s reached).",
        flush=True,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    ap.add_argument("--concurrency", type=int, default=4, help="parallel growing chats")
    ap.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="turns per chat before that chat stops (0 = unlimited until stall)",
    )
    ap.add_argument(
        "--max-duration",
        type=float,
        default=1800.0,
        help="overall wall-clock cap for the whole bombardment (seconds)",
    )
    ap.add_argument(
        "--grow-tokens",
        type=int,
        default=120,
        help="filler words added per user turn to grow context/prefill",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="max_tokens per completion (output size per turn)",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--first-token-timeout",
        type=float,
        default=180.0,
        help="budget for the first content delta (covers cold start + prefill)",
    )
    ap.add_argument(
        "--inter-token-stall",
        type=float,
        default=30.0,
        help="max gap between content deltas once streaming has begun",
    )
    ap.add_argument(
        "--turn-timeout",
        type=float,
        default=600.0,
        help="hard wall-clock cap on a single turn (httpx read timeout too)",
    )
    ap.add_argument(
        "--connect-timeout", type=float, default=30.0, help="TCP connect timeout"
    )
    ap.add_argument(
        "--probe-timeout",
        type=float,
        default=120.0,
        help="wall clock cap on the pre-load health probe (cold start included)",
    )
    ap.add_argument(
        "--turn-delay",
        type=float,
        default=0.0,
        help="sleep between turns per chat (paces the bombardment)",
    )
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed for filler")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification")
    ap.add_argument(
        "--heartbeat",
        action="store_true",
        help="log every completed turn (progress vs wedge); implies per-turn visibility",
    )
    ap.add_argument("--verbose", action="store_true", help="partial content on stall")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the turn-1 request and exit"
    )
    args = ap.parse_args(argv)
    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("\n[bombard] interrupted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
