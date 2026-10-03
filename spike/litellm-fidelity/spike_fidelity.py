"""Run: uv run python spike_fidelity.py

Starts the toy server, points litellm.aresponses at it, records every event
that survives litellm's parse/re-serialization, and diffs against the
original emitted events.
"""

import asyncio
import json
import sys

import httpx
import litellm
import uvicorn

from toy_server import EVENTS, app


def canonical(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if k != "sequence_number"}


async def main() -> int:
    config = uvicorn.Config(app, host="127.0.0.1", port=8931, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    received: list[dict] = []
    try:
        stream = await litellm.aresponses(
            model="openai/toy-model",
            input="hi",
            api_base="http://127.0.0.1:8931/v1",
            custom_llm_provider="openai",
            api_key="fake",
            stream=True,
        )
        async for event in stream:
            received.append(event.model_dump(exclude_unset=True))
    except Exception as exc:  # noqa: BLE001
        print(f"LITELLM STREAM ERROR: {type(exc).__name__}: {exc}")

    server.should_exit = True
    await task

    orig = [canonical(e) for e in EVENTS]
    got_types = [e.get("type") for e in received]

    print("\n=== EVENTS EMITTED BY TOY SERVER ===")
    for e in orig:
        print(" ", e["type"])
    print("\n=== EVENTS SURVIVING LITELLM ===")
    for e in received:
        print(" ", e.get("type"))

    missing = [e for e in orig if e not in [canonical(r) for r in received]]
    print("\n=== DIFF: ORIGINAL EVENTS NOT PRESERVED VERBATIM ===")
    got_canon = [canonical(r) for r in received]
    for e in orig:
        if e not in got_canon:
            print(json.dumps(e, indent=2)[:400])
            match = next((r for r in received if r.get("type") == e["type"]), None)
            if match:
                print("   ^ survived with modifications as:", json.dumps(match, default=str)[:400])
            else:
                print("   ^ DROPPED ENTIRELY")

    # field-level loss check for delta events
    print("\n=== FIELD-LEVEL CHECK (deltas) ===")
    for i, e in enumerate(orig):
        r = received[i] if i < len(received) else {}
        if r.get("type") != e["type"]:
            print(f"  POSITION MISMATCH at {i}: expected {e['type']} got {r.get('type')}")
            continue
        lost = set(e) - set(r)
        added = set(r) - set(e)
        if lost or added:
            print(f"  {e['type']}: lost={sorted(lost)} added={sorted(added)}")

    print(f"\nSUMMARY: emitted={len(orig)} received={len(received)}")
    return 0 if len(received) >= len(orig) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
