"""Fidelity spike v2: register model to force native streaming, then diff."""
import asyncio, json, sys, litellm, uvicorn
from toy_server import EVENTS, app

litellm.register_model({
    "toy-model": {
        "supported_openai_params": ["stream", "tools", "tool_choice", "input", "previous_response_id"],
        "supports_native_streaming": True,
        "input_cost_per_token": 0, "output_cost_per_token": 0,
        "litellm_provider": "openai", "mode": "responses",
    }
})

def canonical(ev): return {k: v for k, v in ev.items() if k != "sequence_number"}

async def main():
    config = uvicorn.Config(app, host="127.0.0.1", port=8932, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started: await asyncio.sleep(0.05)

    received = []
    try:
        stream = await litellm.aresponses(
            model="openai/toy-model", input="hi",
            api_base="http://127.0.0.1:8932/v1",
            custom_llm_provider="openai", api_key="fake", stream=True,
        )
        async for event in stream:
            received.append(event.model_dump(exclude_unset=True))
    except Exception as exc:
        print(f"STREAM ERROR: {type(exc).__name__}: {repr(exc)[:300]}")
    server.should_exit = True; await task

    orig = [canonical(e) for e in EVENTS]
    got_canon = [canonical(r) for r in received]
    print(f"\nemitted={len(orig)} received={len(received)}")
    print("\n=== per-event fidelity ===")
    exact = modified = dropped = 0
    for i, e in enumerate(orig):
        r = received[i] if i < len(received) else None
        if r is None or r.get("type") != e["type"]:
            dropped += 1
            print(f"  [{i}] {e['type']}: DROPPED/shifted (got {r.get('type') if r else None})")
        elif canonical(r) == e:
            exact += 1
        else:
            modified += 1
            lost = set(e) - set(r); added = set(r) - set(e)
            diff_fields = []
            for k in set(e) & set(r):
                if e[k] != r[k]:
                    diff_fields.append(k)
            print(f"  [{i}] {e['type']}: MODIFIED lost={sorted(lost)} added={sorted(added)} changed={sorted(diff_fields)}")
    print(f"\nexact={exact} modified={modified} dropped={dropped}")

    # show what a GenericEvent passthrough looks like
    print("\n=== received event types ===")
    for r in received:
        print(" ", r.get("type"))

asyncio.run(main())
