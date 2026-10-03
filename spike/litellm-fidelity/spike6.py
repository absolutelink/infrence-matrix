import asyncio, json, litellm, uvicorn
from toy_server3 import app
litellm.register_model({"toy-model": {"supports_native_streaming": True,
    "input_cost_per_token": 0, "output_cost_per_token": 0,
    "litellm_provider": "openai", "mode": "responses"}})
async def main():
    cfg = uvicorn.Config(app, host="127.0.0.1", port=8936, log_level="error")
    s = uvicorn.Server(cfg); t = asyncio.create_task(s.serve())
    while not s.started: await asyncio.sleep(0.05)
    got = []
    try:
        stream = await litellm.aresponses(model="openai/toy-model", input="hi",
            api_base="http://127.0.0.1:8936/v1", custom_llm_provider="openai",
            api_key="fake", stream=True)
        async for e in stream: got.append(e.model_dump(exclude_unset=True))
    except Exception as exc:
        print("ERR:", type(exc).__name__, str(exc)[:200])
    s.should_exit = True; await t
    for g in got:
        print("GOT", g.get("type"), json.dumps(g, default=str)[:250])
asyncio.run(main())
