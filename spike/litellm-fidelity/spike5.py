import asyncio, json, litellm, uvicorn
from toy_server2 import app
litellm.register_model({"toy-model": {"supports_native_streaming": True,
    "input_cost_per_token": 0, "output_cost_per_token": 0,
    "litellm_provider": "openai", "mode": "responses"}})

async def main():
    cfg = uvicorn.Config(app, host="127.0.0.1", port=8935, log_level="error")
    s = uvicorn.Server(cfg); t = asyncio.create_task(s.serve())
    while not s.started: await asyncio.sleep(0.05)
    got = []
    stream = await litellm.aresponses(model="openai/toy-model", input="hi",
        api_base="http://127.0.0.1:8935/v1", custom_llm_provider="openai",
        api_key="fake", stream=True,
        tools=[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}])
    async for e in stream: got.append(e.model_dump(exclude_unset=True))
    s.should_exit = True; await t
    comp = got[-1]["response"]
    print("output preserved:", json.dumps(comp["output"], default=str)[:600])
    print("num output items:", len(comp["output"]))
    print("types:", [o.get("type") for o in comp["output"]])
asyncio.run(main())
