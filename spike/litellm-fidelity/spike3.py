import asyncio, json, litellm, uvicorn
from toy_server import EVENTS, app

litellm.register_model({"toy-model": {
    "supports_native_streaming": True,
    "input_cost_per_token": 0, "output_cost_per_token": 0,
    "litellm_provider": "openai", "mode": "responses"}})

async def main():
    config = uvicorn.Config(app, host="127.0.0.1", port=8933, log_level="error")
    server = uvicorn.Server(config); task = asyncio.create_task(server.serve())
    while not server.started: await asyncio.sleep(0.05)
    received = []
    stream = await litellm.aresponses(model="openai/toy-model", input="hi",
        api_base="http://127.0.0.1:8933/v1", custom_llm_provider="openai",
        api_key="fake", stream=True)
    async for e in stream:
        received.append(e.model_dump(exclude_unset=True))
    server.should_exit = True; await task

    for idx in (0, 25):
        got = received[idx].get("response")
        sent = EVENTS[idx]["response"]
        print(f"\n=== {received[idx]['type']} response diff ===")
        print("SENT:", json.dumps(sent, default=str))
        print("GOT :", json.dumps(got, default=str))
asyncio.run(main())
