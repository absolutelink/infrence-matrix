import asyncio, json, base64, litellm, uvicorn
from toy_server import app, EVENTS

# add sequence_number to toy events to test passthrough
for i, e in enumerate(EVENTS, 1):
    e.setdefault("sequence_number", i)

litellm.register_model({"toy-model": {
    "supports_native_streaming": True,
    "input_cost_per_token": 0, "output_cost_per_token": 0,
    "litellm_provider": "openai", "mode": "responses"}})

async def main():
    config = uvicorn.Config(app, host="127.0.0.1", port=8934, log_level="error")
    server = uvicorn.Server(config); task = asyncio.create_task(server.serve())
    while not server.started: await asyncio.sleep(0.05)
    received = []
    stream = await litellm.aresponses(model="openai/toy-model", input="hi",
        api_base="http://127.0.0.1:8934/v1", custom_llm_provider="openai",
        api_key="fake", stream=True)
    async for e in stream:
        received.append(e.model_dump(exclude_unset=True))
    server.should_exit = True; await task

    seqs = [r.get("sequence_number") for r in received]
    print("sequence_number preserved:", seqs)

    # test decode of wrapped response id
    rid = received[-1]["response"]["id"]
    inner = rid.removeprefix("resp_")
    print("decoded:", base64.b64decode(inner).decode())

    # check ResponsesAPIRequestUtils decode helper exists
    from litellm.responses.utils import ResponsesAPIRequestUtils as U
    print([m for m in dir(U) if "decode" in m.lower() or "strip" in m.lower()])
asyncio.run(main())
