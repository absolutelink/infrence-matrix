"""Toy OpenResponses-spec SSE server emitting the full event set we care about.

Serves POST /v1/responses with streaming SSE covering:
- lifecycle: response.created / in_progress / completed
- reasoning (summary + dual-name reasoning_text events)
- output_item.added/done, content_part.added/done
- output_text.delta/done, annotation.added
- refusal.delta/done
- function_call_arguments.delta/done
- response.reasoning_text.delta (OpenWebUI dual-name, non-canonical)
- unknown custom event type (tests GenericEvent passthrough)
"""

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

RESPONSE_OBJ = {
    "id": "resp_toy123",
    "object": "response",
    "created_at": 1760000000,
    "status": "in_progress",
    "model": "toy-model",
    "output": [],
    "usage": None,
}


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


EVENTS = [
    {"type": "response.created", "response": RESPONSE_OBJ},
    {"type": "response.in_progress", "response": RESPONSE_OBJ},
    # reasoning item
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": "rs_1", "type": "reasoning", "summary": [], "content": []},
    },
    {
        "type": "response.reasoning_summary_part.added",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "part": {"type": "summary_text", "text": ""},
    },
    {
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "delta": "Thinking hard",
    },
    {
        "type": "response.reasoning_summary_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "text": "Thinking hard",
    },
    {
        "type": "response.reasoning_summary_part.done",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "part": {"type": "summary_text", "text": "Thinking hard"},
    },
    # dual-name reasoning text (OpenWebUI compat)
    {
        "type": "response.reasoning_text.delta",
        "item_id": "rs_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "raw chain of thought",
    },
    {
        "type": "response.reasoning_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "content_index": 0,
        "text": "raw chain of thought",
    },
    {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Thinking hard"}],
        },
    },
    # message item
    {
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {"id": "msg_1", "type": "message", "role": "assistant", "content": []},
    },
    {
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    },
    {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "delta": "Hello ",
    },
    {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "delta": "world",
    },
    {
        "type": "response.output_text.annotation.added",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "annotation_index": 0,
        "annotation": {"type": "url_citation", "url": "http://x", "title": "t", "start_index": 0, "end_index": 5},
    },
    {
        "type": "response.output_text.done",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "text": "Hello world",
    },
    {
        "type": "response.content_part.done",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "part": {"type": "output_text", "text": "Hello world", "annotations": []},
    },
    {
        "type": "response.output_item.done",
        "output_index": 1,
        "item": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello world", "annotations": []}],
        },
    },
    # refusal
    {
        "type": "response.refusal.delta",
        "item_id": "msg_1",
        "output_index": 2,
        "content_index": 0,
        "delta": "no",
    },
    {
        "type": "response.refusal.done",
        "item_id": "msg_1",
        "output_index": 2,
        "content_index": 0,
        "text": "no",
    },
    # function call
    {
        "type": "response.output_item.added",
        "output_index": 3,
        "item": {"id": "fc_1", "type": "function_call", "name": "get_weather", "arguments": ""},
    },
    {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_1",
        "output_index": 3,
        "delta": '{"city":',
    },
    {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_1",
        "output_index": 3,
        "delta": '"SF"}',
    },
    {
        "type": "response.function_call_arguments.done",
        "item_id": "fc_1",
        "output_index": 3,
        "arguments": '{"city":"SF"}',
    },
    {
        "type": "response.output_item.done",
        "output_index": 3,
        "item": {"id": "fc_1", "type": "function_call", "name": "get_weather", "arguments": '{"city":"SF"}'},
    },
    # completed with usage
    {
        "type": "response.completed",
        "response": {
            **RESPONSE_OBJ,
            "status": "completed",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 6},
            },
        },
    },
]


async def stream():
    for e in EVENTS:
        yield sse(e)
        await asyncio.sleep(0.01)
    yield "data: [DONE]\n\n"


@app.post("/v1/responses")
async def responses():
    return StreamingResponse(stream(), media_type="text/event-stream")
