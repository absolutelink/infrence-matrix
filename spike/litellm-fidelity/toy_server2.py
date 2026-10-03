"""Toy server v2: completed with populated output, failed event, and tool-call flow."""
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

OUT = [
    {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "because"}]},
    {"id": "msg_1", "type": "message", "role": "assistant",
     "content": [{"type": "output_text", "text": "Hi!", "annotations": []}]},
    {"id": "fc_1", "type": "function_call", "name": "get_weather", "arguments": '{"city":"SF"}'},
]

@app.post("/v1/responses")
async def responses(req: Request):
    body = await req.json()
    async def stream():
        yield f"data: {json.dumps({'type':'response.created','sequence_number':1,'response':{'id':'resp_x','object':'response','created_at':1,'status':'in_progress','model':'toy','output':[],'usage':None}})}\n\n"
        if body.get("input") and any(isinstance(i, dict) and i.get("type") == "function_call_output" for i in [body["input"]] if isinstance(body["input"], list)):
            pass
        # second turn: final answer
        yield f"data: {json.dumps({'type':'response.output_text.delta','sequence_number':2,'item_id':'msg_2','output_index':0,'content_index':0,'delta':'Sunny'})}\n\n"
        done = {"id":"resp_x","object":"response","created_at":1,"status":"completed","model":"toy",
               "output": OUT, "usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12}}
        yield f"data: {json.dumps({'type':'response.completed','sequence_number':3,'response':done})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
