"""Toy server v3: failed + error event paths."""
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
app = FastAPI()

@app.post("/v1/responses")
async def responses():
    async def stream():
        yield f"data: {json.dumps({'type':'response.created','sequence_number':1,'response':{'id':'resp_f','object':'response','created_at':1,'status':'in_progress','model':'toy','output':[],'usage':None}})}\n\n"
        yield f"data: {json.dumps({'type':'response.failed','sequence_number':2,'response':{'id':'resp_f','object':'response','created_at':1,'status':'failed','model':'toy','output':[],'error':{'code':'server_error','message':'backend exploded'}}})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/v1/responses_err")
async def responses_err():
    async def stream():
        yield f"data: {json.dumps({'type':'error','sequence_number':1,'message':'stream broke','code':'broken'})}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
