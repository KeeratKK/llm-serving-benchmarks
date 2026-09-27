import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time
import uuid
import json
import asyncio

TTFT_DELAY = float(os.environ.get("TTFT_DELAY", "0.2")) # Time To First Token Delay
ITL_DELAY = float(os.environ.get("ITL_DELAY", "0.02")) # Inter Token Delay

app = FastAPI()

class CompletionRequest(BaseModel):
    model: str
    prompt: list[int]
    max_tokens: int
    stream: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    temperature: float | None = None
    seed: int | None = None
    stream_options: dict | None = None

async def stream_generator(request: CompletionRequest):

    request_id = f"cmpl-{uuid.uuid4().hex}"

    # Mocking the decode steps of once per generated token with ITL delay in between
    chunks_sent = 0
    for idx in range(request.max_tokens):

        # If its the first token, we simulate the TTFT delay
        if idx == 0:
            await asyncio.sleep(TTFT_DELAY)
        else:
            await asyncio.sleep(ITL_DELAY)

        chunk = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": " tok",
                    "logprobs": None,
                    "finish_reason": None if idx != request.max_tokens - 1 else "length"
                }
            ],
        }

        yield f"data: {json.dumps(chunk)}\n\n"

        chunks_sent += 1
        

    if request.stream_options and request.stream_options.get("include_usage", False):
        usage_chunk = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [],
            "usage": {
                "prompt_tokens": len(request.prompt), # Number of tokens in input
                "completion_tokens": chunks_sent, # Number of tokens in output
                "total_tokens": len(request.prompt) + chunks_sent
            }
        }

        yield f"data: {json.dumps(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):

    # Check if the request is to be streamed
    if request.stream:
        
        return StreamingResponse(
            stream_generator(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    # Otherwise, for our mock purpose this is an error
    raise HTTPException(status_code=400, detail="Mock Server Only Supports Streaming (i.e. stream=true)")
    