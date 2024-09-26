# app/routes/chat.py

from fastapi import APIRouter, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import time
import uuid

from ..models import OpenAIRequestBody
from ..utils import chat_generation, stream_chat_response

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: OpenAIRequestBody,
    authorization: Optional[str] = Header(None)
):
    model = body.model
    messages = body.messages
    temperature = body.temperature or 1.0
    max_tokens = body.max_tokens or 256
    stream = body.stream

    if stream:
        generator = stream_chat_response(model, messages, temperature, max_tokens)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        generated_text = await chat_generation(model, messages, temperature, max_tokens)
        response_body = {
            "id": "chatcmpl-" + str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        return JSONResponse(content=response_body)
