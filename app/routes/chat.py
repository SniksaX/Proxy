# app/routes/chat.py

from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import time
import uuid

from ..models import OpenAIRequestBody
from ..utils import chat_generation, stream_chat_response, get_token_count

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

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    prompt = ""
    for message in messages:
        prompt += f"{message.role.capitalize()}: {message.content}\n"

    prompt_tokens = await get_token_count(model_name=model, text=prompt)

    if stream:
        # Note: Token counting for streaming responses is not implemented here
        generator = stream_chat_response(model, messages, temperature, max_tokens)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        # Generate the chat completion
        generated_text = await chat_generation(model, messages, temperature, max_tokens)

        # Get the completion token count
        completion_tokens = await get_token_count(model_name=model, text=generated_text)

        total_tokens = prompt_tokens + completion_tokens

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
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
        return JSONResponse(content=response_body)

# curl -X POST "http://localhost:8000/v1/chat/completions" -H "Content-Type: application/json" -d '{
#   "model": "chat", 
#   "messages": [
#     {
#       "role": "user",
#       "content": "Hello, how are you?"
#     }
#   ],
#   "temperature": 1.0,
#   "max_tokens": 150,
#   "stream": true/false
# }'
