# app/utils.py

import json
import time
import uuid
import httpx
import logging
import base64
import os
from typing import List
from fastapi import HTTPException
from app.config import MODEL_MAP, TEXTSYNTH_BASE_URL, IMAGE_SAVE_DIRECTORY, IMAGE_DOMAIN
from app.models import Message
from typing import List, Optional, Union
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
tokenizer_cache = {}
os.makedirs(IMAGE_SAVE_DIRECTORY, exist_ok=True)

# -------- Tokenizer ---------

async def get_token_count(model_name: str, text: str) -> int:
    textsynth_model_name = MODEL_MAP.get(model_name)
    if not textsynth_model_name:
        logger.error(f"Model {model_name} not supported for tokenization")
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported for tokenization")

    tokenize_url = f"{TEXTSYNTH_BASE_URL}/{textsynth_model_name}/tokenize"
    payload = {"text": text}

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(tokenize_url, json=payload)
            response.raise_for_status()
            result = response.json()
            tokens = result.get("tokens", [])
            token_count = len(tokens)
            return token_count
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during tokenization: {e}")
            raise HTTPException(status_code=response.status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            raise HTTPException(status_code=500, detail="Error during tokenization")

# tokenization: 
# curl -X POST "http://localhost:8080/v1/engines/gpt2_345M_q8/tokenize" -H "Content-Type: application/json" -d '{
#   "text": "This is a test prompt."
# }'

# -------- Embeddings --------

async def generate_embeddings(model_name: str, input_text: Union[str, List[str]]):
    textsynth_model_name = MODEL_MAP.get(model_name)
    if not textsynth_model_name:
        logger.error(f"Model {model_name} not supported")
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")

    if isinstance(input_text, str):
        inputs = [input_text]
    else:
        inputs = input_text

    payload = {"input": inputs}

    textsynth_url = f"{TEXTSYNTH_BASE_URL}/{textsynth_model_name}/embeddings"
    print(textsynth_url)

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(textsynth_url, json=payload)
            response.raise_for_status()
            result = response.json()
            print(result)
            data = result.get("data", [])
            if not data:
                logger.error("No embeddings returned from TextSynth")
                raise HTTPException(status_code=500, detail="No embeddings returned from TextSynth")

            # Extract embeddings from the "data" field
            embeddings = [item["embedding"] for item in data]
            return embeddings
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            raise HTTPException(status_code=response.status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Error contacting TextSynth server: {e}")
            raise HTTPException(status_code=500, detail="Error contacting TextSynth server")


# -------- Chat Generation --------

# Function for generating chat completions
async def chat_generation(model_name: str, messages: List[Message], temperature: float, max_tokens: int):
    # Get the corresponding TextSynth model from the model map
    textsynth_model_name = MODEL_MAP.get(model_name)
    # If the model is not supported, log the error and raise an HTTPException
    if not textsynth_model_name:
        logger.error(f"Model {model_name} not supported")
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")

    # Build the chat prompt from the messages
    prompt = ""
    for message in messages:
        prompt += f"{message.role.capitalize()}: {message.content}\n"

    # Set up the payload to send to TextSynth API
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # Send the request to the TextSynth API
    textsynth_url = f"{TEXTSYNTH_BASE_URL}/{textsynth_model_name}/completions"

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(textsynth_url, json=payload)
            response.raise_for_status()
            result = response.json()
            print(result)
            generated_text = result.get("text", "")
            return generated_text
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            raise HTTPException(status_code=response.status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Error contacting TextSynth server: {e}")
            raise HTTPException(status_code=500, detail="Error contacting TextSynth server")


# Function for streaming chat responses in real time
async def stream_chat_response(model_name: str, messages: List[Message], temperature: float, max_tokens: int):
    textsynth_model_name = MODEL_MAP.get(model_name)
    if not textsynth_model_name:
        # Return an error if model isn't supported
        yield json.dumps({"error": f"Model {model_name} not supported"})
        return

    # Build the prompt similar to the normal chat function
    prompt = ""
    for message in messages:
        prompt += f"{message.role.capitalize()}: {message.content}\n"

    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True  # Enable streaming mode
    }

    textsynth_url = f"{TEXTSYNTH_BASE_URL}/{textsynth_model_name}/completions"

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", textsynth_url, json=payload) as response:
                response.raise_for_status()

                # Stream the response as it's being generated
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        generated_text = data.get("text", "")
                        chunk = {
                            "id": "chatcmpl-" + str(uuid.uuid4()),
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "delta": {
                                        "content": generated_text
                                    },
                                    "index": 0,
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Send the finish chunk to signal completion of the stream
                finish_chunk = {
                    "id": "chatcmpl-" + str(uuid.uuid4()),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "delta": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n"
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during streaming: {e}")
            yield json.dumps({"error": str(e)})
        except Exception as e:
            logger.error(f"Error contacting TextSynth server during streaming: {e}")
            yield json.dumps({"error": "Error contacting TextSynth server"})


# -------- Image Generation --------

# Function to generate images based on a prompt using TextSynth
async def generate_image(
    model_name: str,
    prompt: str,
    n: int = 1,
    size: str = '512x512',
    guidance_scale: float = 7.5,
    steps: int = 50
):
    textsynth_model_name = MODEL_MAP.get(model_name)
    if not textsynth_model_name:
        logger.error(f"Model {model_name} not supported")
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")

    try:
        # Parse the width and height from the size string (e.g., '512x512')
        width_str, height_str = size.lower().split('x')
        width = int(width_str)
        height = int(height_str)
    except ValueError:
        logger.error(f"Invalid size format: {size}")
        raise HTTPException(status_code=400, detail=f"Invalid size format: {size}")

    payload = {
        "prompt": prompt,
        "num_images": n,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "width": width,
        "height": height
    }

    textsynth_url = f"{TEXTSYNTH_BASE_URL}/{textsynth_model_name}/text_to_image"

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            # Send the request to generate the image(s)
            response = await client.post(textsynth_url, json=payload)
            response.raise_for_status()
            result = response.json()

            # Extract the base64-encoded images from the response
            images_data = result.get('images', [])
            if not images_data:
                logger.error("No images returned from TextSynth")
                raise HTTPException(status_code=500, detail="No images returned from TextSynth")
            
            return images_data
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            raise HTTPException(status_code=response.status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Error contacting TextSynth server: {e}")
            raise HTTPException(status_code=500, detail="Error contacting TextSynth server")


# -------- Audio Generation --------

# Function to generate a transcription from audio bytes
async def generate_audio(
    model_name: str,
    audio_bytes: bytes,
    language: Optional[str] = None,
    prompt: Optional[str] = None
):
    textsynth_model_name = MODEL_MAP.get(model_name)
    if not textsynth_model_name:
        logger.error(f"Model {model_name} not supported")
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")

    json_payload = {}
    if language:
        json_payload["language"] = language
    if prompt:
        json_payload["prompt"] = prompt

    # Prepare the files for the request
    files = {
        'json': (None, json.dumps(json_payload), 'application/json'),
        'file': ('audio_file', audio_bytes, 'application/octet-stream')
    }

    textsynth_url = f"{TEXTSYNTH_BASE_URL}/{textsynth_model_name}/transcript"

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(textsynth_url, files=files)
            response.raise_for_status()
            result = response.json()
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            raise HTTPException(status_code=response.status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Error contacting TextSynth server: {e}")
            raise HTTPException(status_code=500, detail="Error contacting TextSynth server")


# -------- Helper Functions --------

# Function to save a base64-encoded image and return its URL
def save_image_and_get_url(img_b64: str) -> str:
    img_data = base64.b64decode(img_b64)
    filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join(IMAGE_SAVE_DIRECTORY, filename)
    with open(image_path, "wb") as f:
        f.write(img_data)
    image_url = f"{IMAGE_DOMAIN}/{filename}"
    return image_url

# Function to convert seconds to a timestamp (SRT or VTT format)
def seconds_to_timestamp(seconds, separator=','):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    if separator == ',':
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
    else:
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

# Convert transcription segments into SRT subtitle format
def segments_to_srt(segments):
    srt_content = ''
    for idx, segment in enumerate(segments, start=1):
        start_time = seconds_to_timestamp(segment.get('start', 0), separator=',')
        end_time = seconds_to_timestamp(segment.get('end', 0), separator=',')
        text = segment.get('text', '').strip()
        srt_content += f"{idx}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_content.strip()

# Convert transcription segments into VTT subtitle format
def segments_to_vtt(segments):
    vtt_content = 'WEBVTT\n\n'
    for idx, segment in enumerate(segments, start=1):
        start_time = seconds_to_timestamp(segment.get('start', 0), separator='.')
        end_time = seconds_to_timestamp(segment.get('end', 0), separator='.')
        text = segment.get('text', '').strip()
        vtt_content += f"{start_time} --> {end_time}\n{text}\n\n"
    return vtt_content.strip()