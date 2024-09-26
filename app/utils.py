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
from typing import List, Optional

logger = logging.getLogger(__name__)

# Ensure the image save directory exists
os.makedirs(IMAGE_SAVE_DIRECTORY, exist_ok=True)

# -------- Chat Generation --------

async def chat_generation(model_name: str, messages: List[Message], temperature: float, max_tokens: int):
    textsynth_model_name = MODEL_MAP.get(model_name)
    if not textsynth_model_name:
        logger.error(f"Model {model_name} not supported")
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")

    prompt = ""
    for message in messages:
        prompt += f"{message.role.capitalize()}: {message.content}\n"

    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    textsynth_url = f"{TEXTSYNTH_BASE_URL}/{textsynth_model_name}/completions"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(textsynth_url, json=payload)
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("text", "")
            return generated_text
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            raise HTTPException(status_code=response.status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Error contacting TextSynth server: {e}")
            raise HTTPException(status_code=500, detail="Error contacting TextSynth server")

async def stream_chat_response(model_name: str, messages: List[Message], temperature: float, max_tokens: int):
    textsynth_model_name = MODEL_MAP.get(model_name)
    if not textsynth_model_name:
        yield json.dumps({"error": f"Model {model_name} not supported"})
        return

    prompt = ""
    for message in messages:
        prompt += f"{message.role.capitalize()}: {message.content}\n"

    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    textsynth_url = f"{TEXTSYNTH_BASE_URL}/{textsynth_model_name}/completions"

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", textsynth_url, json=payload) as response:
                response.raise_for_status()

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

# app/utils.py

async def generate_image(
    model_name: str,
    prompt: str,
    n: int = 1,
    size: str = '512x512',
    guidance_scale: float = 7.5,
    steps: int = 50
):
    # Map OpenAI model to TextSynth model
    textsynth_model_name = MODEL_MAP.get(model_name)
    if not textsynth_model_name:
        logger.error(f"Model {model_name} not supported")
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")

    # Parse size (OpenAI uses a string like '512x512')
    try:
        width_str, height_str = size.lower().split('x')
        width = int(width_str)
        height = int(height_str)
    except ValueError:
        logger.error(f"Invalid size format: {size}")
        raise HTTPException(status_code=400, detail=f"Invalid size format: {size}")

    # Build payload for TextSynth API (matches your curl command)
    payload = {
        "prompt": prompt,
        "num_images": n,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "width": width,
        "height": height
    }

    # URL for the TextSynth image generation API
    textsynth_url = f"{TEXTSYNTH_BASE_URL}/{textsynth_model_name}/text_to_image"

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            # Send POST request to TextSynth
            response = await client.post(textsynth_url, json=payload)
            response.raise_for_status()
            result = response.json()

            # Return the images (assuming TextSynth returns them as base64-encoded strings in 'images')
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

    # Prepare the 'json' field
    json_payload = {}
    if language:
        json_payload["language"] = language
    if prompt:
        json_payload["prompt"] = prompt

    # Prepare the files for the multipart/form-data request
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

def save_image_and_get_url(img_b64: str) -> str:
    # Decode the base64 image
    img_data = base64.b64decode(img_b64)
    # Generate a unique filename
    filename = f"{uuid.uuid4()}.png"
    # Save the image to a directory (e.g., 'static/images')
    image_path = os.path.join(IMAGE_SAVE_DIRECTORY, filename)
    with open(image_path, "wb") as f:
        f.write(img_data)
    # Construct the URL to access the image
    image_url = f"{IMAGE_DOMAIN}/{filename}"
    return image_url


def seconds_to_timestamp(seconds, separator=','):
    # Convert seconds to timestamp format 'HH:MM:SS,mmm' for SRT or 'HH:MM:SS.mmm' for VTT
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    if separator == ',':
        # For SRT
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
    else:
        # For VTT
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

def segments_to_srt(segments):
    srt_content = ''
    for idx, segment in enumerate(segments, start=1):
        start_time = seconds_to_timestamp(segment.get('start', 0), separator=',')
        end_time = seconds_to_timestamp(segment.get('end', 0), separator=',')
        text = segment.get('text', '').strip()
        srt_content += f"{idx}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_content.strip()

def segments_to_vtt(segments):
    vtt_content = 'WEBVTT\n\n'
    for idx, segment in enumerate(segments, start=1):
        start_time = seconds_to_timestamp(segment.get('start', 0), separator='.')
        end_time = seconds_to_timestamp(segment.get('end', 0), separator='.')
        text = segment.get('text', '').strip()
        vtt_content += f"{start_time} --> {end_time}\n{text}\n\n"
    return vtt_content.strip()