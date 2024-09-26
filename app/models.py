# app/models.py

from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict

class Message(BaseModel):
    role: str
    content: str

class OpenAIRequestBody(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 256
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ImageGenerationRequestBody(BaseModel):
    model: Optional[str] = None
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = '512x512'
    response_format: Optional[str] = 'b64_json'  # 'b64_json' or 'url'
    guidance_scale: Optional[float] = 7.5
    steps: Optional[int] = 50
    user: Optional[str] = None

class AudioGenerationRequestBody(BaseModel):
    model: Optional[str] = Field(default="whisper-1")
    audio: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = 'json'
    user: Optional[str] = None
