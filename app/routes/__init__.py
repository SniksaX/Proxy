# app/routes/__init__.py

from .chat import router as chat_router
from .image import router as image_router
from .audio import router as audio_router

__all__ = ["chat_router", "image_router", "audio_router"]