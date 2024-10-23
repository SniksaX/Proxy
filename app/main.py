# app/main.py

from fastapi import FastAPI, Request
import logging
from fastapi.staticfiles import StaticFiles
import os

from app.routes.chat import router as chat_router
from app.routes.image import router as image_router
from app.routes.audio import router as audio_router
from app.routes.embeddings import router as embeddings_router


os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/req.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Func to save Logs.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Read the body
    body_bytes = await request.body()
    # Log the body
    logging.info(f"Client IP: {request.client.host} | Method: {request.method} | URL: {str(request.url)} | Body: {body_bytes.decode('utf-8')}")
    
    # Reset the body so downstream handlers can read it
    async def receive():
        return {'type': 'http.request', 'body': body_bytes}
    request._receive = receive

    response = await call_next(request)
    logging.info(f"Response Status: {response.status_code}")

    return response


def test ()  ->  dict[str]:
    return 0,
app.include_router(chat_router)
app.include_router(image_router)
app.include_router(audio_router)
app.include_router(embeddings_router)
