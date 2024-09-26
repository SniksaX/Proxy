# app/main.py

from fastapi import FastAPI, Request
import logging
from fastapi.staticfiles import StaticFiles
import os

from app.routes.chat import router as chat_router
from app.routes.image import router as image_router
from app.routes.audio import router as audio_router

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

    client_ip = request.client.host
    method = request.method
    url = str(request.url)
    
    try:
        body = await request.json()
    except Exception:
        body = "Unable to read body"

    logging.info(f"Client IP: {client_ip} | Method: {method} | URL: {url} | Body: {body}")
    response = await call_next(request)
    logging.info(f"Response Status: {response.status_code}")

    return response

app.include_router(chat_router)
app.include_router(image_router)
app.include_router(audio_router)
