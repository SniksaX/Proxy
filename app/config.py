# app/config.py

MODEL_MAP = {
    "chat": "flan_t5_base_q8",        # Chat model
    "image": "stable_diffusion_1.4",  # Image model
    "audio": "whisper_large_v3_q8"    # Audio model
}

TEXTSYNTH_BASE_URL = "http://127.0.0.1:8080/v1/engines"

NODEJS_SERVER_URL = "http://127.0.0.1:5000/v1/images/generations"

IMAGE_SAVE_DIRECTORY = "static/images"

IMAGE_DOMAIN = "http://localhost:8000/static/images"