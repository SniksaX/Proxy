# app/config.py

MODEL_MAP = {
    "chat": "gpt2_345M_q8",          # Chat model
    "image": "stable_diffusion_1.4",    # Image model
    "audio": "whisper_large_v3_q8",     # Audio model
    "embeddings": "gte_qwen2_1.5B_instruct_q8" # Embedings model
}

TEXTSYNTH_BASE_URL = "http://127.0.0.1:8080/v1/engines"

IMAGE_SAVE_DIRECTORY = "static/images"

AUDIO_SAVE_DIRECTORY = 'static/audio'

IMAGE_DOMAIN = "http://localhost:8000/static/images"