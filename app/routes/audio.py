# app/routes/audio.py

from fastapi import APIRouter, Request, Header, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from typing import Optional
from app.config import AUDIO_SAVE_DIRECTORY
import uuid
import os

from ..utils import generate_audio, segments_to_srt, segments_to_vtt

router = APIRouter()

# directory where audio files will be saved
os.makedirs(AUDIO_SAVE_DIRECTORY, exist_ok=True)

@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form('json'),
    temperature: Optional[float] = Form(None),
    language: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    if response_format not in ['json', 'text', 'srt', 'verbose_json', 'vtt']:
        raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")


    audio_bytes = await file.read()

    # Save the audio file before transmitting it to TextSynth
    filename = f"{uuid.uuid4()}_{file.filename}"
    audio_path = os.path.join(AUDIO_SAVE_DIRECTORY, filename)
    with open(audio_path, 'wb') as f:
        f.write(audio_bytes)

    transcription_result = await generate_audio(
        model_name=model,
        audio_bytes=audio_bytes,
        language=language,
        prompt=prompt
    )

    if response_format == 'json':
        response_body = {
            "text": transcription_result.get('text', '')
        }
        return JSONResponse(content=response_body)

    elif response_format == 'verbose_json':
        response_body = {
            "text": transcription_result.get('text', ''),
            "language": transcription_result.get('language', ''),
            "duration": transcription_result.get('duration', 0),
            "segments": transcription_result.get('segments', [])
        }
        return JSONResponse(content=response_body)

    elif response_format == 'text':
        return Response(content=transcription_result.get('text', '').strip(), media_type='text/plain')

    elif response_format == 'srt':
        srt_content = segments_to_srt(transcription_result.get('segments', []))
        return Response(content=srt_content, media_type='application/x-subrip')

    elif response_format == 'vtt':
        vtt_content = segments_to_vtt(transcription_result.get('segments', []))
        return Response(content=vtt_content, media_type='text/vtt')

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")
