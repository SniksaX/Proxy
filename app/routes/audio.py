# app/routes/audio.py

from fastapi import APIRouter, Request, Header, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from typing import Optional
from ..utils import generate_audio, segments_to_srt, segments_to_vtt

router = APIRouter()

@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form('json'),
    temperature: Optional[float] = Form(None),  # Not used in this implementation
    language: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    if response_format not in ['json', 'text', 'srt', 'verbose_json', 'vtt']:
        raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")

    # Read the audio file
    audio_bytes = await file.read()

    # Prepare the request to TextSynth
    transcription_result = await generate_audio(
        model_name=model,
        audio_bytes=audio_bytes,
        language=language,
        prompt=prompt
    )

    # Now, process the transcription_result to match OpenAI's API response
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