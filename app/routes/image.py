# app/routes/image.py

from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import time

from ..models import ImageGenerationRequestBody
from ..utils import generate_image, save_image_and_get_url

router = APIRouter()

@router.post("/v1/images/generations")
async def image_generations(
    request: Request,
    body: ImageGenerationRequestBody,
    authorization: Optional[str] = Header(None)
):
    model = body.model

    if not model :
        raise HTTPException(status_code=400, detail=f"Please enter a model: {response_format}")

    prompt = body.prompt
    n = body.n or 1
    size = body.size or '512x512'
    guidance_scale = body.guidance_scale or 7.5
    steps = body.steps or 50
    response_format = body.response_format or 'b64_json'

    if response_format not in ['url', 'b64_json']:
        raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")

    images_data = await generate_image(
        model_name=model,
        prompt=prompt,
        n=n,
        size=size,
        guidance_scale=guidance_scale,
        steps=steps
    )

    response_body = {
        "created": int(time.time()),
        "data": []
    }

    if response_format == 'b64_json':
        for img_b64 in images_data:
            response_body['data'].append({"b64_json": img_b64})
    elif response_format == 'url':
        image_urls = [save_image_and_get_url(img_b64) for img_b64 in images_data]
        for img_url in image_urls:
            response_body['data'].append({"url": img_url})
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")

    return JSONResponse(content=response_body)
