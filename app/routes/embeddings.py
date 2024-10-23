# app/routes/embeddings.py

from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Union

from ..models import EmbeddingsRequestBody, EmbeddingsResponseBody, EmbeddingData
from ..utils import generate_embeddings

router = APIRouter()

@router.post("/v1/embeddings")
async def embeddings_endpoint(
    request: Request,
    body: EmbeddingsRequestBody,
    authorization: Optional[str] = Header(None)
):
    model = body.model
    input_text = body.input


    embeddings = await generate_embeddings(model_name=model, input_text=input_text)

    data = []
    for idx, embedding in enumerate(embeddings):
        embedding_data = EmbeddingData(
            object="embedding",
            embedding=embedding,
            index=idx
        )
        data.append(embedding_data)

    response_body = EmbeddingsResponseBody(
        object="list",
        data=data,
        model=model,
        usage={
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    )
    return JSONResponse(content=response_body.dict())


# Req ex : 
# curl -X POST "http://localhost:8000/v1/embeddings" -H "Content-Type: application/json" -d '{
#   "model": "gte_qwen2_1.5B_instruct_q8",
#   "input": "This is a test input for embeddings",
#   "user": "test_user"
# }'