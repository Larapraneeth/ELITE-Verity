import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.services.rag_service import answer_question


logger = logging.getLogger(__name__)

app = FastAPI(title="ELITE Verity API")


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1)
    document: str = Field(min_length=1)


class ChatResponse(BaseModel):
    question: str
    answer: str
    source: dict[str, Any] | None = None
    distance: float | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = answer_question(request.question, request.document)
        return ChatResponse.model_validate(result)
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process chat request."
        )
