import logging
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document
from app.repositories import DocumentRepository
from app.services.rag_service import answer_question
from app.services.redis_service import redis_service
from app.services.document_processing import document_processing_service
from app.config import (
    CACHE_TTL_SECONDS,
    CHAT_RATE_LIMIT,
    CHAT_RATE_LIMIT_WINDOW_SECONDS,
)


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


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: str
    status: str
    created_at: datetime
    updated_at: datetime


class DocumentStatusResponse(BaseModel):
    id: int
    status: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    client_host = (
        http_request.client.host
        if http_request.client is not None
        else "unknown"
    )

    if not redis_service.allow_request(
        client_host,
        CHAT_RATE_LIMIT,
        CHAT_RATE_LIMIT_WINDOW_SECONDS,
    ):
        raise HTTPException(status_code=429, detail="Too many chat requests.")

    cache_key = redis_service.cache_key(request.document, request.question)
    cached_result = redis_service.get_json(cache_key)
    if cached_result is not None:
        try:
            return ChatResponse.model_validate(cached_result)
        except ValidationError:
            logger.warning("Ignoring invalid cached chat response")

    try:
        result = answer_question(request.question, request.document)
        response = ChatResponse.model_validate(result)
        redis_service.set_json(
            cache_key,
            response.model_dump(mode="json"),
            CACHE_TTL_SECONDS,
        )
        return response
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process chat request."
        )


@app.get("/api/documents", response_model=list[DocumentResponse])
def list_documents(session: Session = Depends(get_db)) -> list[Document]:
    return DocumentRepository(session).list()


@app.get("/api/documents/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    session: Session = Depends(get_db),
) -> Document:
    document = DocumentRepository(session).get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@app.get("/api/documents/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(
    document_id: int,
    session: Session = Depends(get_db),
) -> DocumentStatusResponse:
    document = DocumentRepository(session).get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentStatusResponse(id=document.id, status=document.status)


@app.post("/api/documents/{document_id}/process", response_model=DocumentResponse, status_code=202)
def queue_document_processing(
    document_id: int,
    session: Session = Depends(get_db),
) -> Document:
    repository = DocumentRepository(session)
    document = repository.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if (
        document.status == "processing"
        or document_processing_service.is_active(document_id)
    ):
        raise HTTPException(status_code=409, detail="Document is already being processed.")

    document = repository.update_status(document_id, "pending")
    if not document_processing_service.enqueue(document_id):
        repository.update_status(document_id, "failed")
        raise HTTPException(status_code=503, detail="Unable to queue document processing.")
    return document
