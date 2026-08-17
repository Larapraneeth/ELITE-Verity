import logging
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document
from app.repositories import DocumentRepository
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


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: str
    status: str
    created_at: datetime
    updated_at: datetime


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
