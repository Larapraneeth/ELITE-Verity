from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import Conversation, Document, Message
from app.repositories import DocumentRepository


@pytest.fixture
def session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine)
    database_session = testing_session()

    try:
        yield database_session
    finally:
        database_session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def client(session: Session) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_document_repository_and_models(session: Session):
    repository = DocumentRepository(session)
    document = repository.create("sample.pdf", "pdf", status="processed")

    conversation = Conversation(document_id=document.id)
    session.add(conversation)
    session.commit()
    session.refresh(conversation)

    message = Message(
        conversation_id=conversation.id,
        role="user",
        content="What is this document about?",
    )
    session.add(message)
    session.commit()
    session.refresh(document)

    assert repository.get(document.id) == document
    assert repository.list() == [document]
    assert document.conversations == [conversation]
    assert conversation.messages == [message]


def test_document_endpoints(client: TestClient, session: Session):
    document = DocumentRepository(session).create("sample.pdf", "pdf")

    list_response = client.get("/api/documents")
    assert list_response.status_code == 200
    assert list_response.json()[0]["id"] == document.id
    assert list_response.json()[0]["filename"] == "sample.pdf"

    detail_response = client.get(f"/api/documents/{document.id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["status"] == "pending"

    missing_response = client.get("/api/documents/9999")
    assert missing_response.status_code == 404
    assert missing_response.json() == {"detail": "Document not found."}
