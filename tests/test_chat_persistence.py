from collections.abc import Generator
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import Conversation, Document, Message
from app.repositories import DocumentRepository


ANSWER = {
    "question": "What is this document about?",
    "answer": "It is about document intelligence.",
    "source": {
        "filename": "sample.pdf",
        "page": 1,
        "section": "Overview",
        "file_type": "pdf",
        "chunk_id": "sample_page1_chunk1",
        "distance": 0.1,
    },
    "distance": 0.1,
}


@pytest.fixture
def session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    database_session = factory()
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
    yield TestClient(app)
    app.dependency_overrides.clear()


def _install_redis_mock(monkeypatch, cached=None):
    redis_service = Mock()
    redis_service.allow_request.return_value = True
    redis_service.cache_key.side_effect = (
        lambda document, question: f"chat:{document}:{question}"
    )
    redis_service.get_json.return_value = cached
    monkeypatch.setattr("app.main.redis_service", redis_service)
    return redis_service


def _assert_message_pair(session: Session, question: str, answer: str):
    messages = list(session.scalars(select(Message)))
    assert len(messages) == 2
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == question
    assert messages[1].content == answer
    assert messages[0].conversation_id == messages[1].conversation_id
    conversation = session.get(Conversation, messages[0].conversation_id)
    assert conversation is not None
    return conversation


def test_chat_cache_miss_persists_user_and_assistant_messages(
    client, session, monkeypatch
):
    document = DocumentRepository(session).create("sample.pdf", "pdf", status="ready")
    _install_redis_mock(monkeypatch)

    with patch("app.main.answer_question", return_value=ANSWER):
        response = client.post(
            "/api/chat",
            json={
                "question": "What is this document about?",
                "document": "sample.pdf",
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == ANSWER["answer"]
    conversation = _assert_message_pair(
        session,
        "What is this document about?",
        "It is about document intelligence.",
    )
    assert conversation.document_id == document.id
    assert session.scalars(select(Conversation)).one().id == conversation.id


def test_chat_reuses_existing_conversation_id(client, session, monkeypatch):
    document = DocumentRepository(session).create("sample.pdf", "pdf", status="ready")
    conversation = Conversation(document_id=document.id)
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    _install_redis_mock(monkeypatch)

    with patch("app.main.answer_question", return_value=ANSWER):
        response = client.post(
            "/api/chat",
            json={
                "question": "What is this document about?",
                "document": "sample.pdf",
                "conversation_id": conversation.id,
            },
        )

    assert response.status_code == 200
    messages = list(session.scalars(select(Message)))
    assert {message.conversation_id for message in messages} == {conversation.id}
    assert session.scalars(select(Conversation)).one().id == conversation.id


def test_chat_mismatched_conversation_id_creates_new_conversation(
    client, session, monkeypatch
):
    document_a = DocumentRepository(session).create("a.pdf", "pdf", status="ready")
    document_b = DocumentRepository(session).create("b.pdf", "pdf", status="ready")
    conversation_a = Conversation(document_id=document_a.id)
    session.add(conversation_a)
    session.commit()
    session.refresh(conversation_a)
    _install_redis_mock(monkeypatch)

    with patch("app.main.answer_question", return_value=ANSWER):
        response = client.post(
            "/api/chat",
            json={
                "question": "What is this document about?",
                "document": "b.pdf",
                "conversation_id": conversation_a.id,
            },
        )

    assert response.status_code == 200
    conversations = list(session.scalars(select(Conversation)))
    assert len(conversations) == 2
    new_conversation = next(
        item for item in conversations if item.id != conversation_a.id
    )
    assert new_conversation.document_id == document_b.id
    messages = list(session.scalars(select(Message)))
    assert {message.conversation_id for message in messages} == {
        new_conversation.id
    }


def test_chat_cache_hit_persists_messages_and_skips_rag(
    client, session, monkeypatch
):
    DocumentRepository(session).create("sample.pdf", "pdf", status="ready")
    _install_redis_mock(
        monkeypatch,
        cached={
            "question": "What is this document about?",
            "answer": "Cached answer.",
            "source": None,
            "distance": None,
        },
    )

    with patch("app.main.answer_question") as answer_question:
        response = client.post(
            "/api/chat",
            json={
                "question": "What is this document about?",
                "document": "sample.pdf",
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Cached answer."
    answer_question.assert_not_called()
    conversation = _assert_message_pair(
        session,
        "What is this document about?",
        "Cached answer.",
    )
    assert conversation.document_id is not None


def test_chat_succeeds_when_persistence_fails(client, session, monkeypatch):
    _install_redis_mock(monkeypatch)

    with patch(
        "app.main.answer_question", return_value=ANSWER
    ), patch(
        "app.main.DocumentRepository.get_or_create_by_filename",
        side_effect=RuntimeError("database unavailable"),
    ):
        response = client.post(
            "/api/chat",
            json={
                "question": "What is this document about?",
                "document": "sample.pdf",
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == ANSWER["answer"]
    assert list(session.scalars(select(Message))) == []


def test_chat_creates_missing_document_row(client, session, monkeypatch):
    _install_redis_mock(monkeypatch)

    with patch("app.main.answer_question", return_value=ANSWER):
        response = client.post(
            "/api/chat",
            json={
                "question": "What is this document about?",
                "document": "unknown.pdf",
            },
        )

    assert response.status_code == 200
    documents = list(session.scalars(select(Document)))
    assert len(documents) == 1
    assert documents[0].filename == "unknown.pdf"
    assert documents[0].file_type == "pdf"
    assert documents[0].status == "ready"
    assert _assert_message_pair(
        session,
        "What is this document about?",
        "It is about document intelligence.",
    ).document_id == documents[0].id


def test_chat_does_not_create_duplicate_documents(client, session, monkeypatch):
    _install_redis_mock(monkeypatch)

    with patch("app.main.answer_question", return_value=ANSWER):
        response_one = client.post(
            "/api/chat",
            json={"question": "First question?", "document": "same.pdf"},
        )
        response_two = client.post(
            "/api/chat",
            json={"question": "Second question?", "document": "same.pdf"},
        )

    assert response_one.status_code == 200
    assert response_two.status_code == 200
    documents = list(session.scalars(select(Document)))
    assert len(documents) == 1
    assert documents[0].filename == "same.pdf"


def test_get_or_create_by_filename_preserves_existing_row(session: Session):
    repository = DocumentRepository(session)

    first = repository.get_or_create_by_filename("sample.pdf")
    second = repository.get_or_create_by_filename("sample.pdf")

    assert first.id == second.id
    documents = list(session.scalars(select(Document)))
    assert len(documents) == 1
    assert documents[0].filename == "sample.pdf"
    assert documents[0].file_type == "pdf"
    assert documents[0].status == "ready"

