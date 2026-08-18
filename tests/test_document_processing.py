from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, call, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.repositories import DocumentRepository
from app.services.document_processing import DocumentProcessingService


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
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_enqueue_prevents_duplicate_jobs():
    executor = Mock()
    service = DocumentProcessingService(session_factory=Mock(), executor=executor)

    assert service.enqueue(12) is True
    assert service.enqueue(12) is False
    executor.submit.assert_called_once_with(service.process, 12)


def test_worker_marks_document_ready_after_existing_ingestion_services_run():
    session = Mock()
    document = Mock(filename="sample.pdf")
    repository = Mock()
    repository.claim_pending.return_value = document
    vector_store = Mock()
    service = DocumentProcessingService(session_factory=lambda: session, executor=Mock())

    with patch(
        "app.services.document_processing.DocumentRepository",
        return_value=repository,
    ), patch(
        "app.services.document_processing.process_document",
        return_value=[{"text": "document content"}],
    ), patch(
        "app.services.document_processing.generate_embeddings",
        return_value=[[0.1, 0.2]],
    ), patch(
        "app.services.document_processing.VectorStore",
        return_value=vector_store,
    ):
        service.process(12)

    repository.claim_pending.assert_called_once_with(12)
    assert repository.update_status.call_args_list == [call(12, "ready")]
    vector_store.add_chunks.assert_called_once()
    session.close.assert_called_once()


def test_worker_marks_document_failed_when_processing_raises():
    session = Mock()
    repository = Mock()
    repository.claim_pending.return_value = Mock(filename="sample.pdf")
    service = DocumentProcessingService(
        session_factory=lambda: session,
        executor=Mock(),
        max_retries=2,
        retry_delay_seconds=0,
    )

    with patch(
        "app.services.document_processing.DocumentRepository",
        return_value=repository,
    ), patch(
        "app.services.document_processing.process_document",
        side_effect=ValueError("bad PDF"),
    ):
        service.process(12)

    repository.claim_pending.assert_called_once_with(12)
    assert repository.update_status.call_args_list == [call(12, "failed")]


def test_processing_and_status_endpoints(client: TestClient, session: Session, monkeypatch):
    document = DocumentRepository(session).create("sample.pdf", "pdf")
    monkeypatch.setattr("app.main.document_processing_service.is_active", lambda _id: False)
    enqueue = Mock(return_value=True)
    monkeypatch.setattr("app.main.document_processing_service.enqueue", enqueue)

    response = client.post(f"/api/documents/{document.id}/process")
    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    enqueue.assert_called_once_with(document.id)

    status_response = client.get(f"/api/documents/{document.id}/status")
    assert status_response.status_code == 200
    assert status_response.json() == {"id": document.id, "status": "pending"}


def test_processing_endpoint_rejects_active_duplicate(client: TestClient, session: Session, monkeypatch):
    document = DocumentRepository(session).create("sample.pdf", "pdf")
    monkeypatch.setattr("app.main.document_processing_service.is_active", lambda _id: True)

    response = client.post(f"/api/documents/{document.id}/process")

    assert response.status_code == 409
    assert response.json() == {"detail": "Document is already being processed."}


def test_first_attempt_fails_then_retry_succeeds():
    session = Mock()
    repository = Mock()
    repository.claim_pending.return_value = Mock(filename="sample.pdf")
    vector_store = Mock()
    service = DocumentProcessingService(
        session_factory=lambda: session,
        executor=Mock(),
        max_retries=3,
        retry_delay_seconds=0,
    )

    with patch(
        "app.services.document_processing.DocumentRepository",
        return_value=repository,
    ), patch(
        "app.services.document_processing.process_document",
        side_effect=[
            ValueError("transient failure"),
            [{"text": "document content"}],
        ],
    ), patch(
        "app.services.document_processing.generate_embeddings",
        return_value=[[0.1, 0.2]],
    ), patch(
        "app.services.document_processing.VectorStore",
        return_value=vector_store,
    ):
        service.process(12)

    # The atomic claim is only ever executed once (no duplicate processing).
    repository.claim_pending.assert_called_once_with(12)
    assert repository.update_status.call_args_list == [call(12, "ready")]
    vector_store.add_chunks.assert_called_once()
    session.close.assert_called_once()


def test_all_retries_fail_marks_document_failed():
    session = Mock()
    repository = Mock()
    repository.claim_pending.return_value = Mock(filename="sample.pdf")
    service = DocumentProcessingService(
        session_factory=lambda: session,
        executor=Mock(),
        max_retries=2,
        retry_delay_seconds=0,
    )

    with patch(
        "app.services.document_processing.DocumentRepository",
        return_value=repository,
    ), patch(
        "app.services.document_processing.process_document",
        side_effect=ValueError("bad PDF"),
    ) as process_document:
        service.process(12)

    # 1 initial attempt + 2 retries, then the document is marked failed.
    assert process_document.call_count == 3
    repository.claim_pending.assert_called_once_with(12)
    assert repository.update_status.call_args_list == [call(12, "failed")]


def test_retry_limit_is_respected():
    session = Mock()
    repository = Mock()
    repository.claim_pending.return_value = Mock(filename="sample.pdf")
    service = DocumentProcessingService(
        session_factory=lambda: session,
        executor=Mock(),
        max_retries=5,
        retry_delay_seconds=0,
    )

    with patch(
        "app.services.document_processing.DocumentRepository",
        return_value=repository,
    ), patch(
        "app.services.document_processing.process_document",
        side_effect=ValueError("bad PDF"),
    ) as process_document:
        service.process(12)

    # Never exceeds the configured limit: initial attempt + max_retries retries.
    assert service.max_retries == 5
    assert process_document.call_count == 6
    assert repository.update_status.call_args_list == [call(12, "failed")]


def test_other_documents_still_process_after_failure():
    session = Mock()
    repository = Mock()

    def claim_pending_side_effect(document_id: int):
        return (
            Mock(filename="bad.pdf") if document_id == 1 else Mock(filename="good.pdf")
        )

    repository.claim_pending.side_effect = claim_pending_side_effect
    vector_store = Mock()
    service = DocumentProcessingService(
        session_factory=lambda: session,
        executor=Mock(),
        max_retries=1,
        retry_delay_seconds=0,
    )

    with patch(
        "app.services.document_processing.DocumentRepository",
        return_value=repository,
    ), patch(
        "app.services.document_processing.process_document",
        side_effect=[
            ValueError("bad PDF"),
            ValueError("bad PDF"),
            [{"text": "good content"}],
        ],
    ), patch(
        "app.services.document_processing.generate_embeddings",
        return_value=[[0.1, 0.2]],
    ), patch(
        "app.services.document_processing.VectorStore",
        return_value=vector_store,
    ):
        service.process(1)
        service.process(2)

    # The first document failed without crashing the worker; the second one was
    # still claimed, processed, and marked ready.
    assert repository.claim_pending.call_args_list == [call(1), call(2)]
    assert repository.update_status.call_args_list == [
        call(1, "failed"),
        call(2, "ready"),
    ]
    vector_store.add_chunks.assert_called_once()


def test_recover_stale_processing_resets_stuck_documents(session: Session):
    repository = DocumentRepository(session)
    document = repository.create("stuck.pdf", "pdf")
    assert repository.claim_pending(document.id) is not None

    # Simulate a worker crash that left the document stuck long ago.
    document.updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
    session.commit()

    assert repository.recover_stale_processing(stale_seconds=300) == 1
    session.refresh(document)
    assert document.status == "pending"

