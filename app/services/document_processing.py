import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from app.database import SessionLocal
from app.config import (
    DOCUMENT_PROCESSING_MAX_RETRIES,
    DOCUMENT_PROCESSING_RETRY_DELAY_SECONDS,
    DOCUMENT_PROCESSING_STALE_SECONDS,
    WORKER_POLL_INTERVAL_SECONDS,
)
from app.logging_config import configure_logging
from app.repositories import DocumentRepository
from app.services.document_processor import process_document
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import VectorStore


configure_logging()
logger = logging.getLogger(__name__)
UPLOAD_DIR = Path("data/uploads")


class DocumentProcessingService:
    def __init__(
        self,
        session_factory=SessionLocal,
        executor=None,
        max_retries: int | None = None,
        retry_delay_seconds: float | None = None,
    ):
        self.session_factory = session_factory
        self.executor = executor or ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="document-processing",
        )
        self.max_retries = (
            DOCUMENT_PROCESSING_MAX_RETRIES if max_retries is None else max_retries
        )
        self.retry_delay_seconds = (
            DOCUMENT_PROCESSING_RETRY_DELAY_SECONDS
            if retry_delay_seconds is None
            else retry_delay_seconds
        )
        self.active_jobs: set[int] = set()
        self.lock = Lock()

    def enqueue(self, document_id: int) -> bool:
        with self.lock:
            if document_id in self.active_jobs:
                return False
            self.active_jobs.add(document_id)

        try:
            self.executor.submit(self.process, document_id)
            logger.info("Document processing queued", extra={"event": "document.queued", "document_id": document_id})
            return True
        except Exception:
            with self.lock:
                self.active_jobs.discard(document_id)
            logger.exception("Could not queue document processing", extra={"event": "document.queue_failed", "document_id": document_id})
            return False

    def is_active(self, document_id: int) -> bool:
        with self.lock:
            return document_id in self.active_jobs

    def process(self, document_id: int) -> None:
        session = self.session_factory()
        repository = DocumentRepository(session)

        try:
            document = repository.claim_pending(document_id)
            if document is None:
                return

            logger.info("Document processing started", extra={"event": "document.processing_started", "document_id": document_id})
            file_path = UPLOAD_DIR / Path(document.filename).name

            attempts = 0
            while True:
                try:
                    chunks = process_document(file_path)
                    if not chunks:
                        raise ValueError("No extractable content was found.")

                    embeddings = generate_embeddings([chunk["text"] for chunk in chunks])
                    VectorStore().add_chunks(chunks, embeddings)
                    repository.update_status(document_id, "ready")
                    logger.info("Document processing completed", extra={"event": "document.ready", "document_id": document_id})
                    return
                except Exception:
                    attempts += 1
                    if attempts <= self.max_retries:
                        logger.warning(
                            "Document processing failed (attempt %s of %s); retrying",
                            attempts,
                            self.max_retries,
                            extra={"event": "document.retry", "document_id": document_id},
                            exc_info=True,
                        )
                        time.sleep(self.retry_delay_seconds)
                    else:
                        raise
        except Exception:
            logger.exception("Document processing failed", extra={"event": "document.failed", "document_id": document_id})
            try:
                repository.update_status(document_id, "failed")
            except Exception:
                logger.exception("Could not mark document as failed", extra={"event": "document.status_update_failed", "document_id": document_id})
        finally:
            session.close()
            with self.lock:
                self.active_jobs.discard(document_id)

    def enqueue_pending(self) -> None:
        session = self.session_factory()
        try:
            repository = DocumentRepository(session)
            recovered = repository.recover_stale_processing(
                DOCUMENT_PROCESSING_STALE_SECONDS
            )
            if recovered:
                logger.info(
                    "Recovered %s stale processing document(s)",
                    recovered,
                    extra={"event": "document.stale_recovered"},
                )
            document_ids = [
                document.id
                for document in repository.list_pending()
            ]
        finally:
            session.close()

        for document_id in document_ids:
            self.enqueue(document_id)


def run_worker() -> None:
    while True:
        try:
            document_processing_service.enqueue_pending()
        except Exception:
            logger.exception("Could not poll pending document jobs", extra={"event": "document.poll_failed"})
        time.sleep(WORKER_POLL_INTERVAL_SECONDS)


document_processing_service = DocumentProcessingService()
