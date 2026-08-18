import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from app.database import SessionLocal
from app.config import WORKER_POLL_INTERVAL_SECONDS
from app.repositories import DocumentRepository
from app.services.document_processor import process_document
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import VectorStore


logger = logging.getLogger(__name__)
UPLOAD_DIR = Path("data/uploads")


class DocumentProcessingService:
    def __init__(self, session_factory=SessionLocal, executor=None):
        self.session_factory = session_factory
        self.executor = executor or ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="document-processing",
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
            return True
        except Exception:
            with self.lock:
                self.active_jobs.discard(document_id)
            logger.exception("Could not queue document processing")
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

            file_path = UPLOAD_DIR / Path(document.filename).name
            chunks = process_document(file_path)
            if not chunks:
                raise ValueError("No extractable content was found.")

            embeddings = generate_embeddings([chunk["text"] for chunk in chunks])
            VectorStore().add_chunks(chunks, embeddings)
            repository.update_status(document_id, "ready")
        except Exception:
            logger.exception("Document processing failed for document %s", document_id)
            try:
                repository.update_status(document_id, "failed")
            except Exception:
                logger.exception("Could not mark document %s as failed", document_id)
        finally:
            session.close()
            with self.lock:
                self.active_jobs.discard(document_id)

    def enqueue_pending(self) -> None:
        session = self.session_factory()
        try:
            document_ids = [
                document.id
                for document in DocumentRepository(session).list_pending()
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
            logger.exception("Could not poll pending document jobs")
        time.sleep(WORKER_POLL_INTERVAL_SECONDS)


document_processing_service = DocumentProcessingService()
