import logging
import signal
import threading
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
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._shutdown = False

    def enqueue(self, document_id: int) -> bool:
        with self.lock:
            if self._shutdown:
                return False
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

    def _ensure_executor(self) -> None:
        """Recreate the executor after the service has been shut down."""
        if self._shutdown:
            self.executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="document-processing",
            )
            self._shutdown = False

    def start_worker(self) -> None:
        """Start the background polling worker thread.

        Idempotent: if a worker is already running this is a no-op, so
        initialising the application lifecycle more than once cannot create
        duplicate workers."""
        with self.lock:
            self._ensure_executor()
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._poll_loop,
                args=(self._stop_event,),
                name="document-processing-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop_worker(self) -> None:
        """Signal the polling worker thread to stop and wait for it to exit."""
        self._stop_event.set()
        with self.lock:
            thread = self._worker_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=WORKER_POLL_INTERVAL_SECONDS + 1)
        with self.lock:
            if self._worker_thread is thread:
                self._worker_thread = None

    def shutdown(self) -> None:
        """Stop the worker thread and shut down the executor cleanly.

        Queued jobs are cancelled; already-running jobs are allowed to finish.
        Idempotent, so it is safe to run when the worker is already stopped or
        when the application lifecycle is shut down more than once."""
        self.stop_worker()
        with self.lock:
            self._shutdown = True
            self.active_jobs.clear()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _poll_loop(self, stop_event: threading.Event) -> None:
        """Poll for pending documents until ``stop_event`` is set."""
        while not stop_event.is_set():
            stop_event.wait(WORKER_POLL_INTERVAL_SECONDS)
            if stop_event.is_set():
                break
            try:
                self.enqueue_pending()
            except Exception:
                logger.exception(
                    "Could not poll pending document jobs",
                    extra={"event": "document.poll_failed"},
                )


def run_worker(stop_event: threading.Event | None = None) -> None:
    """Run the document processing worker until told to stop.

    The Docker worker container calls this function (see docker-compose.yml).
    In the main thread, SIGTERM/SIGINT trigger a clean shutdown of the
    document processing service."""
    if stop_event is None:
        stop_event = threading.Event()
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_event.set())
            signal.signal(signal.SIGINT, lambda _signum, _frame: stop_event.set())
    try:
        document_processing_service._poll_loop(stop_event)
    finally:
        document_processing_service.shutdown()


document_processing_service = DocumentProcessingService()
