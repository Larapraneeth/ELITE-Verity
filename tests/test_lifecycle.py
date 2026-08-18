import threading
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.document_processing import document_processing_service, run_worker


def test_app_lifespan_starts_and_stops_document_processing_worker(monkeypatch):
    start_worker = Mock()
    shutdown = Mock()
    monkeypatch.setattr("app.main.document_processing_service.start_worker", start_worker)
    monkeypatch.setattr("app.main.document_processing_service.shutdown", shutdown)

    with TestClient(app):
        pass

    start_worker.assert_called_once()
    shutdown.assert_called_once()


def test_app_lifespan_skips_worker_when_processing_disabled(monkeypatch):
    start_worker = Mock()
    shutdown = Mock()
    monkeypatch.setattr("app.main.DOCUMENT_PROCESSING_ENABLED", False)
    monkeypatch.setattr("app.main.document_processing_service.start_worker", start_worker)
    monkeypatch.setattr("app.main.document_processing_service.shutdown", shutdown)

    with TestClient(app):
        pass

    start_worker.assert_not_called()
    shutdown.assert_called_once()


def test_app_lifespan_does_not_start_duplicate_workers():
    with TestClient(app):
        first_worker = document_processing_service._worker_thread
        assert first_worker is not None and first_worker.is_alive()

        # Re-initialising the lifecycle must not spawn a second worker thread.
        with TestClient(app):
            assert document_processing_service._worker_thread is first_worker

    assert not first_worker.is_alive()


def test_run_worker_polls_and_shuts_down_service_on_stop():
    stop_event = threading.Event()
    service = Mock()
    with patch("app.services.document_processing.document_processing_service", service):
        run_worker(stop_event)

    service._poll_loop.assert_called_once_with(stop_event)
    service.shutdown.assert_called_once()


def test_run_worker_installs_signal_handlers_when_owning_loop():
    service = Mock()
    with patch(
        "app.services.document_processing.document_processing_service", service
    ), patch("app.services.document_processing.signal.signal") as signal_handler:
        run_worker(None)

    # SIGTERM and SIGINT handlers are installed when run_worker owns the loop.
    assert signal_handler.call_count == 2
    service.shutdown.assert_called_once()
