import logging
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.logging_config import RequestContextFilter


def test_request_id_is_generated_and_preserved():
    client = TestClient(app)

    generated = client.get("/health")
    assert generated.status_code == 200
    assert generated.headers["X-Request-ID"]

    preserved = client.get("/health", headers={"X-Request-ID": "correlation-123"})
    assert preserved.headers["X-Request-ID"] == "correlation-123"


def test_unhandled_error_is_logged_with_request_id():
    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CaptureHandler()
    handler.addFilter(RequestContextFilter())
    logger = logging.getLogger("app.main")
    logger.addHandler(handler)

    try:
        with patch("app.main.DocumentRepository.list", side_effect=RuntimeError("boom")):
            response = TestClient(app, raise_server_exceptions=False).get(
                "/api/documents",
                headers={"X-Request-ID": "request-error-123"},
            )
    finally:
        logger.removeHandler(handler)

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "request-error-123"
    error_record = next(record for record in records if record.msg == "Unhandled API error")
    assert error_record.request_id == "request-error-123"
