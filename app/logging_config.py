import json
import logging
from contextvars import ContextVar, Token
from datetime import UTC, datetime

from app.config import LOG_LEVEL


request_id_context: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> Token:
    return request_id_context.set(request_id)


def reset_request_id(token: Token) -> None:
    request_id_context.reset(token)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        return True


class JsonFormatter(logging.Formatter):
    fields = (
        "event",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "document_id",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_context.get()),
        }
        for field in self.fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    app_logger = logging.getLogger("app")
    if getattr(app_logger, "_elite_verity_configured", False):
        app_logger.setLevel(LOG_LEVEL)
        return

    handler = logging.StreamHandler()
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(JsonFormatter())
    app_logger.addHandler(handler)
    app_logger.setLevel(LOG_LEVEL)
    app_logger.propagate = False
    app_logger._elite_verity_configured = True
