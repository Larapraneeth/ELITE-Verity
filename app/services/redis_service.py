import hashlib
import json
import logging
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from app.config import REDIS_URL
from app.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


class RedisService:
    def __init__(self, url: str | None = REDIS_URL):
        try:
            self.client = Redis.from_url(url, decode_responses=True) if url else None
        except (RedisError, ValueError):
            self.client = None

    def health_check(self) -> bool:
        if self.client is None:
            return False

        try:
            return bool(self.client.ping())
        except RedisError:
            logger.warning("Redis health check failed", extra={"event": "redis.health_failed"})
            return False

    @staticmethod
    def cache_key(document: str, question: str) -> str:
        value = f"{document}\x00{question}".encode("utf-8")
        return f"chat:{hashlib.sha256(value).hexdigest()}"

    def get_json(self, key: str) -> dict[str, Any] | None:
        if self.client is None:
            return None

        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except (RedisError, json.JSONDecodeError, TypeError):
            logger.warning("Redis cache read failed", extra={"event": "redis.cache_read_failed"})
            return None

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        if self.client is None:
            return

        try:
            self.client.setex(key, ttl_seconds, json.dumps(value))
        except (RedisError, TypeError):
            logger.warning("Redis cache write failed", extra={"event": "redis.cache_write_failed"})

    def allow_request(self, identifier: str, limit: int, window_seconds: int) -> bool:
        if self.client is None:
            return True

        key = f"rate-limit:{identifier}"
        try:
            count = self.client.incr(key)
            if count == 1:
                self.client.expire(key, window_seconds)
            return count <= limit
        except RedisError:
            logger.warning("Redis rate-limit operation failed", extra={"event": "redis.rate_limit_failed"})
            return True


redis_service = RedisService()
