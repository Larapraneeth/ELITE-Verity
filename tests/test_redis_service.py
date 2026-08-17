from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError

from app.main import app
from app.services.redis_service import RedisService


def test_redis_service_uses_mocked_client_for_cache_and_health():
    client = Mock()
    client.ping.return_value = True
    client.get.return_value = '{"question": "What?", "answer": "Cached"}'
    client.incr.return_value = 1

    with patch("app.services.redis_service.Redis.from_url", return_value=client):
        service = RedisService("redis://cache.example:6379/0")

    key = service.cache_key("sample.pdf", "What?")
    assert key == service.cache_key("sample.pdf", "What?")
    assert key != service.cache_key("other.pdf", "What?")
    assert service.health_check() is True
    assert service.get_json(key) == {"question": "What?", "answer": "Cached"}

    service.set_json(key, {"answer": "Stored"}, 60)
    client.setex.assert_called_once_with(key, 60, '{"answer": "Stored"}')
    assert service.allow_request("127.0.0.1", 3, 60) is True
    client.expire.assert_called_once()


def test_redis_service_fails_open_when_redis_is_unavailable():
    client = Mock()
    client.ping.side_effect = ConnectionError("unavailable")
    client.incr.side_effect = ConnectionError("unavailable")

    with patch("app.services.redis_service.Redis.from_url", return_value=client):
        service = RedisService("redis://cache.example:6379/0")

    assert service.health_check() is False
    assert service.allow_request("127.0.0.1", 3, 60) is True


def test_chat_returns_cached_response_without_calling_rag(monkeypatch):
    redis_service = Mock()
    redis_service.allow_request.return_value = True
    redis_service.cache_key.return_value = "chat:cached"
    redis_service.get_json.return_value = {
        "question": "What is this document about?",
        "answer": "Cached answer.",
        "source": None,
        "distance": None,
    }
    monkeypatch.setattr("app.main.redis_service", redis_service)

    with patch("app.main.answer_question") as answer_question:
        response = TestClient(app).post(
            "/api/chat",
            json={
                "question": "What is this document about?",
                "document": "sample.pdf",
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Cached answer."
    answer_question.assert_not_called()


def test_chat_returns_429_when_redis_rate_limit_is_reached(monkeypatch):
    redis_service = Mock()
    redis_service.allow_request.return_value = False
    monkeypatch.setattr("app.main.redis_service", redis_service)

    response = TestClient(app).post(
        "/api/chat",
        json={"question": "What?", "document": "sample.pdf"},
    )

    assert response.status_code == 429
    assert response.json() == {"detail": "Too many chat requests."}
