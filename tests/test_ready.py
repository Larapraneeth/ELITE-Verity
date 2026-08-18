from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _patch_checks(monkeypatch, database, redis, chroma):
    monkeypatch.setattr("app.main._check_database", lambda: database)
    monkeypatch.setattr("app.main._check_redis", lambda: redis)
    monkeypatch.setattr("app.main._check_chroma", lambda: chroma)


def test_ready_returns_200_when_all_dependencies_are_healthy(monkeypatch):
    _patch_checks(monkeypatch, database=True, redis=True, chroma=True)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "redis": "ok",
        "chroma": "ok",
    }


def test_ready_returns_503_when_database_is_unhealthy(monkeypatch):
    _patch_checks(monkeypatch, database=False, redis=True, chroma=True)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": "unavailable",
        "redis": "ok",
        "chroma": "ok",
    }


def test_ready_returns_503_when_redis_is_unhealthy(monkeypatch):
    _patch_checks(monkeypatch, database=True, redis=False, chroma=True)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["redis"] == "unavailable"
    assert response.json()["database"] == "ok"
    assert response.json()["chroma"] == "ok"


def test_ready_returns_503_when_chroma_is_unhealthy(monkeypatch):
    _patch_checks(monkeypatch, database=True, redis=True, chroma=False)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["chroma"] == "unavailable"
    assert response.json()["database"] == "ok"
    assert response.json()["redis"] == "ok"
