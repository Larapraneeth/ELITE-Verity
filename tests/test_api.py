from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_invalid_request():
    response = client.post("/api/chat", json={"question": "", "document": "sample.pdf"})

    assert response.status_code == 422


@patch("app.main.answer_question")
def test_chat_returns_rag_response(mock_answer_question):
    mock_answer_question.return_value = {
        "question": "What is this document about?",
        "answer": "It is about document intelligence.",
        "source": {
            "filename": "sample.pdf",
            "page": 1,
            "section": "Overview",
            "file_type": "pdf",
            "chunk_id": "sample_page1_chunk1",
            "distance": 0.1,
        },
        "distance": 0.1,
    }

    response = client.post(
        "/api/chat",
        json={
            "question": "What is this document about?",
            "document": "sample.pdf",
        },
    )

    assert response.status_code == 200
    assert response.json() == mock_answer_question.return_value
    mock_answer_question.assert_called_once_with(
        "What is this document about?",
        "sample.pdf",
    )
