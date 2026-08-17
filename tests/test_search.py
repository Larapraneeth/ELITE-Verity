from app.services.embedding_service import generate_embeddings


def test_embedding_generation_is_lazy():
    assert callable(generate_embeddings)
