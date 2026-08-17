from app.services.chunker import chunk_text


TEXT = """
Artificial intelligence is a field of computer science.
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks to learn patterns from data.
Natural language processing allows computers to work with human language.
Retrieval augmented generation combines information retrieval
with language model generation.
""" * 20


def test_chunk_text_returns_overlapping_chunks():
    chunks = chunk_text(TEXT)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    assert all(len(chunk) <= 1000 for chunk in chunks)
