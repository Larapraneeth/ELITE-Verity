from unittest.mock import Mock, patch

from app.services.rag_service import (
    get_candidate_limit,
    rerank_candidates,
    retrieve_relevant_chunks,
)
from app.services.reranker import Reranker


def candidate(chunk_id: str, document: str, distance: float) -> dict:
    return {
        "document": document,
        "metadata": {"chunk_id": chunk_id, "page": 1},
        "distance": distance,
        "score": 0.0,
    }


def test_reranker_combines_semantic_keyword_and_phrase_scores_with_cache():
    cache = Mock()
    cache.get_json.return_value = None
    reranker = Reranker(cache=cache)
    candidates = [
        candidate("semantic", "An unrelated passage.", 0.1),
        candidate("lexical", "The python error handling guide.", 0.5),
    ]

    ranked = reranker.rerank("python error handling", candidates)

    assert [item["metadata"]["chunk_id"] for item in ranked] == [
        "lexical",
        "semantic",
    ]
    cache.set_json.assert_called_once()

    cache.get_json.return_value = {"chunk_ids": ["semantic", "lexical"]}
    cached_ranked = reranker.rerank("python error handling", candidates)
    assert [item["metadata"]["chunk_id"] for item in cached_ranked] == [
        "semantic",
        "lexical",
    ]


def test_retrieval_reranks_larger_candidate_set_before_final_selection():
    vector_store = Mock()
    vector_store.search.return_value = {
        "documents": [[f"passage {index}" for index in range(12)]],
        "metadatas": [[
            {"chunk_id": f"chunk-{index}", "page": index + 1}
            for index in range(12)
        ]],
        "distances": [[float(index) for index in range(12)]],
    }

    with patch(
        "app.services.rag_service.generate_embeddings",
        return_value=[[0.1]],
    ), patch(
        "app.services.rag_service.reranker.rerank",
        side_effect=lambda _question, items: list(reversed(items)),
    ):
        selected = retrieve_relevant_chunks(
            "Explain the implementation details",
            "sample.pdf",
            vector_store,
        )

    assert get_candidate_limit("Explain the implementation details") == 12
    assert vector_store.search.call_args.kwargs["n_results"] == 12
    assert [item["metadata"]["chunk_id"] for item in selected] == [
        "chunk-11",
        "chunk-10",
        "chunk-9",
    ]


def test_reranker_failure_falls_back_to_existing_candidate_order():
    candidates = [
        candidate("first", "first passage", 0.1),
        candidate("second", "second passage", 0.2),
    ]

    with patch(
        "app.services.rag_service.reranker.rerank",
        side_effect=RuntimeError("reranker unavailable"),
    ):
        assert rerank_candidates("question", candidates) == candidates
