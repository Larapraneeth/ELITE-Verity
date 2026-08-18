from unittest.mock import Mock, patch

from app.config import RAG_CONTEXT_MAX_CHARS
from app.services.rag_service import (
    build_context,
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

def test_rag_context_max_chars_config_default_is_12000():
    assert RAG_CONTEXT_MAX_CHARS == 12000


def test_build_context_unchanged_for_normal_3_5_6_chunk_contexts():
    for chunk_count in (3, 5, 6):
        candidates = [
            {
                "document": f"passage {index} " + ("x" * 900),
                "metadata": {"page": index + 1, "section": "General"},
            }
            for index in range(chunk_count)
        ]

        expected_parts = []
        for index, candidate in enumerate(candidates, start=1):
            header = (
                f"PASSAGE {index}\n"
                f"Page: {candidate['metadata'].get('page', 'Unknown')}\n"
                f"Section: {candidate['metadata'].get('section', 'General')}\n"
            )
            expected_parts.append(header + "\n" + candidate["document"])
        expected = "\n\n".join(expected_parts)

        context = build_context(candidates)

        # Normal sized contexts must stay byte-for-byte identical to the
        # pre-budget output; the budget is only a defensive upper bound.
        assert context == expected
        assert len(context) <= RAG_CONTEXT_MAX_CHARS


def test_build_context_caps_context_at_budget(monkeypatch):
    monkeypatch.setattr("app.services.rag_service.RAG_CONTEXT_MAX_CHARS", 100)
    candidates = [
        {"document": "A" * 60, "metadata": {"page": 1, "section": "Intro"}},
        {"document": "B" * 60, "metadata": {"page": 2, "section": "Body"}},
        {"document": "C" * 60, "metadata": {"page": 3, "section": "End"}},
    ]

    context = build_context(candidates)

    assert len(context) <= 100
    assert "PASSAGE 1" in context
    assert "PASSAGE 2" not in context


def test_build_context_never_cuts_a_passage(monkeypatch):
    monkeypatch.setattr("app.services.rag_service.RAG_CONTEXT_MAX_CHARS", 100)
    candidates = [
        {"document": "A" * 60, "metadata": {"page": 1, "section": "Intro"}},
        {"document": "B" * 60, "metadata": {"page": 2, "section": "Body"}},
    ]

    context = build_context(candidates)

    # The capped context contains complete passages only, never a fragment.
    assert context == "PASSAGE 1\nPage: 1\nSection: Intro\n\n" + ("A" * 60)


def test_build_context_keeps_first_passage_when_it_exceeds_budget(monkeypatch):
    monkeypatch.setattr("app.services.rag_service.RAG_CONTEXT_MAX_CHARS", 20)
    candidates = [
        {"document": "Z" * 500, "metadata": {"page": 1, "section": "General"}},
        {"document": "Y" * 500, "metadata": {"page": 2, "section": "General"}},
    ]

    context = build_context(candidates)

    assert "Z" * 500 in context
    assert "Y" * 500 not in context
    assert context.count("PASSAGE") == 1