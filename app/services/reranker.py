import hashlib
import json
from typing import Any

from app.config import CACHE_TTL_SECONDS
from app.services.redis_service import redis_service


class Reranker:
    """Locally rerank Chroma candidates without loading another model."""

    def __init__(self, cache=redis_service):
        self.cache = cache

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [word for word in text.lower().split() if len(word) > 1]

    def _score(self, question: str, candidate: dict[str, Any]) -> float:
        question_terms = set(self._terms(question))
        document_terms = set(self._terms(candidate["document"]))
        keyword_overlap = (
            len(question_terms & document_terms) / len(question_terms)
            if question_terms else 0
        )

        question_phrase = " ".join(self._terms(question))
        document_text = " ".join(self._terms(candidate["document"]))
        phrase_overlap = 1.0 if (
            len(question_phrase.split()) >= 2
            and question_phrase in document_text
        ) else 0.0

        distance = candidate.get("distance")
        semantic_similarity = (
            1 / (1 + float(distance))
            if distance is not None else 0.0
        )

        return (
            semantic_similarity * 0.55
            + keyword_overlap * 0.30
            + phrase_overlap * 0.15
        )

    @staticmethod
    def _cache_key(question: str, candidates: list[dict[str, Any]]) -> str:
        payload = [
            {
                "id": candidate["metadata"].get("chunk_id"),
                "document": candidate["document"],
                "distance": candidate.get("distance"),
            }
            for candidate in candidates
        ]
        value = json.dumps([question, payload], sort_keys=True).encode("utf-8")
        return f"rerank:{hashlib.sha256(value).hexdigest()}"

    def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        cache_key = self._cache_key(question, candidates)
        cached = self.cache.get_json(cache_key) if self.cache else None
        if cached and isinstance(cached.get("chunk_ids"), list):
            candidates_by_id = {
                candidate["metadata"].get("chunk_id"): candidate
                for candidate in candidates
            }
            cached_candidates = [
                candidates_by_id[chunk_id]
                for chunk_id in cached["chunk_ids"]
                if chunk_id in candidates_by_id
            ]
            if len(cached_candidates) == len(candidates):
                return cached_candidates

        ranked = []
        for candidate in candidates:
            ranked_candidate = dict(candidate)
            ranked_candidate["rerank_score"] = self._score(question, candidate)
            ranked.append(ranked_candidate)

        ranked.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)

        if self.cache:
            self.cache.set_json(
                cache_key,
                {
                    "chunk_ids": [
                        candidate["metadata"].get("chunk_id")
                        for candidate in ranked
                    ]
                },
                CACHE_TTL_SECONDS,
            )

        return ranked
