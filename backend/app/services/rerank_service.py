# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from inspect import signature
from threading import RLock
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.security import decrypt_secret
from app.models import ModelConfig


BGE_PROVIDER = "bge"
_BGE_RERANKER_CACHE: dict[tuple[str, str, bool], Any] = {}
_BGE_RERANKER_LOCK = RLock()


class RerankServiceError(RuntimeError):
    pass


def _lexical_score(query: str, content: str) -> float:
    q = Counter(query.lower().split())
    c = Counter(content.lower().split())
    if not q or not c:
        return 0.0
    overlap = sum(min(q[token], c[token]) for token in q)
    return overlap / max(sum(q.values()), 1)


def _provider(model: ModelConfig | None) -> str:
    return (getattr(model, "provider", "") or "").strip().lower()


def _lexical_rerank_results(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for item in results:
        merged = dict(item)
        merged["rerank_score"] = _lexical_score(query, item.get("content", ""))
        scored.append(merged)
    return sorted(scored, key=lambda item: item.get("rerank_score", 0.0), reverse=True)


def _http_rerank_results(query: str, results: list[dict[str, Any]], model: ModelConfig) -> list[dict[str, Any]] | None:
    endpoint = model.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    api_key = decrypt_secret(model.api_key_encrypted)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model.model_name,
        "query": query,
        "documents": [item.get("content", "") for item in results],
    }
    with httpx.Client(timeout=model.timeout or 60) as client:
        response = client.post(endpoint, headers=headers, json=payload)
        if response.status_code >= 400:
            return None
        data = response.json()
        scored = []
        for item in data.get("results", []):
            index = item.get("index", 0)
            if 0 <= index < len(results):
                merged = dict(results[index])
                merged["rerank_score"] = float(item.get("relevance_score", item.get("score", 0.0)))
                scored.append(merged)
        if scored:
            return sorted(scored, key=lambda item: item.get("rerank_score", 0.0), reverse=True)
    return None


def _build_bge_reranker(model: ModelConfig):
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RerankServiceError(
            "BGE rerank requires FlagEmbedding. Install optional dependency: uv sync --extra rerank-bge"
        ) from exc

    settings = get_settings()
    kwargs: dict[str, Any] = {"use_fp16": settings.BGE_RERANK_USE_FP16}
    device = settings.BGE_RERANK_DEVICE.strip()
    if device:
        try:
            parameters = signature(FlagReranker).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "device" in parameters:
            kwargs["device"] = device
        elif "devices" in parameters:
            kwargs["devices"] = [device]
    return FlagReranker(model.model_name, **kwargs)


def _get_bge_reranker(model: ModelConfig):
    settings = get_settings()
    key = (model.model_name, settings.BGE_RERANK_DEVICE.strip(), settings.BGE_RERANK_USE_FP16)
    with _BGE_RERANKER_LOCK:
        reranker = _BGE_RERANKER_CACHE.get(key)
        if reranker is None:
            reranker = _build_bge_reranker(model)
            _BGE_RERANKER_CACHE[key] = reranker
        return reranker


def _score_list(raw_scores: Any) -> list[float]:
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if isinstance(raw_scores, int | float):
        return [float(raw_scores)]
    return [float(score) for score in raw_scores]


def _bge_rerank_results(query: str, results: list[dict[str, Any]], model: ModelConfig) -> list[dict[str, Any]]:
    settings = get_settings()
    batch_size = max(settings.RERANK_BATCH_SIZE, 1)
    max_doc_chars = max(settings.RERANK_MAX_DOC_CHARS, 1)
    reranker = _get_bge_reranker(model)

    scored: list[dict[str, Any]] = []
    for start in range(0, len(results), batch_size):
        batch = results[start:start + batch_size]
        pairs = [[query, str(item.get("content", ""))[:max_doc_chars]] for item in batch]
        raw_scores = reranker.compute_score(pairs)
        scores = _score_list(raw_scores)
        if len(scores) != len(batch):
            raise RerankServiceError(
                f"BGE rerank returned {len(scores)} scores for {len(batch)} documents"
            )
        for item, score in zip(batch, scores, strict=True):
            merged = dict(item)
            merged["rerank_score"] = score
            scored.append(merged)
    return sorted(scored, key=lambda item: item.get("rerank_score", 0.0), reverse=True)


def test_bge_rerank_connectivity(model: ModelConfig) -> dict[str, Any]:
    reranked = _bge_rerank_results(
        "ping",
        [{"content": "ping"}, {"content": "unrelated"}],
        model,
    )
    return {
        "ok": True,
        "provider": BGE_PROVIDER,
        "model": model.model_name,
        "score": reranked[0].get("rerank_score", 0.0) if reranked else 0.0,
    }


def rerank_results(query: str, results: list[dict[str, Any]], model: ModelConfig | None) -> list[dict[str, Any]]:
    if not results:
        return []
    if model and _provider(model) == BGE_PROVIDER:
        return _bge_rerank_results(query, results, model)
    if model and model.base_url:
        reranked = _http_rerank_results(query, results, model)
        if reranked:
            return reranked
    return _lexical_rerank_results(query, results)
