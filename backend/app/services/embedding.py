# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from app.core.config import get_settings
from app.core.security import decrypt_secret
from app.models import ModelConfig

OPENAI_PROVIDER = "openai"
LEGACY_OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"
ZHIPUAI_PROVIDER = "zhipuai"
OLLAMA_PROVIDER = "ollama"
SUPPORTED_EMBEDDING_PROVIDERS = {OPENAI_PROVIDER, ZHIPUAI_PROVIDER, OLLAMA_PROVIDER}
EMBEDDING_BATCH_SIZE = 10


def normalize_embedding_provider(provider: str | None) -> str:
    value = (provider or OPENAI_PROVIDER).strip().lower()
    if value == LEGACY_OPENAI_COMPATIBLE_PROVIDER:
        return OPENAI_PROVIDER
    if value not in SUPPORTED_EMBEDDING_PROVIDERS:
        supported = ", ".join(sorted({*SUPPORTED_EMBEDDING_PROVIDERS, LEGACY_OPENAI_COMPATIBLE_PROVIDER}))
        raise ValueError(f"Unsupported embedding provider: {provider}. Supported providers: {supported}")
    return value


def model_dimension(model: ModelConfig | None) -> int:
    if model and model.dimension:
        return model.dimension
    return get_settings().DEFAULT_EMBEDDING_DIMENSION


def _openai_embeddings_cls():
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError("required openai embedding provider") from exc
    return OpenAIEmbeddings


def _zhipuai_embeddings_cls():
    try:
        from langchain_community.embeddings import ZhipuAIEmbeddings
    except ImportError as exc:
        raise RuntimeError("required zhipuai embedding provider") from exc
    return ZhipuAIEmbeddings


def _ollama_embeddings_cls():
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise RuntimeError("required ollama embedding provider") from exc
    return OllamaEmbeddings


def _build_openai_embeddings(model: ModelConfig):
    api_key = decrypt_secret(model.api_key_encrypted)
    base_url = _normalize_openai_base_url(model.base_url)
    kwargs: dict[str, Any] = {
        "model": model.model_name,
        "api_key": api_key,
        "dimensions": model.dimension,
        "timeout": model.timeout or 60,
    }
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["check_embedding_ctx_length"] = False
    return _openai_embeddings_cls()(**kwargs)


def _normalize_openai_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    value = base_url.rstrip("/")
    if value.endswith("/embeddings"):
        return value[: -len("/embeddings")]
    return value


def _build_zhipuai_embeddings(model: ModelConfig):
    api_key = decrypt_secret(model.api_key_encrypted)
    return _zhipuai_embeddings_cls()(
        model=model.model_name,
        api_key=api_key
    )


def _build_ollama_embeddings(model: ModelConfig):
    kwargs: dict[str, Any] = {"model": model.model_name}
    if model.base_url:
        kwargs["base_url"] = model.base_url.rstrip("/")
    return _ollama_embeddings_cls()(**kwargs)


def _build_embeddings(model: ModelConfig):
    provider = normalize_embedding_provider(model.provider)
    if provider == OPENAI_PROVIDER:
        return _build_openai_embeddings(model)
    if provider == ZHIPUAI_PROVIDER:
        return _build_zhipuai_embeddings(model)
    if provider == OLLAMA_PROVIDER:
        return _build_ollama_embeddings(model)
    raise AssertionError(f"Unhandled embedding provider: {provider}")


def _validate_vectors(texts: list[str], vectors: list[list[float]], dimension: int) -> list[list[float]]:
    if len(vectors) != len(texts):
        raise ValueError(f"Embedding result count mismatch: expected {len(texts)}, got {len(vectors)}")
    for index, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise ValueError(
                f"Embedding dimension mismatch at index {index}: expected {dimension}, got {len(vector)}"
            )
    return vectors


def embed_texts(model: ModelConfig | None, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if model is None:
        raise ValueError("Embedding model is required")
    dimension = model_dimension(model)
    embeddings = _build_embeddings(model)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        vectors.extend(embeddings.embed_documents(batch))
    return _validate_vectors(texts, vectors, dimension)


def test_model_connectivity(model: ModelConfig) -> dict[str, Any]:
    if model.model_type == "embedding":
        vector = embed_texts(model, ["ping"])[0]
        return {"ok": True, "dimension": len(vector), "provider": normalize_embedding_provider(model.provider)}
    if model.model_type == "llm":
        from app.services.llm import test_llm_connectivity

        return test_llm_connectivity(model)
    if model.model_type == "rerank" and (model.provider or "").strip().lower() == "bge":
        from app.services.rerank_service import test_bge_rerank_connectivity

        return test_bge_rerank_connectivity(model)
    return {"ok": True, "mode": "local-rerank-fallback" if not model.base_url else "http-ready"}
