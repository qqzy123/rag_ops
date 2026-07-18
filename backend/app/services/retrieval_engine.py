# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.errors import ApiError, ErrorCode
from app.models import DocumentChunk, KnowledgeBase, ModelConfig, RetrievalDebugRecord, User
from app.schemas import RetrievalIn
from app.services.embedding import embed_texts, model_dimension
from app.services.external_stores import es_store, milvus_store, neo4j_store

logger = logging.getLogger(__name__)

RECALL_CHANNELS = ("bm25", "vector", "graph")
MODE_DEFAULT_WEIGHTS = {
    "bm25": {"bm25": 1.0, "vector": 0.0, "graph": 0.0},
    "vector": {"bm25": 0.0, "vector": 1.0, "graph": 0.0},
    "hybrid": {"bm25": 0.5, "vector": 0.5, "graph": 0.0},
    "hybrid_graph": {"bm25": 0.4, "vector": 0.4, "graph": 0.2},
}
RRF_K = 60.0


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


# 避免召回阶段的“候选池大小”
def _candidate_k(top_k: int) -> int:
    return min(max(top_k * 5, 30), 100)


def _mode_weights(recall_mode: str, weights: dict[str, float] | None) -> dict[str, float]:
    defaults = MODE_DEFAULT_WEIGHTS.get(recall_mode, MODE_DEFAULT_WEIGHTS["hybrid"])
    normalized_weights: dict[str, float] = {}
    for channel in RECALL_CHANNELS:
        default = defaults.get(channel, 0.0)
        value = (weights or {}).get(channel, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = default
        if default <= 0:
            value = 0.0
        normalized_weights[channel] = max(value, 0.0)
    # 归一化 处理得分
    total = sum(normalized_weights.values())
    if total <= 0:
        normalized_weights = dict(defaults)
        total = sum(normalized_weights.values()) or 1.0
    return {channel: value / total for channel, value in normalized_weights.items()}


def _add_rrf_scores(rank_scores: dict[str, dict[str, float]], channel: str, results: list[dict[str, Any]]) -> None:
    """
    权重：bm25=0.3,vector=0.5,graph=0.2
    bm25 里排名第 1：0.3×1/(60+1)≈0.0049
    vector 里排名第 3：0.5×1/(60+3)≈0.0079
    graph 里排名第 5：0.2×1/(60+5)≈0.0031
    """
    seen: set[str] = set()
    for rank, item in enumerate(results, start=1):
        chunk_id = item.get("chunk_id")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        rank_scores[channel][chunk_id] = 1.0 / (RRF_K + rank)


def _fallback_sql(db: Session, user_id: str, knowledge_base_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
    # 纯 SQL 关键词兜底检索：根据数据库过滤知识库，对输入 query 做空格分词模糊匹配 content 字段，按创建时间倒序取 top_k 条，返回统一格式的 chunk 结果（bm25_score 占位 0.1）
    tokens = [token for token in query.split() if token]
    stmt = select(DocumentChunk).where(DocumentChunk.user_id == user_id, DocumentChunk.knowledge_base_id == knowledge_base_id)
    if tokens:
        stmt = stmt.where(or_(*[DocumentChunk.content.like(f"%{token}%") for token in tokens]))
    rows = db.scalars(stmt.order_by(DocumentChunk.created_at.desc()).limit(top_k)).all()
    return [
        {
            "chunk_id": row.id,
            "file_id": row.file_id,
            "chunk_index": row.chunk_index,
            "content": row.content,
            "metadata": row.metadata_json or {},
            "bm25_score": 0.1,
            "recall_types": ["sql_fallback"],
        }
        for row in rows
    ]


def _hydrate_missing(db: Session, user_id: str, merged: dict[str, dict[str, Any]]) -> None:
    missing = [
        chunk_id
        for chunk_id, item in merged.items()
        if not item.get("content")
           or not item.get("metadata")
           or item.get("file_id") is None
           or item.get("chunk_index") is None
    ]
    if not missing:
        return
    rows = db.scalars(select(DocumentChunk).where(DocumentChunk.user_id == user_id, DocumentChunk.id.in_(missing))).all()
    for row in rows:
        item = merged[row.id]
        if not item.get("content"):
            item["content"] = row.content
        if item.get("file_id") is None:
            item["file_id"] = row.file_id
        if item.get("chunk_index") is None:
            item["chunk_index"] = row.chunk_index
        if not item.get("metadata"):
            item["metadata"] = row.metadata_json or {}
        if not item.get("milvus_metadata"):
            item["milvus_metadata"] = item.get("metadata") or {}


def retrieve(db: Session, current_user: User, payload: RetrievalIn) -> tuple[list[dict[str, Any]], RetrievalDebugRecord]:
    kb = db.get(KnowledgeBase, payload.knowledge_base_id)
    if not kb or (current_user.role != "admin" and kb.user_id != current_user.id):
        raise ApiError(ErrorCode.NOT_FOUND, "未找到知识库", 404)
    owner_id = kb.user_id
    model = None
    if payload.embedding_model_id:
        model = db.scalar(
            select(ModelConfig).where(
                ModelConfig.id == payload.embedding_model_id,
                ModelConfig.user_id == owner_id,
                ModelConfig.model_type == "embedding",
                ModelConfig.enabled.is_(True),
            )
        )
    if not model:
        model = db.scalar(
            select(ModelConfig).where(
                ModelConfig.user_id == owner_id,
                ModelConfig.model_type == "embedding",
                ModelConfig.enabled.is_(True),
                ModelConfig.is_default.is_(True),
            )
        )

    needs_embedding = payload.recall_mode in {"vector", "hybrid", "hybrid_graph"}
    if needs_embedding and not model:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "请先设置 embedding model", 400)

    candidate_k = _candidate_k(payload.top_k) if payload.recall_mode == "hybrid_graph" else payload.top_k
    results_by_channel: list[dict[str, Any]] = []
    rank_scores: dict[str, dict[str, float]] = defaultdict(dict)
    if payload.recall_mode in {"bm25", "hybrid", "hybrid_graph"}:
        try:
            bm25 = es_store.bm25_search(owner_id, payload.knowledge_base_id, payload.query, candidate_k)
        except Exception:
            logger.warning("Elasticsearch BM25 search failed; falling back to SQL", exc_info=True)
            bm25 = []
        # 降级
        bm25 = bm25 or _fallback_sql(db, owner_id, payload.knowledge_base_id, payload.query, candidate_k)
        # RRF 公式
        _add_rrf_scores(rank_scores, "bm25", bm25)
        results_by_channel.extend(bm25)
    if payload.recall_mode in {"vector", "hybrid", "hybrid_graph"}:
        # 查询语句向量化
        query_vector = embed_texts(model, [payload.query])[0]
        vector = milvus_store.vector_search(owner_id, payload.knowledge_base_id, query_vector, candidate_k, model_dimension(model))
        # RRF 公式
        _add_rrf_scores(rank_scores, "vector", vector)
        results_by_channel.extend(vector)
    if payload.recall_mode == "hybrid_graph":
        # 保持原顺序去重，得到向量 / BM25 粗召回的种子 chunk 列表，用于知识图谱邻域扩展检索
        candidate_chunk_ids = [
            item["chunk_id"]
            for item in results_by_channel
            if item.get("chunk_id")
        ]
        # 去重
        candidate_chunk_ids = list(dict.fromkeys(candidate_chunk_ids))
        try:
            graph = neo4j_store.graph_search(
                owner_id,
                payload.knowledge_base_id,
                payload.query,
                candidate_k,
                candidate_chunk_ids=candidate_chunk_ids or None,
                query_terms=None,
            )
        except Exception:
            logger.warning("Neo4j graph search failed; falling back to text/vector results", exc_info=True)
            graph = []
        # RRF 公式
        _add_rrf_scores(rank_scores, "graph", graph)
        results_by_channel.extend(graph)

    # 组建返回结果
    merged: dict[str, dict[str, Any]] = {}
    for item in results_by_channel:
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue
        target = merged.setdefault(
            chunk_id, {
                "chunk_id": chunk_id,
                "file_id": item.get("file_id"),
                "chunk_index": item.get("chunk_index"),
                "content": item.get("content", ""),
                "metadata": item.get("metadata") or {},
                "milvus_metadata": item.get("metadata") or {},
                "recall_types": [],
                "bm25_score": None,
                "vector_score": None,
                "graph_score": None,
            })
        # 召回方式
        for recall_type in item.get("recall_types", []):
            if recall_type not in target["recall_types"]:
                target["recall_types"].append(recall_type)
        # 原始分数
        for key in ("bm25_score", "vector_score", "graph_score"):
            if key in item:
                target[key] = item[key]
        if item.get("content"):
            target["content"] = item["content"]
    _hydrate_missing(db, owner_id, merged)

    # RRF 最终得分计算
    weights = _mode_weights(payload.recall_mode, payload.weights)
    raw_fusion_scores: dict[str, float] = {}
    final = []
    for chunk_id, item in merged.items():
        score = 0.0
        for channel in RECALL_CHANNELS:
            score += rank_scores.get(channel, {}).get(chunk_id, 0.0) * weights.get(channel, 0.0)
        raw_fusion_scores[chunk_id] = score
        item["fusion_score"] = score
        final.append(item)
    # 归一化得分
    normalized_fusion_scores = _normalize(raw_fusion_scores)
    for item in final:
        item["fusion_score"] = normalized_fusion_scores.get(item["chunk_id"], item["fusion_score"])
    final = sorted(final, key=lambda row: row["fusion_score"], reverse=True)[: payload.top_k]

    record = RetrievalDebugRecord(
        user_id=owner_id,
        knowledge_base_id=payload.knowledge_base_id,
        query=payload.query,
        recall_mode=payload.recall_mode,
        top_k=payload.top_k,
        weights_json=payload.weights,
        results_json=final,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return final, record
