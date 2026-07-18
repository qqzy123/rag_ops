# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import time
import traceback
from datetime import datetime, UTC
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import ApiError, ErrorCode
from app.models import ChunkStrategy, DocumentChunk, FileAsset, IngestionTask, ModelConfig
from app.services.cleanup import delete_file_resources
from app.services.document_processing import (
    DocumentProcessingError,
    LoaderOptions,
    load_documents_result,
    split_documents,
    split_texts,
    strategy_to_dict,
)
from app.services.embedding import embed_texts, model_dimension
from app.services.external_stores import es_store, milvus_store, neo4j_store
from app.services.graph_extraction import extract_graph
from app.services.storage import storage


def default_strategy(file_asset: FileAsset) -> ChunkStrategy:
    return ChunkStrategy(
        id="preview-default",
        user_id=file_asset.user_id,
        name="默认递归分块",
        file_types=[file_asset.extension],
        splitter_type="recursive",
        chunk_size=800,
        chunk_overlap=0,
        count_by_token=False,
        table_policy="preserve",
        code_policy="preserve",
        heading_policy="preserve_path",
        keep_page_number=True,
        keep_paragraph_number=True,
        keep_title_path=True,
        metadata_rules={},
    )


def strategy_from_snapshot(file_asset: FileAsset, snapshot: dict[str, Any] | None) -> ChunkStrategy:
    if not snapshot:
        return default_strategy(file_asset)
    data = dict(snapshot)
    return ChunkStrategy(
        id=data.get("id") or "confirmed-preview",
        user_id=file_asset.user_id,
        name=data.get("name") or "本次确认策略",
        description=data.get("description"),
        file_types=data.get("file_types") or [file_asset.extension],
        splitter_type=data.get("splitter_type") or "recursive",
        chunk_size=int(data.get("chunk_size") or 800),
        chunk_overlap=int(data.get("chunk_overlap") or 0),
        count_by_token=bool(data.get("count_by_token", False)),
        table_policy=data.get("table_policy") or "preserve",
        code_policy=data.get("code_policy") or "preserve",
        heading_policy=data.get("heading_policy") or "preserve_path",
        keep_page_number=bool(data.get("keep_page_number", True)),
        keep_paragraph_number=bool(data.get("keep_paragraph_number", True)),
        keep_title_path=bool(data.get("keep_title_path", True)),
        metadata_rules=data.get("metadata_rules") or {},
    )


def resolve_strategy(db: Session, file_asset: FileAsset, strategy_id: str | None) -> ChunkStrategy:
    if strategy_id:
        strategy = db.scalar(
            select(ChunkStrategy).where(ChunkStrategy.id == strategy_id, ChunkStrategy.user_id == file_asset.user_id)
        )
        if not strategy:
            raise ApiError(ErrorCode.NOT_FOUND, "分块策略不存在", 404)
        return strategy
    return default_strategy(file_asset)


def resolve_embedding_model(db: Session, file_asset: FileAsset, model_id: str | None) -> ModelConfig:
    if model_id:
        model = db.scalar(
            select(ModelConfig).where(
                ModelConfig.id == model_id,
                ModelConfig.user_id == file_asset.user_id,
                ModelConfig.model_type == "embedding",
                ModelConfig.enabled.is_(True),
            )
        )
        if not model:
            raise ApiError(ErrorCode.NOT_FOUND, "Embedding 模型不存在或未启用", 404)
        return model
    model = db.scalar(
        select(ModelConfig).where(
            ModelConfig.user_id == file_asset.user_id,
            ModelConfig.model_type == "embedding",
            ModelConfig.enabled.is_(True),
            ModelConfig.is_default.is_(True),
        )
    )
    if not model:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "请选择一个启用的 Embedding 模型", 400)
    return model


def resolve_llm_model(db: Session, file_asset: FileAsset, model_id: str | None) -> ModelConfig:
    if model_id:
        model = db.scalar(
            select(ModelConfig).where(
                ModelConfig.id == model_id,
                ModelConfig.user_id == file_asset.user_id,
                ModelConfig.model_type == "llm",
                ModelConfig.enabled.is_(True),
            )
        )
        if not model:
            raise ApiError(ErrorCode.NOT_FOUND, "LLM model not found or disabled", 404)
        return model
    model = db.scalar(
        select(ModelConfig).where(
            ModelConfig.user_id == file_asset.user_id,
            ModelConfig.model_type == "llm",
            ModelConfig.enabled.is_(True),
            ModelConfig.is_default.is_(True),
        )
    )
    if not model:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "Please select an enabled LLM extractor model", 400)
    return model


def _file_bytes(file_asset: FileAsset) -> bytes:
    return storage.get_bytes(file_asset.object_key, file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)


def _load_result(file_asset: FileAsset, load_options: dict[str, Any] | None):
    try:
        return load_documents_result(_file_bytes(file_asset), file_asset.filename, file_asset.extension, load_options)
    except DocumentProcessingError as exc:
        raise ApiError(ErrorCode.VALIDATION_ERROR, str(exc), 400) from exc


def _load(file_asset: FileAsset, load_options: dict[str, Any] | None):
    return _load_result(file_asset, load_options).documents


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _looks_like_temp_source(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    normalized = value.replace("\\", "/").lower()
    return (
        normalized.startswith("/tmp/")
        or normalized.startswith("/var/folders/")
        or "/appdata/local/temp/" in normalized
        or "/temp/tmp" in normalized
    )


def _storage_metadata(file_asset: FileAsset, metadata: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(metadata or {})
    loader_source = result.get("source")
    if loader_source and loader_source != file_asset.object_key and not _looks_like_temp_source(loader_source):
        result.setdefault("loader_source", loader_source)

    result.update(
        {
            "source": file_asset.object_key,
            "object_key": file_asset.object_key,
            "source_filename": file_asset.filename,
            "storage_backend": "minio",
            "storage_bucket": get_settings().MINIO_BUCKET,
            "file_id": file_asset.id,
            "knowledge_base_id": file_asset.knowledge_base_id,
        }
    )
    return result


def _loaded_metadata(file_asset: FileAsset, metadata: dict[str, Any] | None) -> dict[str, Any]:
    result = _storage_metadata(file_asset, metadata)
    page_number = _first_present(result.get("page_number"), result.get("page"))
    if page_number is not None:
        result["page_number"] = page_number
    return result


def _chunk_metadata(file_asset: FileAsset, chunk: dict[str, Any], dimension: int | None = None) -> dict[str, Any]:
    result = _storage_metadata(file_asset, chunk.get("metadata") or {})
    result["chunk_index"] = chunk["chunk_index"]

    page_number = _first_present(chunk.get("page_number"), result.get("page_number"), result.get("page"))
    if page_number is not None:
        result["page_number"] = page_number

    title_path = _first_present(chunk.get("title_path"), result.get("title_path"))
    if title_path is not None:
        result["title_path"] = title_path

    if dimension is not None:
        result["milvus_dimension"] = dimension
    return result


def _split_result(file_asset: FileAsset, strategy: ChunkStrategy, load_options: dict[str, Any] | None,
                  limit: int | None):
    loaded = _load_result(file_asset, load_options)
    try:
        return split_documents(loaded.documents, strategy, file_asset.extension, limit=limit), loaded.loader
    except DocumentProcessingError as exc:
        raise ApiError(ErrorCode.VALIDATION_ERROR, str(exc), 400) from exc


def _split(file_asset: FileAsset, strategy: ChunkStrategy, load_options: dict[str, Any] | None, limit: int | None):
    return _split_result(file_asset, strategy, load_options, limit)[0]


def preview_loaded_file(db: Session, file_asset: FileAsset, load_options: dict[str, Any] | None, limit: int) -> dict[
    str, Any]:
    options = LoaderOptions.from_mapping(load_options)
    loaded = _load_result(file_asset, options.as_dict())
    docs = loaded.documents
    documents = []
    for index, doc in enumerate(docs[:limit]):
        metadata = _loaded_metadata(file_asset, doc.metadata)
        page_number = _first_present(metadata.get("page_number"), metadata.get("page"))
        source = metadata.get("source")
        documents.append(
            {
                "index": index,
                "text": doc.page_content[: options.preview_content_chars],
                "metadata": metadata,
                "char_count": len(doc.page_content),
                "page_number": page_number,
                "source": source,
            }
        )
    return {
        "documents": documents,
        "total_count": len(docs),
        "returned_count": len(documents),
        "options": options.as_public_dict(),
        "loader": loaded.loader,
    }


def preview_chunks(
        db: Session,
        file_asset: FileAsset,
        strategy: ChunkStrategy,
        load_options: dict[str, Any] | None,
        limit: int,
) -> dict[str, Any]:
    options = LoaderOptions.from_mapping(load_options)
    loaded = _load_result(file_asset, options.as_dict())
    preview_limited = False
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for doc in loaded.documents:
        preview_text = doc.page_content[: options.preview_content_chars]
        preview_limited = preview_limited or len(preview_text) < len(doc.page_content)
        texts.append(preview_text)
        metadatas.append(dict(doc.metadata or {}))
    try:
        chunks = split_texts(texts, strategy, file_asset.extension, metadatas=metadatas, limit=limit)
    except DocumentProcessingError as exc:
        raise ApiError(ErrorCode.VALIDATION_ERROR, str(exc), 400) from exc
    for chunk in chunks:
        chunk["metadata"] = _chunk_metadata(file_asset, chunk)
    return {
        "chunks": chunks,
        "total_count": len(chunks),
        "returned_count": len(chunks),
        "strategy_snapshot": strategy_to_dict(strategy),
        "load_options": options.as_public_dict(),
        "loader": loaded.loader,
        "preview_limited": preview_limited,
        "preview_content_chars": options.preview_content_chars,
        "source_document_count": len(loaded.documents),
    }


def _task_stages(overwrite: bool, enable_neo4j: bool) -> dict[str, dict[str, Any]]:
    return {
        "loader": {"status": "pending", "message": "loader full data pending"},
        "chunk": {"status": "pending", "message": "chunk full data pending"},
        "elasticsearch": {"status": "pending", "message": "waiting to write all chunks to Elasticsearch"},
        "milvus": {"status": "pending", "message": "waiting to embed and write all chunks to Milvus"},
        "neo4j": {
            "status": "pending" if enable_neo4j else "skipped",
            "message": "waiting to extract triples and write Neo4j" if enable_neo4j else "Neo4j not selected",
        },
        "done": {"status": "pending", "message": "waiting to finish"},
    }


def _stage(
        db: Session,
        task: IngestionTask,
        statuses: dict[str, dict[str, Any]],
        name: str,
        progress: int,
        status: str,
        message: str,
        **extra: Any,
) -> None:
    item = statuses.setdefault(name, {})
    item.update({"status": status, "message": message, **extra})
    task.stage = name
    task.progress = progress
    task.stage_status_json = statuses
    task.message = message
    db.commit()


def _clear_file_results(db: Session, file_asset: FileAsset, strict_external: bool = False) -> None:
    es_store.delete_by_scope(file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)
    milvus_store.delete_by_scope(file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)
    neo4j_store.delete_by_scope(file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)
    db.execute(delete(DocumentChunk).where(DocumentChunk.file_id == file_asset.id,
                                           DocumentChunk.user_id == file_asset.user_id))
    file_asset.chunk_count = 0


def _external_rows(file_asset: FileAsset, rows: list[DocumentChunk]) -> list[dict[str, Any]]:
    external_rows: list[dict[str, Any]] = []
    for row in rows:
        metadata = dict(row.metadata_json or {})
        external_rows.append(
            {
                "user_id": row.user_id,
                "knowledge_base_id": row.knowledge_base_id,
                "file_id": row.file_id,
                "chunk_id": row.id,
                "chunk_index": row.chunk_index,
                "content": row.content,
                "metadata": metadata,
            }
        )
        row.es_doc_id = row.id
        row.milvus_pk = row.id
    return external_rows


def _file_chunks(db: Session, file_asset: FileAsset) -> list[DocumentChunk]:
    return db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.file_id == file_asset.id, DocumentChunk.user_id == file_asset.user_id)
        .order_by(DocumentChunk.chunk_index.asc())
    ).all()


def _write_elasticsearch(file_asset: FileAsset, external_rows: list[dict[str, Any]]) -> dict[str, Any]:
    es_store.delete_by_scope(file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)
    es_store.index_chunks(external_rows)
    return {"count": len(external_rows)}


def _write_milvus(file_asset: FileAsset, model: ModelConfig, external_rows: list[dict[str, Any]]) -> dict[str, Any]:
    milvus_store.delete_by_scope(file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)
    vectors = embed_texts(model, [row["content"] for row in external_rows])
    milvus_store.insert_chunks(model_dimension(model), external_rows, vectors)
    return {"count": len(external_rows)}


def _write_neo4j(file_asset: FileAsset, graph_model: ModelConfig, external_rows: list[dict[str, Any]]) -> dict[
    str, Any]:
    neo4j_store.delete_by_scope(file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)
    graph = extract_graph(graph_model, external_rows)
    neo4j_store.upsert_chunks(
        external_rows,
        entities_by_chunk=graph.entities_by_chunk,
        relations_by_chunk=graph.relations_by_chunk,
        raise_errors=True,
    )
    return {"count": len(external_rows), **graph.summary()}


def _write_target(
        target: str,
        file_asset: FileAsset,
        embedding_model: ModelConfig,
        external_rows: list[dict[str, Any]],
        graph_model: ModelConfig | None = None,
) -> dict[str, Any]:
    if target == "elasticsearch":
        return _write_elasticsearch(file_asset, external_rows)
    if target == "milvus":
        return _write_milvus(file_asset, embedding_model, external_rows)
    if target == "neo4j":
        if not graph_model:
            raise ApiError(ErrorCode.VALIDATION_ERROR, "Neo4j write requires an LLM extractor model", 400)
        return _write_neo4j(file_asset, graph_model, external_rows)
    raise ApiError(ErrorCode.VALIDATION_ERROR, f"Unsupported write target: {target}", 400)


def _result(
        file_asset: FileAsset,
        rows: list[DocumentChunk],
        statuses: dict[str, dict[str, Any]],
        model: ModelConfig,
        graph_model: ModelConfig | None,
        strategy: ChunkStrategy,
        load_options: dict[str, Any] | None,
        loader_info: dict[str, Any] | None,
        started: float,
) -> dict[str, Any]:
    dimension = model_dimension(model)
    return {
        "file_id": file_asset.id,
        "filename": file_asset.filename,
        "chunk_count": len(rows),
        "embedding_model": {
            "id": model.id,
            "name": model.name,
            "model_name": model.model_name,
            "dimension": dimension,
        },
        "graph_model": {
            "id": graph_model.id,
            "name": graph_model.name,
            "model_name": graph_model.model_name,
        }
        if graph_model
        else None,
        "statuses": statuses,
        "strategy_snapshot": strategy_to_dict(strategy),
        "load_options": LoaderOptions.from_mapping(load_options).as_public_dict(),
        "loader": loader_info,
        "duration_ms": int((time.perf_counter() - started) * 1000),
    }


def _mark_failed(
        db: Session,
        task: IngestionTask,
        file_asset: FileAsset,
        statuses: dict[str, dict[str, Any]],
        stage_name: str,
        message: str,
) -> None:
    statuses.setdefault(stage_name, {}).update({"status": "failed", "message": message})
    _clear_file_results(db, file_asset, strict_external=False)
    file_asset.status = "failed"
    task.status = "failed"
    task.stage = stage_name
    task.progress = max(task.progress, 1)
    task.stage_status_json = statuses
    task.error = message
    task.finished_at = datetime.now(UTC)
    task.result_json = {
        "file_id": file_asset.id,
        "filename": file_asset.filename,
        "statuses": statuses,
        "error": message,
    }
    db.commit()


def ingest_file(
        db: Session,
        task: IngestionTask,
        strategy_id: str | None,
        model_id: str | None,
        graph_model_id: str | None,
        overwrite: bool,
        load_options: dict[str, Any] | None = None,
        strategy_snapshot: dict[str, Any] | None = None,
        enable_neo4j: bool = False,
) -> None:
    file_asset = db.get(FileAsset, task.file_id)
    if not file_asset:
        task.status = "success"
        task.progress = 100
        task.message = "文件已不存在，任务按幂等成功处理"
        task.finished_at = datetime.now(UTC)
        db.commit()
        return

    strategy = strategy_from_snapshot(file_asset, strategy_snapshot) if strategy_snapshot else resolve_strategy(db, file_asset, strategy_id)
    model = resolve_embedding_model(db, file_asset, model_id)
    graph_model = resolve_llm_model(db, file_asset, graph_model_id) if enable_neo4j else None

    started = time.perf_counter()
    statuses = _task_stages(overwrite, enable_neo4j)
    task.status = "running"
    task.progress = 5
    task.error = None
    task.result_json = None
    task.stage_status_json = statuses
    task.started_at = datetime.now(UTC)
    task.finished_at = None
    file_asset.status = "chunking"
    db.commit()

    # cleanup
    if overwrite:
        try:
            _stage(db, task, statuses, "cleanup", 8, "running", "running cleanup ...")
            _clear_file_results(db, file_asset, strict_external=False)
            db.commit()
            _stage(db, task, statuses, "cleanup", 15, "success", f"cleanup success", )
        except Exception as exc:
            traceback.print_exc()
            _mark_failed(db, task, file_asset, statuses, "clean", f"cleanup failed: {exc}")
            raise ApiError(ErrorCode.EXTERNAL_SERVICE_ERROR, "failed", 502) from exc

    # loader
    try:
        _stage(db, task, statuses, "loader", 15, "running", "running loader ...")
        loaded = _load_result(file_asset, load_options)
        loader_info = loaded.loader
        _stage(db, task, statuses, "loader", 25, "success", f"loader success", count=len(loaded.documents))
    except Exception as exc:
        _mark_failed(db, task, file_asset, statuses, "loader", f"loader failed: {exc}")
        raise ApiError(ErrorCode.VALIDATION_ERROR, "failed", 400) from exc

    # chunk
    try:
        _stage(db, task, statuses, "chunk", 32, "running", "running chunk ...")
        chunks = split_documents(loaded.documents, strategy, file_asset.extension, limit=None)
        if not chunks:
            _mark_failed(db, task, file_asset, statuses, "success", "chunk failed: chunk size is empty")
            raise ApiError(ErrorCode.VALIDATION_ERROR, "failed", 400)

        dimension = model_dimension(model)
        db_rows: list[DocumentChunk] = []
        for chunk in chunks:
            metadata = _chunk_metadata(file_asset, chunk, dimension=dimension)
            row = DocumentChunk(
                user_id=file_asset.user_id,
                knowledge_base_id=file_asset.knowledge_base_id,
                file_id=file_asset.id,
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                char_count=chunk["char_count"],
                token_count=chunk["token_count"],
                title_path=chunk.get("title_path"),
                page_number=chunk.get("page_number"),
                metadata_json=metadata,
            )
            db.add(row)
            db_rows.append(row)
        db.flush()
        file_asset.chunk_count = len(db_rows)
        db.commit()
        external_rows = _external_rows(file_asset, db_rows)
        _stage(db, task, statuses, "chunk", 45, "success", f"chunk success {len(db_rows)} chunks", count=len(db_rows))
    except Exception as exc:
        _mark_failed(db, task, file_asset, statuses, "chunk", f"chunk failed: {exc}")
        raise ApiError(ErrorCode.VALIDATION_ERROR, "failed", 400) from exc

    # write
    targets = ["elasticsearch", "milvus"]
    if enable_neo4j:
        targets.append("neo4j")
    _stage(db, task, statuses, "write", 50, "running", f"running write ...")

    failures: list[str] = []
    running_progress = {"elasticsearch": 60, "milvus": 70, "neo4j": 88}
    success_progress = {"elasticsearch": 65, "milvus": 80, "neo4j": 94}
    for target in targets:
        try:
            _stage(db, task, statuses, target, running_progress[target], "running", f"{target} writing")
            write_result = _write_target(target, file_asset, model, external_rows, graph_model)
            _stage(db,task,statuses,target,success_progress[target],"success",f"{target} write complete",**write_result,)
        except Exception as exc:
            print(exc)
            message = exc.message if isinstance(exc, ApiError) else f"{target} write failed: {exc}"
            failures.append(message)
            statuses[target] = {"status": "failed", "message": message}
            task.stage = target
            task.progress = success_progress[target]
            task.stage_status_json = statuses
            task.message = message
            db.commit()

    result = _result(file_asset, db_rows, statuses, model, graph_model, strategy, load_options, loader_info, started)
    if failures:
        result["warning"] = "; ".join(failures)
        task.status = "partial_success"
        task.message = f"ingest partially complete, {len(db_rows)} chunks; {len(failures)} write target(s) failed"
        statuses["done"] = {"status": "partial_success", "message": task.message}
    else:
        task.status = "success"
        task.message = f"ingest complete, {len(db_rows)} chunks"
        statuses["done"] = {"status": "success", "message": task.message}
    file_asset.status = "ingested"
    file_asset.chunk_count = len(db_rows)
    file_asset.metadata_json = {**(file_asset.metadata_json or {}), "last_ingest_result": result}
    task.stage = "done"
    task.progress = 100
    task.stage_status_json = statuses
    task.result_json = result
    task.finished_at = datetime.utcnow()
    db.commit()


def write_task_target(
        db: Session,
        task: IngestionTask,
        target: str,
        embedding_model_id: str | None = None,
        graph_model_id: str | None = None,
) -> None:
    if target not in {"elasticsearch", "milvus", "neo4j"}:
        raise ApiError(ErrorCode.VALIDATION_ERROR, f"Unsupported write target: {target}", 400)
    file_asset = db.get(FileAsset, task.file_id)
    if not file_asset:
        task.status = "failed"
        task.error = "file not found"
        task.finished_at = datetime.now(UTC)
        db.commit()
        return

    rows = _file_chunks(db, file_asset)
    if not rows:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "file has no chunks to write", 400)

    model = resolve_embedding_model(db, file_asset, embedding_model_id)
    graph_model = resolve_llm_model(db, file_asset, graph_model_id) if target == "neo4j" else None
    external_rows = _external_rows(file_asset, rows)
    statuses = {"elasticsearch": {"status": "skipped", "message": "not selected"},
                "milvus": {"status": "skipped", "message": "not selected"},
                "neo4j": {"status": "skipped", "message": "not selected"},
                "done": {"status": "pending", "message": "waiting to finish"},
                target: {"status": "pending", "message": f"waiting to rewrite {target}"}
                }
    started = time.perf_counter()
    task.status = "running"
    task.progress = 5
    task.stage = target
    task.error = None
    task.result_json = None
    task.stage_status_json = statuses
    task.started_at = datetime.now(UTC)
    task.finished_at = None
    db.commit()

    try:
        _stage(db, task, statuses, target, 35, "running", f"rewriting {target}")
        write_result = _write_target(target, file_asset, model, external_rows, graph_model)
        _stage(db, task, statuses, target, 90, "success", f"{target} rewrite complete", **write_result)
        statuses["done"] = {"status": "success", "message": f"{target} rewrite complete"}
        task.status = "success"
        task.message = f"{target} rewrite complete"
        task.result_json = {
            "file_id": file_asset.id,
            "filename": file_asset.filename,
            "target": target,
            "chunk_count": len(rows),
            "embedding_model": {
                "id": model.id,
                "name": model.name,
                "model_name": model.model_name,
                "dimension": model_dimension(model),
            },
            "graph_model": {
                "id": graph_model.id,
                "name": graph_model.name,
                "model_name": graph_model.model_name,
            }
            if graph_model
            else None,
            "statuses": statuses,
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        message = exc.message if isinstance(exc, ApiError) else f"{target} rewrite failed: {exc}"
        statuses[target] = {"status": "failed", "message": message}
        statuses["done"] = {"status": "failed", "message": message}
        task.status = "failed"
        task.error = message
        task.result_json = {
            "file_id": file_asset.id,
            "filename": file_asset.filename,
            "target": target,
            "chunk_count": len(rows),
            "statuses": statuses,
            "error": message,
        }
    task.stage = "done"
    task.progress = 100
    task.stage_status_json = statuses
    task.finished_at = datetime.utcnow()
    db.commit()


def ingestion_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# def delete_file_by_id(db: Session, task: IngestionTask) -> None:
#     file_asset = db.get(FileAsset, task.file_id)
#     if file_asset:
#         delete_file_resources(db, file_asset)
#     task.status = "success"
#     task.progress = 100
#     task.stage = "complete"
#     task.message = "文件删除完成"
#     task.finished_at = datetime.now(UTC)
#     db.commit()
