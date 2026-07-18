# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import FileAsset, IngestionTask, KnowledgeBase, User
from app.services.ingestion import ingestion_fingerprint


ACTIVE_TASK_STATUSES = {"pending", "running"}


def _get_or_create_task(
    db: Session,
    user_id: str,
    knowledge_base_id: str | None,
    file_id: str | None,
    task_type: str,
    idempotency_key: str,
    payload: dict | None = None,
) -> IngestionTask:
    existing = db.scalar(select(IngestionTask).where(IngestionTask.idempotency_key == idempotency_key))
    if existing:
        if existing.status in ACTIVE_TASK_STATUSES:
            return existing
        existing.status = "pending"
        existing.progress = 0
        existing.stage = None
        existing.stage_status_json = None
        existing.result_json = None
        existing.message = None
        existing.error = None
        existing.payload_json = payload or {}
        existing.started_at = None
        existing.finished_at = None
        db.commit()
        db.refresh(existing)
        return existing
    task = IngestionTask(
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        file_id=file_id,
        task_type=task_type,
        idempotency_key=idempotency_key,
        status="pending",
        progress=0,
        payload_json=payload or {},
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _create_task(
    db: Session,
    user_id: str,
    knowledge_base_id: str | None,
    file_id: str | None,
    task_type: str,
    payload: dict | None = None,
) -> IngestionTask:
    task = IngestionTask(
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        file_id=file_id,
        task_type=task_type,
        idempotency_key=f"{task_type}:{user_id}:{knowledge_base_id}:{file_id}:{uuid.uuid4()}",
        status="pending",
        progress=0,
        payload_json=payload or {},
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def dispatch_ingest_file(
    db: Session,
    file_asset: FileAsset,
    current_user: User,
    strategy_id: str | None,
    model_id: str | None,
    graph_model_id: str | None,
    overwrite: bool,
    load_options: dict | None = None,
    strategy_snapshot: dict | None = None,
    enable_neo4j: bool = False,
    rerun: bool = False,
) -> IngestionTask:
    payload = {
        "strategy_id": strategy_id,
        "model_id": model_id,
        "graph_model_id": graph_model_id,
        "overwrite": overwrite,
        "load_options": load_options,
        "strategy_snapshot": strategy_snapshot,
        "enable_neo4j": enable_neo4j,
        "filename": file_asset.filename,
    }
    if rerun:
        return _create_task(db, file_asset.user_id, file_asset.knowledge_base_id, file_asset.id, "ingest_file", payload)

    fingerprint = ingestion_fingerprint(payload)
    key = f"ingest:{file_asset.user_id}:{file_asset.knowledge_base_id}:{file_asset.id}:{fingerprint}"
    return _get_or_create_task(
        db,
        file_asset.user_id,
        file_asset.knowledge_base_id,
        file_asset.id,
        "ingest_file",
        key,
        payload,
    )


def dispatch_write_target(
    db: Session,
    source_task: IngestionTask,
    current_user: User,
    target: str,
    embedding_model_id: str | None = None,
    graph_model_id: str | None = None,
) -> IngestionTask:
    payload = {
        "source_task_id": source_task.id,
        "target": target,
        "embedding_model_id": embedding_model_id,
        "graph_model_id": graph_model_id,
    }
    source_result = source_task.result_json or {}
    asset = db.get(FileAsset, source_task.file_id) if source_task.file_id else None
    kb = db.get(KnowledgeBase, source_task.knowledge_base_id) if source_task.knowledge_base_id else None
    payload["filename"] = asset.filename if asset else source_result.get("filename")
    payload["knowledge_base_name"] = kb.name if kb else source_result.get("knowledge_base_name")
    task = _create_task(
        db,
        source_task.user_id,
        source_task.knowledge_base_id,
        source_task.file_id,
        f"write_{target}",
        payload,
    )
    task.result_json = {"source_task_id": source_task.id, "target": target}
    db.commit()
    db.refresh(task)
    return task


def dispatch_delete_file(db: Session, file_asset: FileAsset, current_user: User) -> IngestionTask:
    key = f"delete-file:{file_asset.user_id}:{file_asset.knowledge_base_id}:{file_asset.id}"
    kb = db.get(KnowledgeBase, file_asset.knowledge_base_id)
    return _get_or_create_task(
        db,
        file_asset.user_id,
        file_asset.knowledge_base_id,
        file_asset.id,
        "delete_file",
        key,
        {
            "filename": file_asset.filename,
            "knowledge_base_name": kb.name if kb else None,
        },
    )


def dispatch_delete_knowledge_base(db: Session, kb: KnowledgeBase, current_user: User) -> IngestionTask:
    key = f"delete-kb:{kb.user_id}:{kb.id}"
    return _get_or_create_task(
        db,
        kb.user_id,
        kb.id,
        None,
        "delete_knowledge_base",
        key,
        {"knowledge_base_name": kb.name},
    )
