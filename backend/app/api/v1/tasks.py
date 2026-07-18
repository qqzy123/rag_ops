# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.deps import db_session, get_current_user
from app.core.errors import ApiError, ErrorCode
from app.core.response import ok
from app.core.scope import scoped_get_filter, scoped_select
from app.models import FileAsset, IngestionTask, KnowledgeBase, User
from app.schemas import TaskOut, TaskWriteIn, WorkerHealthOut
from app.tasks.consumer import task_consumer
from app.tasks.dispatch import dispatch_write_target

router = APIRouter()
settings = get_settings()


def _task_count(db: Session, status: str) -> int:
    return int(db.scalar(select(func.count()).select_from(IngestionTask).where(IngestionTask.status == status)) or 0)


def _worker_health_payload(db: Session) -> dict:
    return WorkerHealthOut(
        consumer_enabled=settings.TASK_CONSUMER_ENABLED,
        consumer_running=task_consumer.is_running(),
        pending_count=_task_count(db, "pending"),
        running_count=_task_count(db, "running"),
        poll_interval_seconds=settings.TASK_CONSUMER_POLL_INTERVAL_SECONDS,
        error=task_consumer.last_error,
    ).model_dump()


def _result_summary(task: IngestionTask) -> str | None:
    result = task.result_json or {}
    if task.error:
        return task.error
    if result.get("target"):
        return f"{result.get('target')} write: {task.status}"
    chunk_count = result.get("chunk_count")
    if chunk_count is not None:
        return f"{chunk_count} chunks, {task.status}"
    return task.message


def _snapshot_value(task: IngestionTask, key: str):
    result = task.result_json or {}
    payload = task.payload_json or {}
    return result.get(key) or payload.get(key)


def _is_write_source_task(task: IngestionTask) -> bool:
    return task.task_type == "ingest_file" or task.task_type.startswith("write_")


def _latest_file_task(db: Session, file_id: str, current_user: User) -> IngestionTask | None:
    stmt = select(IngestionTask).where(IngestionTask.file_id == file_id)
    stmt = scoped_select(stmt, IngestionTask, current_user)
    return db.scalar(stmt.order_by(IngestionTask.updated_at.desc(), IngestionTask.created_at.desc()).limit(1))


def _current_tasks(db: Session, current_user: User) -> list[IngestionTask]:
    stmt = select(IngestionTask).where(
        IngestionTask.file_id.is_not(None),
        IngestionTask.task_type == "ingest_file",
    )
    stmt = scoped_select(stmt, IngestionTask, current_user)
    items = db.scalars(stmt.order_by(IngestionTask.updated_at.desc(), IngestionTask.created_at.desc())).all()
    latest_by_file: list[IngestionTask] = []
    seen_file_ids: set[str] = set()
    for item in items:
        if not item.file_id or item.file_id in seen_file_ids:
            continue
        asset = db.get(FileAsset, item.file_id)
        if not asset:
            continue
        if current_user.role != "admin" and asset.user_id != current_user.id:
            continue
        latest_by_file.append(item)
        seen_file_ids.add(item.file_id)
    return latest_by_file


def _task_payload(db: Session, task: IngestionTask, current_user: User | None = None) -> dict:
    payload = TaskOut.model_validate(task).model_dump(mode="json")
    asset = None
    if task.knowledge_base_id:
        kb = db.get(KnowledgeBase, task.knowledge_base_id)
        payload["knowledge_base_name"] = kb.name if kb else _snapshot_value(task, "knowledge_base_name")
    if task.file_id:
        asset = db.get(FileAsset, task.file_id)
        payload["filename"] = asset.filename if asset else _snapshot_value(task, "filename")
    payload["result_summary"] = _result_summary(task)
    payload["can_write"] = False
    if current_user and task.file_id and asset and _is_write_source_task(task) and task.status not in {"pending", "running"}:
        latest = _latest_file_task(db, task.file_id, current_user)
        payload["can_write"] = bool(latest and latest.id == task.id)
    return payload


@router.get("")
def list_tasks(
    request: Request,
    scope: str = "current",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    if scope not in {"current", "history"}:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "unsupported task scope", 400)
    if scope == "current":
        items = _current_tasks(db, current_user)
        return ok(request, [_task_payload(db, item, current_user) for item in items])
    items = db.scalars(
        scoped_select(select(IngestionTask), IngestionTask, current_user).order_by(
            IngestionTask.created_at.desc()
        )
    ).all()
    return ok(request, [_task_payload(db, item, current_user) for item in items])


@router.get("/worker-health")
def worker_health(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    return ok(request, _worker_health_payload(db))


@router.get("/{task_id}")
def get_task(
    task_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    task = db.scalar(select(IngestionTask).where(*scoped_get_filter(IngestionTask, task_id, current_user)))
    if not task:
        raise ApiError(ErrorCode.NOT_FOUND, "task not found", 404)
    return ok(request, _task_payload(db, task, current_user))


@router.post("/{task_id}/write/{target}")
def write_target(
    task_id: str,
    target: str,
    request: Request,
    payload: TaskWriteIn | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    if target not in {"elasticsearch", "milvus", "neo4j"}:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "unsupported write target", 400)
    task = db.scalar(select(IngestionTask).where(*scoped_get_filter(IngestionTask, task_id, current_user)))
    if not task:
        raise ApiError(ErrorCode.NOT_FOUND, "task not found", 404)
    if not task.file_id:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "task has no file to write", 400)
    if task.status in {"pending", "running"}:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "task is still running", 400)
    if not _is_write_source_task(task):
        raise ApiError(ErrorCode.VALIDATION_ERROR, "task type cannot be used for manual write", 400)
    asset = db.get(FileAsset, task.file_id)
    if not asset:
        raise ApiError(ErrorCode.NOT_FOUND, "file not found", 404)
    latest = _latest_file_task(db, task.file_id, current_user)
    if not latest or latest.id != task.id:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "history tasks cannot be used for manual write", 400)
    result = task.result_json or {}
    body = payload or TaskWriteIn()
    embedding_model_id = body.embedding_model_id or (result.get("embedding_model") or {}).get("id")
    graph_model_id = body.graph_model_id or (result.get("graph_model") or {}).get("id")
    new_task = dispatch_write_target(db, task, current_user, target, embedding_model_id, graph_model_id)
    return ok(request, _task_payload(db, new_task, current_user))
