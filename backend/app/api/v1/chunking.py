# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, Request
from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.core.deps import db_session, get_current_user
from app.core.errors import ApiError, ErrorCode
from app.core.response import ok
from app.core.scope import scoped_get_filter
from app.models import ChunkStrategy, FileAsset, User, DocumentChunk, IngestionTask
from app.schemas import ChunkPreviewIn, IngestHistoryOut, IngestIn, LoadPreviewIn, TaskOut
from app.services.document_processing import get_loader_specs
from app.services.ingestion import default_strategy, preview_chunks, preview_loaded_file, resolve_strategy
from app.tasks.dispatch import dispatch_ingest_file

router = APIRouter()


def _file_ingest_history(db: Session, asset: FileAsset) -> dict:
    has_chunks = bool(
        db.scalar(
            select(
                exists().where(
                    DocumentChunk.file_id == asset.id,
                    DocumentChunk.user_id == asset.user_id,
                )
            )
        )
    )
    has_last_ingest_result = bool((asset.metadata_json or {}).get("last_ingest_result"))
    return {
        "has_ingest_history": asset.chunk_count > 0 or has_chunks or has_last_ingest_result,
        "chunk_count": asset.chunk_count,
        "has_chunks": has_chunks,
        "has_last_ingest_result": has_last_ingest_result,
    }


@router.get("/document-loaders")
def document_loaders(
        request: Request,
        _current_user: User = Depends(get_current_user),
):
    return ok(request, {"loaders": get_loader_specs()})


@router.post("/files/{file_id}/load-preview")
def load_preview(
        file_id: str,
        payload: LoadPreviewIn,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(db_session),
):
    asset = db.scalar(select(FileAsset).where(*scoped_get_filter(FileAsset, file_id, current_user)))
    if not asset:
        raise ApiError(ErrorCode.NOT_FOUND, "文件不存在", 404)
    return ok(request, preview_loaded_file(db, asset, payload.model_dump(), payload.limit))


@router.post("/files/{file_id}/chunk-preview")
def chunk_preview(
        file_id: str,
        payload: ChunkPreviewIn,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(db_session),
):
    asset = db.scalar(select(FileAsset).where(*scoped_get_filter(FileAsset, file_id, current_user)))
    if not asset:
        raise ApiError(ErrorCode.NOT_FOUND, "文件不存在", 404)
    if payload.strategy:
        strategy = ChunkStrategy(user_id=asset.user_id, **payload.strategy.model_dump())
    elif payload.strategy_id:
        strategy = resolve_strategy(db, asset, payload.strategy_id)
    else:
        strategy = default_strategy(asset)
    load_options = payload.load_options.model_dump() if payload.load_options else None
    return ok(request, preview_chunks(db, asset, strategy, load_options, payload.limit))


@router.get("/files/{file_id}/ingest-history")
def ingest_history(
        file_id: str,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(db_session),
):
    asset = db.scalar(select(FileAsset).where(*scoped_get_filter(FileAsset, file_id, current_user)))
    if not asset:
        raise ApiError(ErrorCode.NOT_FOUND, "文件不存在", 404)
    return ok(request, IngestHistoryOut(**_file_ingest_history(db, asset)).model_dump())


@router.get("/files/{file_id}/active-task")
def active_task(
        file_id: str,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(db_session),
):
    asset = db.scalar(select(FileAsset).where(*scoped_get_filter(FileAsset, file_id, current_user)))
    if not asset:
        raise ApiError(ErrorCode.NOT_FOUND, "file not found", 404)
    task = db.scalar(
        select(IngestionTask)
        .where(
            IngestionTask.file_id == asset.id,
            IngestionTask.user_id == asset.user_id,
            IngestionTask.task_type == "ingest_file",
            IngestionTask.status.in_(["pending", "running"]),
        )
        .order_by(IngestionTask.created_at.desc())
        .limit(1)
    )
    if not task:
        return ok(request, None)
    payload = TaskOut.model_validate(task).model_dump(mode="json")
    payload["filename"] = asset.filename
    return ok(request, payload)


@router.post("/files/{file_id}/ingest")
def ingest(
        file_id: str,
        payload: IngestIn,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(db_session),
):
    asset = db.scalar(select(FileAsset).where(*scoped_get_filter(FileAsset, file_id, current_user)))
    if not asset:
        raise ApiError(ErrorCode.NOT_FOUND, "文件不存在", 404)
    task = dispatch_ingest_file(
        db,
        asset,
        current_user,
        payload.strategy_id,
        payload.embedding_model_id,
        payload.graph_model_id,
        payload.overwrite,
        payload.load_options.model_dump() if payload.load_options else None,
        payload.strategy_snapshot.model_dump() if payload.strategy_snapshot else None,
        payload.enable_neo4j,
        payload.rerun,
    )
    return ok(request, TaskOut.model_validate(task).model_dump(mode="json"))
