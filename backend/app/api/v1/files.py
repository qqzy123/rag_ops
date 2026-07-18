# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

import uuid

from fastapi import APIRouter, Depends, File, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.deps import db_session, get_current_user
from app.core.errors import ApiError, ErrorCode
from app.core.response import ok
from app.core.scope import scoped_get_filter
from app.models import FileAsset, KnowledgeBase, User
from app.schemas import FileOut
from app.services.cleanup import delete_file_resources
from app.services.external_stores import es_store, milvus_store, neo4j_store
from app.services.storage import build_object_key, normalize_extension, storage
from app.tasks.dispatch import dispatch_delete_file

router = APIRouter()


def _upload_file_size(upload: UploadFile) -> int:
    size = getattr(upload, "size", None)
    if size is not None:
        upload.file.seek(0)
        return int(size)

    upload.file.seek(0, 2)
    size = upload.file.tell()
    upload.file.seek(0)
    return size


@router.post("/knowledge-bases/{knowledge_base_id}/files")
async def upload_file(
    knowledge_base_id: str,
    request: Request,
    upload: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    kb = db.scalar(select(KnowledgeBase).where(*scoped_get_filter(KnowledgeBase, knowledge_base_id, current_user)))
    if not kb:
        raise ApiError(ErrorCode.NOT_FOUND, "知识库不存在", 404)
    size = _upload_file_size(upload)
    max_size = get_settings().MAX_UPLOAD_FILE_BYTES
    if size > max_size:
        max_size_mib = max_size // (1024 * 1024)
        raise ApiError(ErrorCode.VALIDATION_ERROR, f"文件大小不能超过 {max_size_mib} MiB", 413)
    ext = normalize_extension(upload.filename or "")
    file_id = str(uuid.uuid4())
    object_key = build_object_key(kb.user_id, kb.id, file_id, upload.filename or f"file.{ext}")
    await run_in_threadpool(storage.put_file, object_key, upload.file, size, upload.content_type)
    asset = FileAsset(
        id=file_id,
        user_id=kb.user_id,
        knowledge_base_id=kb.id,
        filename=upload.filename or f"file.{ext}",
        content_type=upload.content_type,
        extension=ext,
        size=size,
        object_key=object_key,
        status="uploaded",
        metadata_json={"original_filename": upload.filename},
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return ok(request, FileOut.model_validate(asset).model_dump(mode="json"))


@router.get("/knowledge-bases/{knowledge_base_id}/files")
def list_files(
    knowledge_base_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    kb = db.scalar(select(KnowledgeBase).where(*scoped_get_filter(KnowledgeBase, knowledge_base_id, current_user)))
    if not kb:
        raise ApiError(ErrorCode.NOT_FOUND, "知识库不存在", 404)
    items = db.scalars(
        select(FileAsset)
        .where(FileAsset.knowledge_base_id == kb.id, FileAsset.user_id == kb.user_id)
        .order_by(FileAsset.created_at.desc())
    ).all()
    return ok(request, [FileOut.model_validate(item).model_dump(mode="json") for item in items])


@router.get("/files/{file_id}")
def get_file(
    file_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    asset = db.scalar(select(FileAsset).where(*scoped_get_filter(FileAsset, file_id, current_user)))
    if not asset:
        raise ApiError(ErrorCode.NOT_FOUND, "文件不存在", 404)
    return ok(request, FileOut.model_validate(asset).model_dump(mode="json"))


@router.delete("/files/{file_id}")
def delete_file(
    file_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    asset = db.scalar(select(FileAsset).where(*scoped_get_filter(FileAsset, file_id, current_user)))
    if not asset:
        return ok(request, {"deleted": True})
    delete_file_resources(db, asset)
    return ok(request, {"deleted": True})


@router.get("/files/{file_id}/storage-preview")
def storage_preview(
    file_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    asset = db.scalar(select(FileAsset).where(*scoped_get_filter(FileAsset, file_id, current_user)))
    if not asset:
        raise ApiError(ErrorCode.NOT_FOUND, "文件不存在", 404)
    return ok(
        request,
        {
            "elasticsearch": es_store.preview_file(asset.user_id, asset.knowledge_base_id, asset.id),
            "milvus": milvus_store.preview_file(asset.user_id, asset.knowledge_base_id, asset.id),
            "neo4j": neo4j_store.preview_file(asset.user_id, asset.knowledge_base_id, asset.id),
        },
    )
