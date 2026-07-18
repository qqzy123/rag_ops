# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select, exists
from sqlalchemy.orm import Session

from app.core.deps import db_session, get_current_user
from app.core.errors import ApiError, ErrorCode
from app.core.response import ok
from app.core.scope import owner_id_for_create, scoped_get_filter, scoped_select
from app.models import ChunkStrategy, KnowledgeBase, ModelConfig, User, FileAsset
from app.schemas import KnowledgeBaseCreateIn, KnowledgeBaseOut, KnowledgeBaseUpdateIn

router = APIRouter()


@router.get("")
def list_knowledge_bases(
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(db_session),
):
    stmt = scoped_select(select(KnowledgeBase), KnowledgeBase, current_user).order_by(KnowledgeBase.created_at.desc())
    items = db.scalars(stmt).all()
    return ok(request, [KnowledgeBaseOut.model_validate(item).model_dump(mode="json") for item in items])


@router.post("")
def create_knowledge_base(
        payload: KnowledgeBaseCreateIn,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(db_session),
):
    owner_id = owner_id_for_create(current_user, payload.user_id)
    kb = KnowledgeBase(user_id=owner_id, name=payload.name, description=payload.description)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return ok(request, KnowledgeBaseOut.model_validate(kb).model_dump(mode="json"))


@router.put("/{knowledge_base_id}")
def update_knowledge_base(
        knowledge_base_id: str,
        payload: KnowledgeBaseUpdateIn,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(db_session),
):
    kb = db.scalar(select(KnowledgeBase).where(*scoped_get_filter(KnowledgeBase, knowledge_base_id, current_user)))
    if not kb:
        raise ApiError(ErrorCode.NOT_FOUND, "知识库不存在", 404)
    values = payload.model_dump(exclude_unset=True)
    if values.get("default_strategy_id"):
        exists = db.scalar(
            select(ChunkStrategy).where(
                ChunkStrategy.id == values["default_strategy_id"],
                ChunkStrategy.user_id == kb.user_id,
            )
        )
        if not exists:
            raise ApiError(ErrorCode.NOT_FOUND, "默认分块策略不存在或不属于该用户", 404)
    model_checks = {
        "default_embedding_model_id": "embedding",
        "default_rerank_model_id": "rerank",
    }
    for field, model_type in model_checks.items():
        if values.get(field):
            exists = db.scalar(
                select(ModelConfig).where(
                    ModelConfig.id == values[field],
                    ModelConfig.user_id == kb.user_id,
                    ModelConfig.model_type == model_type,
                )
            )
            if not exists:
                raise ApiError(ErrorCode.NOT_FOUND, f"默认 {model_type} 模型不存在或不属于该用户", 404)
    for field, value in values.items():
        setattr(kb, field, value)
    db.commit()
    db.refresh(kb)
    return ok(request, KnowledgeBaseOut.model_validate(kb).model_dump(mode="json"))


@router.delete("/{knowledge_base_id}")
def delete_knowledge_base(
        knowledge_base_id: str,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(db_session),
):
    stmt = select(exists().where(FileAsset.knowledge_base_id == knowledge_base_id))
    if db.scalar(stmt):
        return ok(request, {"deleted": False, "exist_file": True})

    kb = db.scalar(select(KnowledgeBase).where(*scoped_get_filter(KnowledgeBase, knowledge_base_id, current_user)))
    if kb:
        db.delete(kb)
        db.commit()
        return ok(request, {"deleted": True})
    return ok(request, {"deleted": False})
