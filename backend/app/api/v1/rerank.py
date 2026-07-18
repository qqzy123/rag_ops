# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.deps import db_session, get_current_user
from app.core.errors import ApiError, ErrorCode
from app.core.response import ok
from app.models import ModelConfig, RerankRecord, RetrievalDebugRecord, User
from app.schemas import RerankIn
from app.services.rerank_service import RerankServiceError, rerank_results

router = APIRouter()


@router.post("/compare")
def compare_rerank(
    payload: RerankIn,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    owner_id = current_user.id
    if payload.retrieval_record_id:
        record = db.get(RetrievalDebugRecord, payload.retrieval_record_id)
        if not record or (current_user.role != "admin" and record.user_id != current_user.id):
            raise ApiError(ErrorCode.NOT_FOUND, "召回记录不存在", 404)
        owner_id = record.user_id
    model = None
    if payload.rerank_model_id:
        model = db.scalar(
            select(ModelConfig).where(
                ModelConfig.id == payload.rerank_model_id,
                ModelConfig.user_id == owner_id,
                ModelConfig.model_type == "rerank",
                ModelConfig.enabled.is_(True),
            )
        )
        if not model:
            raise ApiError(ErrorCode.NOT_FOUND, "Rerank 模型不存在或未启用", 404)
    if not model:
        model = db.scalar(
            select(ModelConfig).where(
                ModelConfig.user_id == owner_id,
                ModelConfig.model_type == "rerank",
                ModelConfig.enabled.is_(True),
                ModelConfig.is_default.is_(True),
            )
        )
    try:
        after = rerank_results(payload.query, payload.results, model)
    except RerankServiceError as exc:
        raise ApiError(ErrorCode.EXTERNAL_SERVICE_ERROR, str(exc), 502) from exc
    record = RerankRecord(
        user_id=owner_id,
        retrieval_record_id=payload.retrieval_record_id,
        model_config_id=model.id if model else None,
        model_name=model.model_name if model else "local-lexical-rerank",
        before_json=payload.results,
        after_json=after,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return ok(request, {"record_id": record.id, "before": payload.results, "after": after})
