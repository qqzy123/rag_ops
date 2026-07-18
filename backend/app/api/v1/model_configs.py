# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.deps import db_session, get_current_user
from app.core.errors import ApiError, ErrorCode
from app.core.response import ok
from app.core.scope import scoped_get_filter, scoped_select
from app.core.security import encrypt_secret, secret_hint
from app.models import KnowledgeBase, ModelConfig, User
from app.schemas import ModelConfigIn, ModelConfigOut, ModelConfigPatchIn
from app.services.embedding import normalize_embedding_provider, test_model_connectivity
from app.services.rerank_service import RerankServiceError

router = APIRouter()


def _normalize_provider_or_error(provider: str | None) -> str:
    try:
        return normalize_embedding_provider(provider)
    except ValueError as exc:
        raise ApiError(ErrorCode.VALIDATION_ERROR, str(exc), 422) from exc


def _validate_model(
    payload: ModelConfigIn | ModelConfigPatchIn,
    existing_type: str | None = None,
    existing_dimension: int | None = None,
) -> None:
    model_type = getattr(payload, "model_type", None) or existing_type
    dimension = getattr(payload, "dimension", None)
    provider = getattr(payload, "provider", None)
    if model_type == "embedding" and dimension is None and existing_dimension is None:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "Embedding 模型必须填写 dimension", 422)
    if model_type == "embedding" and provider is not None:
        _normalize_provider_or_error(provider)


@router.get("")
def list_models(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(db_session)):
    items = db.scalars(scoped_select(select(ModelConfig), ModelConfig, current_user).order_by(ModelConfig.is_default.desc(),ModelConfig.created_at.desc())).all()
    return ok(request, [ModelConfigOut.model_validate(item).model_dump(mode="json") for item in items])


@router.post("")
def create_model(
    payload: ModelConfigIn,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    _validate_model(payload)
    if payload.is_default:
        db.execute(
            update(ModelConfig)
            .where(ModelConfig.user_id == current_user.id, ModelConfig.model_type == payload.model_type)
            .values(is_default=False)
        )
    model = ModelConfig(
        user_id=current_user.id,
        name=payload.name,
        model_type=payload.model_type,
        provider=_normalize_provider_or_error(payload.provider) if payload.model_type == "embedding" else payload.provider,
        base_url=payload.base_url,
        api_key_encrypted=encrypt_secret(payload.api_key),
        api_key_hint=secret_hint(payload.api_key),
        model_name=payload.model_name,
        dimension=payload.dimension if payload.model_type == "embedding" else None,
        prompt_template=payload.prompt_template,
        timeout=payload.timeout,
        enabled=payload.enabled,
        is_default=payload.is_default,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return ok(request, ModelConfigOut.model_validate(model).model_dump(mode="json"))


@router.put("/{model_id}")
def update_model(
    model_id: str,
    payload: ModelConfigPatchIn,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    model = db.scalar(select(ModelConfig).where(*scoped_get_filter(ModelConfig, model_id, current_user)))
    if not model:
        raise ApiError(ErrorCode.NOT_FOUND, "模型配置不存在", 404)
    _validate_model(payload, model.model_type, model.dimension)
    values = payload.model_dump(exclude_unset=True)
    if values.get("is_default"):
        db.execute(
            update(ModelConfig)
            .where(ModelConfig.user_id == model.user_id, ModelConfig.model_type == model.model_type)
            .values(is_default=False)
        )
    if "api_key" in values:
        api_key = values.pop("api_key")
        model.api_key_encrypted = encrypt_secret(api_key)
        model.api_key_hint = secret_hint(api_key)
    if model.model_type == "embedding":
        if "provider" in values:
            values["provider"] = _normalize_provider_or_error(values["provider"])
        elif model.provider == "openai_compatible":
            values["provider"] = "openai"
    else:
        values["dimension"] = None
    for field, value in values.items():
        setattr(model, field, value)
    db.commit()
    db.refresh(model)
    return ok(request, ModelConfigOut.model_validate(model).model_dump(mode="json"))


@router.post("/{model_id}/test")
def test_model(
    model_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    model = db.scalar(select(ModelConfig).where(*scoped_get_filter(ModelConfig, model_id, current_user)))
    if not model:
        raise ApiError(ErrorCode.NOT_FOUND, "模型配置不存在", 404)
    try:
        result = test_model_connectivity(model)
    except RerankServiceError as exc:
        raise ApiError(ErrorCode.EXTERNAL_SERVICE_ERROR, str(exc), 502) from exc
    return ok(request, result)


@router.delete("/{model_id}")
def delete_model(
    model_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    model = db.scalar(select(ModelConfig).where(*scoped_get_filter(ModelConfig, model_id, current_user)))
    if not model:
        return ok(request, True)
    referenced = db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.user_id == model.user_id,
            (KnowledgeBase.default_embedding_model_id == model.id) | (KnowledgeBase.default_rerank_model_id == model.id),
        )
    )
    if referenced:
        raise ApiError(ErrorCode.CONFLICT, "模型已被知识库引用，不能删除", 409)
    db.delete(model)
    db.commit()
    return ok(request, True)


