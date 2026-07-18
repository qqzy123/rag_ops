# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.deps import db_session, get_current_user
from app.core.errors import ApiError, ErrorCode
from app.core.response import ok
from app.core.scope import scoped_get_filter, scoped_select
from app.models import ChunkStrategy, KnowledgeBase, User
from app.schemas import ChunkStrategyIn, ChunkStrategyOut

router = APIRouter()


def _normalized_strategy_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "分块策略名称不能为空", 422)
    return normalized


def _payload_values(payload: ChunkStrategyIn) -> dict:
    values = payload.model_dump()
    values["name"] = _normalized_strategy_name(payload.name)
    return values


def _ensure_unique_name(db: Session, user_id: str, name: str, exclude_id: str | None = None) -> None:
    stmt = select(ChunkStrategy.id).where(ChunkStrategy.user_id == user_id, ChunkStrategy.name == name)
    if exclude_id:
        stmt = stmt.where(ChunkStrategy.id != exclude_id)
    if db.scalar(stmt):
        raise ApiError(ErrorCode.CONFLICT, "分块策略名称已存在", 409)


def _commit_strategy(db: Session) -> None:
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ApiError(ErrorCode.CONFLICT, "分块策略名称已存在", 409) from exc


@router.get("")
def list_strategies(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(db_session)):
    items = db.scalars(scoped_select(select(ChunkStrategy), ChunkStrategy, current_user)).all()
    return ok(request, [ChunkStrategyOut.model_validate(item).model_dump(mode="json") for item in items])


@router.post("")
def create_strategy(
    payload: ChunkStrategyIn,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    values = _payload_values(payload)
    _ensure_unique_name(db, current_user.id, values["name"])
    strategy = ChunkStrategy(user_id=current_user.id, **values)
    db.add(strategy)
    _commit_strategy(db)
    db.refresh(strategy)
    return ok(request, ChunkStrategyOut.model_validate(strategy).model_dump(mode="json"))


@router.put("/{strategy_id}")
def update_strategy(
    strategy_id: str,
    payload: ChunkStrategyIn,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    strategy = db.scalar(select(ChunkStrategy).where(*scoped_get_filter(ChunkStrategy, strategy_id, current_user)))
    if not strategy:
        raise ApiError(ErrorCode.NOT_FOUND, "分块策略不存在", 404)
    values = _payload_values(payload)
    _ensure_unique_name(db, current_user.id, values["name"], exclude_id=strategy.id)
    for field, value in values.items():
        setattr(strategy, field, value)
    _commit_strategy(db)
    db.refresh(strategy)
    return ok(request, ChunkStrategyOut.model_validate(strategy).model_dump(mode="json"))


@router.delete("/{strategy_id}")
def delete_strategy(
    strategy_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    strategy = db.scalar(select(ChunkStrategy).where(*scoped_get_filter(ChunkStrategy, strategy_id, current_user)))
    if not strategy:
        return ok(request, True)
    bound = db.scalar(
        select(KnowledgeBase).where(KnowledgeBase.default_strategy_id == strategy.id, KnowledgeBase.user_id == strategy.user_id)
    )
    if bound:
        raise ApiError(ErrorCode.CONFLICT, "该策略已被知识库设为默认策略，请先解除绑定", 409)
    db.delete(strategy)
    db.commit()
    return ok(request, True)
