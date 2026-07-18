# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.deps import db_session, get_current_user, require_admin
from app.core.errors import ApiError, ErrorCode
from app.core.response import ok
from app.core.security import create_salt, hash_password_md5, verify_password
from app.models import User
from app.schemas import PasswordChangeIn, UserCreateIn, UserOut, UserUpdateIn

router = APIRouter()


@router.get("")
def list_users(request: Request, _: User = Depends(require_admin), db: Session = Depends(db_session)):
    users = db.scalars(select(User).where(User.deleted_at.is_(None)).order_by(User.created_at.desc())).all()
    return ok(request, [UserOut.model_validate(item).model_dump(mode="json") for item in users])


@router.post("")
def create_user(
    payload: UserCreateIn,
    request: Request,
    _: User = Depends(require_admin),
    db: Session = Depends(db_session),
):
    existing = db.scalar(select(User).where(User.username == payload.username))
    if existing:
        raise ApiError(ErrorCode.CONFLICT, "用户名已存在", 409)
    salt = create_salt()
    user = User(
        username=payload.username,
        nickname=payload.nickname,
        role=payload.role,
        status=payload.status,
        salt=salt,
        password_hash=hash_password_md5(payload.password, salt),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return ok(request, UserOut.model_validate(user).model_dump(mode="json"))


@router.put("/{user_id}")
def update_user(
    user_id: str,
    payload: UserUpdateIn,
    request: Request,
    _: User = Depends(require_admin),
    db: Session = Depends(db_session),
):
    user = db.get(User, user_id)
    if not user or user.deleted_at is not None:
        raise ApiError(ErrorCode.NOT_FOUND, "用户不存在", 404)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    user.token_version += 1
    db.commit()
    db.refresh(user)
    return ok(request, UserOut.model_validate(user).model_dump(mode="json"))


@router.post("/{user_id}/reset-password")
def reset_password(
    user_id: str,
    payload: PasswordChangeIn,
    request: Request,
    _: User = Depends(require_admin),
    db: Session = Depends(db_session),
):
    user = db.get(User, user_id)
    if not user or user.deleted_at is not None:
        raise ApiError(ErrorCode.NOT_FOUND, "用户不存在", 404)
    salt = create_salt()
    user.salt = salt
    user.password_hash = hash_password_md5(payload.new_password, salt)
    user.password_version += 1
    user.token_version += 1
    db.commit()
    return ok(request, True)


@router.delete("/{user_id}")
def delete_user(
    user_id: str,
    request: Request,
    _: User = Depends(require_admin),
    db: Session = Depends(db_session),
):
    user = db.get(User, user_id)
    if not user or user.deleted_at is not None:
        return ok(request, True)
    if user.role == "admin":
        raise ApiError(ErrorCode.CONFLICT, "管理员账号不能删除", 409)
    user.deleted_at = datetime.now()
    user.status = "disabled"
    user.token_version += 1
    db.commit()
    return ok(request, True)


@router.get("/me/profile")
def profile(request: Request, current_user: User = Depends(get_current_user)):
    return ok(request, UserOut.model_validate(current_user).model_dump(mode="json"))


@router.post("/me/change-password")
def change_password(
    payload: PasswordChangeIn,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    if not payload.old_password or not verify_password(payload.old_password, current_user.salt, current_user.password_hash):
        raise ApiError(ErrorCode.UNAUTHORIZED, "原密码错误", 401)
    salt = create_salt()
    current_user.salt = salt
    current_user.password_hash = hash_password_md5(payload.new_password, salt)
    current_user.password_version += 1
    current_user.token_version += 1
    db.commit()
    return ok(request, True)
