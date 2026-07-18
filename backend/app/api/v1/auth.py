# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.deps import db_session, get_current_user
from app.core.errors import ApiError, ErrorCode
from app.core.response import ok
from app.core.security import create_access_token, verify_password
from app.models import User
from app.schemas import LoginIn, UserOut

router = APIRouter()


@router.post("/login")
def login(payload: LoginIn, request: Request, db: Session = Depends(db_session)):
    user = db.scalar(select(User).where(User.username == payload.username, User.deleted_at.is_(None)))
    if not user or not verify_password(payload.password, user.salt, user.password_hash):
        raise ApiError(ErrorCode.UNAUTHORIZED, "用户名或密码错误", 401)
    if user.status != "active":
        raise ApiError(ErrorCode.UNAUTHORIZED, "用户已被禁用", 401)
    user.last_login_at = datetime.now()
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id, user.role, user.token_version)
    return ok(request, {"access_token": token, "token_type": "bearer", "user": UserOut.model_validate(user).model_dump(mode="json")})


@router.post("/logout")
def logout(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(db_session)):
    current_user.token_version += 1
    db.commit()
    return ok(request, True)


@router.get("/me")
def me(request: Request, current_user: User = Depends(get_current_user)):
    return ok(request, UserOut.model_validate(current_user).model_dump(mode="json"))
