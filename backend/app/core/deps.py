# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.errors import ApiError, ErrorCode
from app.core.security import decode_access_token
from app.db.session import get_db
from app.models import User

bearer = HTTPBearer(auto_error=False)


def db_session() -> Generator[Session, None, None]:
    yield from get_db()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(db_session),
) -> User:
    if credentials is None:
        raise ApiError(ErrorCode.UNAUTHORIZED, "请先登录", 401)
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise ApiError(ErrorCode.UNAUTHORIZED, "无效的登录凭证", 401)
    user = db.get(User, user_id)
    if not user or user.deleted_at is not None:
        raise ApiError(ErrorCode.UNAUTHORIZED, "用户不存在或已删除", 401)
    if user.status != "active":
        raise ApiError(ErrorCode.UNAUTHORIZED, "用户已被禁用", 401)
    if int(payload.get("token_version", -1)) != user.token_version:
        raise ApiError(ErrorCode.UNAUTHORIZED, "登录凭证已失效", 401)
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise ApiError(ErrorCode.FORBIDDEN, "需要管理员权限", 403)
    return current_user


def is_admin(user: User) -> bool:
    return user.role == "admin"
