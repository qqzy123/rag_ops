# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from cryptography.fernet import Fernet

from app.core.config import get_settings
from app.core.errors import ApiError, ErrorCode


def create_salt() -> str:
    return os.urandom(16).hex()


def hash_password_md5(password: str, salt: str) -> str:
    payload = f"{password}:{salt}".encode("utf-8")
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    return hash_password_md5(password, salt) == password_hash


def create_access_token(subject: str, role: str, token_version: int) -> str:
    settings = get_settings()
    expires = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": subject,
        "role": role,
        "token_version": token_version,
        "exp": expires,
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise ApiError(ErrorCode.UNAUTHORIZED, "登录已过期", 401) from exc
    except jwt.PyJWTError as exc:
        raise ApiError(ErrorCode.UNAUTHORIZED, "无效的登录凭证", 401) from exc


def _fernet() -> Fernet:
    secret = get_settings().API_KEY_ENCRYPTION_SECRET.encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret).digest())
    return Fernet(key)


def encrypt_secret(value: str | None) -> str | None:
    if not value:
        return None
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(value: str | None) -> str | None:
    if not value:
        return None
    return _fernet().decrypt(value.encode("utf-8")).decode("utf-8")


def secret_hint(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}***{value[-4:]}"
