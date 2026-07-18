# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from fastapi import Request

from app.core.errors import ErrorCode


def ok(request: Request, data: Any = None, message: str = "success") -> dict[str, Any]:
    return {
        "code": ErrorCode.OK,
        "message": message,
        "data": data,
        "trace_id": getattr(request.state, "trace_id", ""),
    }
