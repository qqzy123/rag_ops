# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.errors import ErrorCode, add_exception_handlers
from app.tasks.consumer import task_consumer


class RequestBodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_body_bytes <= 0:
            await self.app(scope, receive, send)
            return

        trace_id = _header_value(scope, b"x-trace-id") or str(uuid.uuid4())
        content_length = _header_value(scope, b"content-length")
        try:
            content_length_value = int(content_length) if content_length else 0
        except ValueError:
            content_length_value = 0
        if content_length_value > self.max_body_bytes:
            await self._reject(scope, receive, send, trace_id)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise RequestBodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await self._reject(scope, receive, send, trace_id)

    async def _reject(self, scope: Scope, receive: Receive, send: Send, trace_id: str) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "code": ErrorCode.VALIDATION_ERROR,
                "message": "请求体过大",
                "data": {"max_body_bytes": self.max_body_bytes},
                "trace_id": trace_id,
            },
            headers={"X-Trace-Id": trace_id},
        )
        await response(scope, receive, send)


def _header_value(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Trace-Id"] = request.state.trace_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.TASK_CONSUMER_ENABLED:
        task_consumer.start()
    try:
        yield
    finally:
        task_consumer.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.APP_NAME, version="0.1.0", lifespan=lifespan)
    app.add_middleware(TraceMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.MAX_REQUEST_BODY_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    add_exception_handlers(app)
    app.include_router(api_router, prefix=settings.API_PREFIX)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
