# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

from app.api.v1 import (
    auth,
    chunk_strategies,
    chunking,
    files,
    knowledge_bases,
    model_configs,
    rerank,
    retrieval,
    tasks,
    users,
)

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(knowledge_bases.router, prefix="/knowledge-bases", tags=["knowledge-bases"])
api_router.include_router(files.router, tags=["files"])
api_router.include_router(chunking.router, tags=["chunking"])
api_router.include_router(chunk_strategies.router, prefix="/chunk-strategies", tags=["chunk-strategies"])
api_router.include_router(model_configs.router, prefix="/models", tags=["models"])
api_router.include_router(retrieval.router, prefix="/retrieval", tags=["retrieval"])
api_router.include_router(rerank.router, prefix="/rerank", tags=["rerank"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
