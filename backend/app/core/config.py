# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    APP_NAME: str = "Enterprise RAG Manage"
    API_PREFIX: str = "/api/v1"
    ENV: Literal["dev", "test", "prod"] = "dev"
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    SECRET_KEY: str = Field(default="replace-with-at-least-32-byte-secret")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 720

    DATABASE_URL: str = "mysql+pymysql://rag:rag@mysql:3306/rag_manage?charset=utf8mb4"
    TASK_CONSUMER_ENABLED: bool = True
    TASK_CONSUMER_POLL_INTERVAL_SECONDS: float = 1.0
    MAX_UPLOAD_FILE_BYTES: int = 100 * 1024 * 1024
    MAX_REQUEST_BODY_BYTES: int = 110 * 1024 * 1024

    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "rag_minio"
    MINIO_SECRET_KEY: str = "rag_minio_secret"
    MINIO_BUCKET: str = "rag-files"
    MINIO_SECURE: bool = False

    ELASTICSEARCH_URL: str = "http://elasticsearch:9200"
    ELASTICSEARCH_INDEX: str = "rag_chunks"

    MILVUS_HOST: str = "milvus"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_PREFIX: str = "rag_chunks"
    DEFAULT_EMBEDDING_DIMENSION: int = 1024

    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "rag_neo4j_secret"
    GRAPH_EXTRACTION_CONCURRENCY: int = 4

    RERANK_BATCH_SIZE: int = 16
    RERANK_MAX_DOC_CHARS: int = 4000
    BGE_RERANK_USE_FP16: bool = True
    BGE_RERANK_DEVICE: str = ""

    API_KEY_ENCRYPTION_SECRET: str = Field(default="replace-with-32-byte-secret")
    DEFAULT_TOP_K: int = 10

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.CORS_ORIGINS.split(",") if item.strip()]

    @model_validator(mode="after")
    def validate_jwt_secret_key(self) -> "Settings":
        secret_key_bytes = len(self.SECRET_KEY.encode("utf-8"))
        if self.JWT_ALGORITHM.upper() == "HS256" and secret_key_bytes < 32:
            raise ValueError("SECRET_KEY must be at least 32 bytes for HS256")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
