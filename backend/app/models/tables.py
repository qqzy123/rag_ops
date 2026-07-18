# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    nickname: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(24), nullable=False, default="user")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="active")
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    salt: Mapped[str] = mapped_column(String(64), nullable=False)
    password_algo: Mapped[str] = mapped_column(String(32), nullable=False, default="md5_salt")
    password_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class KnowledgeBase(Base, TimestampMixin):
    __tablename__ = "knowledge_bases"
    __table_args__ = (Index("ix_kb_user_name", "user_id", "name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_strategy_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    default_embedding_model_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    default_rerank_model_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    files: Mapped[list["FileAsset"]] = relationship(back_populates="knowledge_base")


class FileAsset(Base, TimestampMixin):
    __tablename__ = "files"
    __table_args__ = (
        Index("ix_files_user_kb", "user_id", "knowledge_base_id"),
        Index("ix_files_user_kb_file", "user_id", "knowledge_base_id", "id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    knowledge_base_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("knowledge_bases.id"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    extension: Mapped[str] = mapped_column(String(16), nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="uploaded")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="files")
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="file")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("file_id", "chunk_index", name="uq_file_chunk_index"),
        Index("ix_chunks_scope", "user_id", "knowledge_base_id", "file_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    file_id: Mapped[str] = mapped_column(String(36), ForeignKey("files.id"), index=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    title_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    es_doc_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    milvus_pk: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    file: Mapped["FileAsset"] = relationship(back_populates="chunks")


class ChunkStrategy(Base, TimestampMixin):
    __tablename__ = "chunk_strategies"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_strategy_user_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_types: Mapped[list | None] = mapped_column(JSON, nullable=True)
    splitter_type: Mapped[str] = mapped_column(String(64), nullable=False, default="recursive")
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False, default=800)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    count_by_token: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    table_policy: Mapped[str] = mapped_column(String(64), nullable=False, default="preserve")
    code_policy: Mapped[str] = mapped_column(String(64), nullable=False, default="preserve")
    heading_policy: Mapped[str] = mapped_column(String(64), nullable=False, default="preserve_path")
    keep_page_number: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    keep_paragraph_number: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    keep_title_path: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    metadata_rules: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ModelConfig(Base, TimestampMixin):
    __tablename__ = "model_configs"
    __table_args__ = (Index("ix_model_user_type", "user_id", "model_type"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="openai")
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_key_hint: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_template: Mapped[str | None] = mapped_column(Text, nullable=True)
    timeout: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class IngestionTask(Base, TimestampMixin):
    __tablename__ = "ingestion_tasks"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_task_idempotency_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    knowledge_base_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    file_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    task_type: Mapped[str] = mapped_column(String(48), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stage_status_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RetrievalDebugRecord(Base):
    __tablename__ = "retrieval_debug_records"
    __table_args__ = (Index("ix_retrieval_user_kb", "user_id", "knowledge_base_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    recall_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    weights_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    results_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class RerankRecord(Base):
    __tablename__ = "rerank_records"
    __table_args__ = (Index("ix_rerank_user_retrieval", "user_id", "retrieval_record_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    retrieval_record_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    model_config_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    before_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    after_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class DeleteAudit(Base):
    __tablename__ = "delete_audits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(36), nullable=False)
    scope_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="success")
    message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

