# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class LoginIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


class PasswordChangeIn(BaseModel):
    old_password: str | None = None
    new_password: str = Field(min_length=6)


class UserCreateIn(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    nickname: str = Field(min_length=1, max_length=128)
    role: Literal["admin", "user"] = "user"
    status: Literal["active", "disabled"] = "active"
    password: str = Field(min_length=6)


class UserUpdateIn(BaseModel):
    nickname: str | None = None
    role: Literal["admin", "user"] | None = None
    status: Literal["active", "disabled"] | None = None


class UserOut(ORMModel):
    id: str
    username: str
    nickname: str
    role: str
    status: str
    password_algo: str
    password_version: int
    created_at: datetime
    updated_at: datetime
    last_login_at: datetime | None = None


class KnowledgeBaseCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    user_id: str | None = None


class KnowledgeBaseUpdateIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = None
    default_strategy_id: str | None = None
    default_embedding_model_id: str | None = None
    default_rerank_model_id: str | None = None


class KnowledgeBaseOut(ORMModel):
    id: str
    user_id: str
    name: str
    description: str | None = None
    default_strategy_id: str | None = None
    default_embedding_model_id: str | None = None
    default_rerank_model_id: str | None = None
    created_at: datetime
    updated_at: datetime


class FileOut(ORMModel):
    id: str
    user_id: str
    knowledge_base_id: str
    filename: str
    content_type: str | None = None
    extension: str
    size: int
    object_key: str
    status: str
    chunk_count: int
    metadata_json: dict | None = None
    created_at: datetime
    updated_at: datetime


class ChunkStrategyIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    file_types: list[str] | None = None
    splitter_type: str = "recursive"
    chunk_size: int = Field(default=800, ge=20, le=20000)
    chunk_overlap: int = Field(default=0, ge=0, le=5000)
    count_by_token: bool = False
    table_policy: str = "preserve"
    code_policy: str = "preserve"
    heading_policy: str = "preserve_path"
    keep_page_number: bool = True
    keep_paragraph_number: bool = True
    keep_title_path: bool = True
    metadata_rules: dict | None = None

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info):
        chunk_size = info.data.get("chunk_size", 800)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value


class ChunkStrategyOut(ChunkStrategyIn, ORMModel):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime


ModelType = Literal["embedding", "rerank", "llm"]


class ModelConfigIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    model_type: ModelType
    provider: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    model_name: str
    dimension: int | None = Field(default=None, ge=1)
    prompt_template: str | None = None
    timeout: int = Field(default=60, ge=1, le=600)
    enabled: bool = True
    is_default: bool = False


class ModelConfigPatchIn(BaseModel):
    name: str | None = None
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    dimension: int | None = Field(default=None, ge=1)
    prompt_template: str | None = None
    timeout: int | None = Field(default=None, ge=1, le=600)
    enabled: bool | None = None
    is_default: bool | None = None


class ModelConfigOut(ORMModel):
    id: str
    user_id: str
    name: str
    model_type: str
    provider: str
    base_url: str | None = None
    api_key_hint: str | None = None
    model_name: str
    dimension: int | None = None
    prompt_template: str | None = None
    timeout: int
    enabled: bool
    is_default: bool
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def normalize_legacy_embedding_provider(self) -> "ModelConfigOut":
        if self.model_type == "embedding" and self.provider == "openai_compatible":
            self.provider = "openai"
        return self


class LoaderOptionsIn(BaseModel):
    encoding: str | None = "utf-8"
    autodetect_encoding: bool = False
    jq_schema: str | None = None
    content_key: str | None = None
    is_content_key_jq_parsable: bool = False
    text_content: bool | None = None
    json_lines: bool = False
    source_column: str | None = None
    metadata_columns: list[str] = Field(default_factory=list)
    content_columns: list[str] = Field(default_factory=list)
    csv_args: dict[str, Any] | None = None
    csv_delimiter: str = Field(default=",", min_length=1, max_length=4)
    csv_skipinitialspace: bool = True
    password: str | None = None
    extract_images: bool = False
    mode: Literal["page", "single"] = "page"
    images_inner_format: Literal["text", "markdown-img", "html-img"] = "text"
    pages_delimiter: str = "\n\f"
    extraction_mode: Literal["plain", "layout"] = "plain"
    extraction_kwargs: dict[str, Any] | None = None
    preview_content_chars: int = Field(default=8000, ge=200, le=50000)


class LoaderParamOut(BaseModel):
    name: str
    type: str
    description_zh: str
    default: Any = None
    required: bool = False
    choices: list[Any] | None = None
    example: Any = None


class LoaderSpecOut(BaseModel):
    loader_type: str
    display_name_zh: str
    class_path: str
    extensions: list[str]
    params: list[LoaderParamOut]
    unsupported_params: list[dict[str, str]] = Field(default_factory=list)
    notes_zh: str = ""



class LoadedDocumentOut(BaseModel):
    index: int
    text: str
    metadata: dict[str, Any]
    char_count: int
    page_number: int | str | None = None
    source: str | None = None


class LoadPreviewIn(LoaderOptionsIn):
    limit: int = Field(default=20, ge=1, le=200)


class LoadPreviewOut(BaseModel):
    documents: list[LoadedDocumentOut]
    total_count: int
    returned_count: int
    options: dict[str, Any]
    loader: dict[str, Any]


class ChunkOut(BaseModel):
    chunk_index: int
    content: str
    char_count: int
    token_count: int
    title_path: str | None = None
    page_number: int | str | None = None
    metadata: dict[str, Any]


class ChunkPreviewIn(BaseModel):
    strategy_id: str | None = None
    strategy: ChunkStrategyIn | None = None
    load_options: LoaderOptionsIn | None = None
    limit: int = Field(default=200, ge=1, le=1000)


class ChunkPreviewOut(BaseModel):
    chunks: list[ChunkOut]
    total_count: int
    returned_count: int
    strategy_snapshot: dict[str, Any]
    load_options: dict[str, Any]
    loader: dict[str, Any]
    preview_limited: bool = False
    preview_content_chars: int | None = None
    source_document_count: int | None = None


class IngestIn(BaseModel):
    strategy_id: str | None = None
    strategy_snapshot: ChunkStrategyIn | None = None
    load_options: LoaderOptionsIn | None = None
    embedding_model_id: str | None = None
    graph_model_id: str | None = None
    enable_neo4j: bool = False
    overwrite: bool = False
    rerun: bool = False


class IngestHistoryOut(BaseModel):
    has_ingest_history: bool
    chunk_count: int
    has_chunks: bool
    has_last_ingest_result: bool


class TaskWriteIn(BaseModel):
    embedding_model_id: str | None = None
    graph_model_id: str | None = None


class WorkerHealthOut(BaseModel):
    consumer_enabled: bool
    consumer_running: bool
    pending_count: int = 0
    running_count: int = 0
    poll_interval_seconds: float
    error: str | None = None


class RetrievalIn(BaseModel):
    knowledge_base_id: str
    query: str = Field(min_length=1)
    recall_mode: Literal["bm25", "vector", "hybrid", "hybrid_graph"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)
    weights: dict[str, float] = Field(default_factory=lambda: {"bm25": 0.4, "vector": 0.4, "graph": 0.2})
    embedding_model_id: str | None = None


class RerankIn(BaseModel):
    retrieval_record_id: str | None = None
    query: str
    results: list[dict[str, Any]]
    rerank_model_id: str | None = None


class TaskOut(ORMModel):
    id: str
    user_id: str
    knowledge_base_id: str | None = None
    file_id: str | None = None
    task_type: str
    status: str
    progress: int
    stage: str | None = None
    stage_status_json: dict | None = None
    result_json: dict | None = None
    message: str | None = None
    error: str | None = None
    knowledge_base_name: str | None = None
    filename: str | None = None
    result_summary: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime



