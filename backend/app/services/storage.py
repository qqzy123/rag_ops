# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from typing import BinaryIO

from app.core.config import get_settings
from app.core.errors import ApiError, ErrorCode

SUPPORTED_EXTENSIONS = {"csv", "json", "txt", "docx", "doc", "pdf", "md"}


def normalize_extension(filename: str) -> str:
    if "." not in filename:
        raise ApiError(ErrorCode.VALIDATION_ERROR, "File extension is required", 422)
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ApiError(ErrorCode.VALIDATION_ERROR, f"Unsupported file extension: {ext}", 422)
    return ext


def build_object_key(
        user_id: str,
        knowledge_base_id: str,
        file_id: str,
        filename: str,
) -> str:
    safe_name = filename.replace("\\", "_").replace("/", "_")
    return f"{user_id}/{knowledge_base_id}/{file_id}/{safe_name}"


class MinioStorage:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None

    def client(self):
        if self._client is None:
            try:
                from minio import Minio
            except ImportError as exc:
                raise ApiError(
                    ErrorCode.EXTERNAL_SERVICE_ERROR,
                    "Object storage dependency failed to load. Reinstall project dependencies.",
                    503,
                ) from exc

            self._client = Minio(
                self.settings.MINIO_ENDPOINT,
                access_key=self.settings.MINIO_ACCESS_KEY,
                secret_key=self.settings.MINIO_SECRET_KEY,
                secure=self.settings.MINIO_SECURE,
            )
        return self._client

    def ensure_bucket(self) -> None:
        client = self.client()
        if not client.bucket_exists(self.settings.MINIO_BUCKET):
            client.make_bucket(self.settings.MINIO_BUCKET)

    def put_bytes(
            self,
            object_key: str,
            data: bytes,
            content_type: str | None = None,
    ) -> None:
        self.ensure_bucket()
        self.client().put_object(
            self.settings.MINIO_BUCKET,
            object_key,
            BytesIO(data),
            length=len(data),
            content_type=content_type or "application/octet-stream",
        )

    def put_file(
            self,
            object_key: str,
            data: BinaryIO,
            length: int,
            content_type: str | None = None,
    ) -> None:
        self.ensure_bucket()
        data.seek(0)
        self.client().put_object(
            self.settings.MINIO_BUCKET,
            object_key,
            data,
            length=length,
            content_type=content_type or "application/octet-stream",
        )

    def get_bytes(
            self,
            object_key: str,
            user_id: str,
            knowledge_base_id: str,
            file_id: str,
    ) -> bytes:
        self._assert_key_scope(object_key, user_id, knowledge_base_id, file_id)
        response = self.client().get_object(self.settings.MINIO_BUCKET, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete_object(
            self,
            object_key: str,
            user_id: str,
            knowledge_base_id: str,
            file_id: str,
    ) -> None:
        self._assert_key_scope(object_key, user_id, knowledge_base_id, file_id)
        try:
            self.client().remove_object(self.settings.MINIO_BUCKET, object_key)
        except Exception:
            return

    def delete_prefix(self, prefix: str, user_id: str, knowledge_base_id: str) -> None:
        expected = f"{user_id}/{knowledge_base_id}/"
        if not prefix.startswith(expected):
            raise ApiError(ErrorCode.FORBIDDEN, "MinIO delete prefix is out of scope", 403)
        try:
            client = self.client()
            items = client.list_objects(self.settings.MINIO_BUCKET, prefix=prefix, recursive=True)
            for item in items:
                client.remove_object(self.settings.MINIO_BUCKET, item.object_name)
        except Exception:
            return

    @staticmethod
    def _assert_key_scope(
            object_key: str,
            user_id: str,
            knowledge_base_id: str,
            file_id: str,
    ) -> None:
        expected = f"{user_id}/{knowledge_base_id}/{file_id}/"
        if not object_key.startswith(expected):
            raise ApiError(ErrorCode.FORBIDDEN, "Object storage key is out of scope", 403)


storage = MinioStorage()
