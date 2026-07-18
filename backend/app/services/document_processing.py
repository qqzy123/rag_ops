# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


class DocumentProcessingError(Exception):
    pass


@dataclass(frozen=True)
class LoadedDocument:
    page_content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedDocumentsResult:
    documents: list[LoadedDocument]
    loader: dict[str, Any]


@dataclass(frozen=True)
class LoaderParamSpec:
    name: str
    type: str
    description_zh: str
    default: Any = None
    required: bool = False
    choices: tuple[Any, ...] | None = None
    example: Any = None

    def as_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "type": self.type,
            "description_zh": self.description_zh,
            "default": self.default,
            "required": self.required,
        }
        if self.choices is not None:
            data["choices"] = list(self.choices)
        if self.example is not None:
            data["example"] = self.example
        return data


@dataclass(frozen=True)
class LoaderSpec:
    loader_type: str
    display_name_zh: str
    class_path: str
    extensions: tuple[str, ...]
    params: tuple[LoaderParamSpec, ...]
    notes_zh: str = ""
    unsupported_params: tuple[dict[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "loader_type": self.loader_type,
            "display_name_zh": self.display_name_zh,
            "class_path": self.class_path,
            "extensions": list(self.extensions),
            "params": [param.as_dict() for param in self.params],
            "unsupported_params": list(self.unsupported_params),
            "notes_zh": self.notes_zh,
        }


@dataclass(frozen=True)
class LoaderOptions:
    encoding: str | None = "utf-8"
    autodetect_encoding: bool = False
    jq_schema: str | None = None
    content_key: str | None = None
    is_content_key_jq_parsable: bool = False
    text_content: bool | None = None
    json_lines: bool = False
    source_column: str | None = None
    metadata_columns: tuple[str, ...] = ()
    content_columns: tuple[str, ...] = ()
    csv_args: dict[str, Any] | None = None
    csv_delimiter: str = ","
    csv_skipinitialspace: bool = True
    password: str | bytes | None = None
    extract_images: bool = False
    mode: str = "page"
    images_inner_format: str = "text"
    pages_delimiter: str = "\n\f"
    extraction_mode: str = "plain"
    extraction_kwargs: dict[str, Any] | None = None
    preview_content_chars: int = 8000

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "LoaderOptions":
        values = dict(data or {})
        if values.get("csv_delimiter") == "":
            values["csv_delimiter"] = ","

        def string_tuple(value: Any) -> tuple[str, ...]:
            if not value:
                return ()
            if isinstance(value, str):
                return (value,)
            return tuple(str(item) for item in value if item is not None)

        return cls(
            encoding=values.get("encoding", "utf-8") or None,
            autodetect_encoding=bool(values.get("autodetect_encoding", False)),
            jq_schema=values.get("jq_schema") or None,
            content_key=values.get("content_key") or None,
            is_content_key_jq_parsable=bool(values.get("is_content_key_jq_parsable", False)),
            text_content=values.get("text_content") if values.get("text_content") is None else bool(values.get("text_content")),
            json_lines=bool(values.get("json_lines", False)),
            source_column=values.get("source_column") or None,
            metadata_columns=string_tuple(values.get("metadata_columns")),
            content_columns=string_tuple(values.get("content_columns")),
            csv_args=dict(values.get("csv_args") or {}) or None,
            csv_delimiter=values.get("csv_delimiter") or ",",
            csv_skipinitialspace=bool(values.get("csv_skipinitialspace", True)),
            password=values.get("password") or None,
            extract_images=bool(values.get("extract_images", False)),
            mode=values.get("mode") or "page",
            images_inner_format=values.get("images_inner_format") or "text",
            pages_delimiter=values.get("pages_delimiter") or "\n\f",
            extraction_mode=values.get("extraction_mode") or "plain",
            extraction_kwargs=dict(values.get("extraction_kwargs") or {}) or None,
            preview_content_chars=int(values.get("preview_content_chars") or 8000),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "autodetect_encoding": self.autodetect_encoding,
            "jq_schema": self.jq_schema,
            "content_key": self.content_key,
            "is_content_key_jq_parsable": self.is_content_key_jq_parsable,
            "text_content": self.text_content,
            "json_lines": self.json_lines,
            "source_column": self.source_column,
            "metadata_columns": list(self.metadata_columns),
            "content_columns": list(self.content_columns),
            "csv_args": self.csv_args,
            "csv_delimiter": self.csv_delimiter,
            "csv_skipinitialspace": self.csv_skipinitialspace,
            "password": self.password,
            "extract_images": self.extract_images,
            "mode": self.mode,
            "images_inner_format": self.images_inner_format,
            "pages_delimiter": self.pages_delimiter,
            "extraction_mode": self.extraction_mode,
            "extraction_kwargs": self.extraction_kwargs,
            "preview_content_chars": self.preview_content_chars,
        }

    def as_public_dict(self) -> dict[str, Any]:
        data = self.as_dict()
        if data.get("password"):
            data["password"] = "******"
        return data


LOADER_SPECS: dict[str, LoaderSpec] = {
    "text": LoaderSpec(
        loader_type="text",
        display_name_zh="\u6587\u672c\u52a0\u8f7d\u5668",
        class_path="langchain_community.document_loaders.TextLoader",
        extensions=("txt", "md"),
        params=(
            LoaderParamSpec("encoding", "string|null", "\u6587\u4ef6\u7f16\u7801\uff0c\u4f8b\u5982 utf-8\u3002\u4e3a\u7a7a\u65f6\u4f7f\u7528\u7cfb\u7edf\u9ed8\u8ba4\u7f16\u7801\u3002", "utf-8", example="utf-8"),
            LoaderParamSpec("autodetect_encoding", "boolean", "\u8bfb\u53d6\u5931\u8d25\u65f6\u81ea\u52a8\u8bc6\u522b\u7f16\u7801\u3002", False),
        ),
        notes_zh="\u6587\u672c\u6587\u4ef6\u4f1a\u52a0\u8f7d\u4e3a\u4e00\u4e2a Document\u3002",
    ),
    "csv": LoaderSpec(
        loader_type="csv",
        display_name_zh="CSV \u52a0\u8f7d\u5668",
        class_path="langchain_community.document_loaders.CSVLoader",
        extensions=("csv",),
        params=(
            LoaderParamSpec("encoding", "string|null", "CSV \u6587\u4ef6\u7f16\u7801\u3002", "utf-8", example="utf-8"),
            LoaderParamSpec("autodetect_encoding", "boolean", "\u8bfb\u53d6\u5931\u8d25\u65f6\u81ea\u52a8\u8bc6\u522b\u7f16\u7801\u3002", False),
            LoaderParamSpec("source_column", "string|null", "\u4f5c\u4e3a\u6765\u6e90\u7684 CSV \u5217\u540d\u3002", None, example="url"),
            LoaderParamSpec("metadata_columns", "string[]", "\u53ea\u4f5c\u4e3a\u5143\u6570\u636e\u7684 CSV \u5217\u3002", [], example=["id", "category"]),
            LoaderParamSpec("content_columns", "string[]", "\u5199\u5165\u6b63\u6587\u7684 CSV \u5217\u3002", [], example=["title", "body"]),
            LoaderParamSpec("csv_args", "object|null", "\u4f20\u7ed9 csv.DictReader \u7684\u53c2\u6570\u3002", None, example={"delimiter": ",", "skipinitialspace": True}),
            LoaderParamSpec("csv_delimiter", "string", "CSV \u5206\u9694\u7b26\uff0c\u4f1a\u5408\u5e76\u5230 csv_args.delimiter\u3002", ","),
            LoaderParamSpec("csv_skipinitialspace", "boolean", "\u662f\u5426\u5ffd\u7565\u5206\u9694\u7b26\u540e\u7a7a\u683c\u3002", True),
        ),
        notes_zh="CSV \u6bcf\u884c\u751f\u6210\u4e00\u4e2a Document\u3002",
    ),
    "json": LoaderSpec(
        loader_type="json",
        display_name_zh="JSON \u52a0\u8f7d\u5668",
        class_path="langchain_community.document_loaders.JSONLoader",
        extensions=("json",),
        params=(
            LoaderParamSpec("jq_schema", "string", "\u7528 jq \u4ece JSON \u9009\u53d6\u6570\u636e\u3002\u6570\u7ec4\u9ed8\u8ba4 .[]\uff0c\u5bf9\u8c61\u9ed8\u8ba4 .\u3002", None, True, example=".[]"),
            LoaderParamSpec("content_key", "string|null", "\u4f5c\u4e3a\u6b63\u6587\u7684 JSON \u5b57\u6bb5\u3002", None, example="desc"),
            LoaderParamSpec("is_content_key_jq_parsable", "boolean", "content_key \u662f\u5426\u6309 jq \u89e3\u6790\u3002", False),
            LoaderParamSpec("text_content", "boolean|null", "\u662f\u5426\u8981\u6c42\u6b63\u6587\u4e3a\u5b57\u7b26\u4e32\u3002", None),
            LoaderParamSpec("json_lines", "boolean", "\u6bcf\u884c\u4e00\u4e2a JSON \u5bf9\u8c61\u3002", False),
        ),
        unsupported_params=(
            {"name": "metadata_func", "reason_zh": "\u56de\u8c03\u51fd\u6570\u4e0d\u80fd\u901a\u8fc7 JSON \u4f20\u5165\u3002"},
        ),
        notes_zh="JSONLoader \u4f9d\u8d56 jq\uff1b\u7cfb\u7edf\u63d0\u4f9b\u5e38\u89c1\u7ed3\u6784\u515c\u5e95\u3002",
    ),
    "pdf": LoaderSpec(
        loader_type="pdf",
        display_name_zh="PDF \u52a0\u8f7d\u5668",
        class_path="langchain_community.document_loaders.PyPDFLoader",
        extensions=("pdf",),
        params=(
            LoaderParamSpec("password", "string|null", "PDF \u6253\u5f00\u5bc6\u7801\u3002\u8fd4\u56de\u7ed3\u679c\u4e2d\u4f1a\u8131\u654f\u3002", None),
            LoaderParamSpec("extract_images", "boolean", "\u662f\u5426\u62bd\u53d6 PDF \u56fe\u7247\u3002", False),
            LoaderParamSpec("mode", "enum", "PDF \u8f93\u51fa\u6a21\u5f0f\uff1apage \u6309\u9875\uff0csingle \u5408\u5e76\u3002", "page", choices=("page", "single")),
            LoaderParamSpec("images_inner_format", "enum", "\u56fe\u7247\u5185\u5bb9\u5199\u5165\u6587\u672c\u7684\u683c\u5f0f\u3002", "text", choices=("text", "markdown-img", "html-img")),
            LoaderParamSpec("pages_delimiter", "string", "\u5408\u5e76 PDF \u9875\u9762\u65f6\u7684\u5206\u9694\u7b26\u3002", "\n\f"),
            LoaderParamSpec("extraction_mode", "enum", "\u6587\u672c\u62bd\u53d6\u6a21\u5f0f\u3002", "plain", choices=("plain", "layout")),
            LoaderParamSpec("extraction_kwargs", "object|null", "\u5e95\u5c42\u62bd\u53d6\u5668\u7684\u9ad8\u7ea7\u53c2\u6570\u3002", None),
        ),
        unsupported_params=(
            {"name": "headers", "reason_zh": "\u5f53\u524d\u4f7f\u7528\u4e0a\u4f20\u6587\u4ef6\uff0c\u4e0d\u901a\u8fc7 URL \u4e0b\u8f7d\u3002"},
            {"name": "images_parser", "reason_zh": "\u89e3\u6790\u5668\u5bf9\u8c61\u4e0d\u80fd\u901a\u8fc7 JSON \u4f20\u5165\u3002"},
        ),
        notes_zh="PDF \u9ed8\u8ba4\u6309\u9875\u52a0\u8f7d\u3002",
    ),
    "docx": LoaderSpec(
        loader_type="docx",
        display_name_zh="Word \u52a0\u8f7d\u5668",
        class_path="langchain_community.document_loaders.Docx2txtLoader",
        extensions=("docx", "doc"),
        params=(),
        notes_zh="Docx2txtLoader \u53ea\u652f\u6301 file_path\uff1b\u4f20\u7edf .doc \u9700\u8f6c\u4e3a .docx\u3002",
    ),
}

EXTENSION_TO_LOADER = {
    "txt": "text",
    "md": "text",
    "csv": "csv",
    "json": "json",
    "pdf": "pdf",
    "docx": "docx",
    "doc": "docx",
}

PRIMARY_EXTENSION = {
    "text": "txt",
    "csv": "csv",
    "json": "json",
    "pdf": "pdf",
    "docx": "docx",
}

BINARY_DOC_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")


def get_loader_specs() -> list[dict[str, Any]]:
    return [spec.as_dict() for spec in LOADER_SPECS.values()]


def _normalized_extension(filename: str, extension: str | None) -> str:
    ext = (extension or "").lower().lstrip(".")
    if not ext and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
    return ext


def _is_docx_zip(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return False
    return "word/document.xml" in names or any(name.startswith("word/") for name in names)


def _decode_text_sample(data: bytes) -> str | None:
    sample = data[:65536]
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return sample.decode(encoding)
        except UnicodeDecodeError:
            continue
    if b"\\x00" not in sample:
        return sample.decode("utf-8", errors="replace")
    return None


def _looks_like_csv(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    sample = "\n".join(lines[:8])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;	|")
        rows = list(csv.reader(lines[:3], dialect))
    except csv.Error:
        return False
    return bool(rows and len(rows[0]) > 1 and all(len(row) == len(rows[0]) for row in rows if row))


def _detect_from_content(data: bytes) -> tuple[str | None, str | None]:
    if data.startswith(b"%PDF-"):
        return "pdf", "pdf"
    if data.startswith(BINARY_DOC_SIGNATURE):
        return "binary_doc", "doc"
    if data.startswith(b"PK") and _is_docx_zip(data):
        return "docx", "docx"

    text = _decode_text_sample(data)
    if text is None:
        return None, None

    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(text)
            return "json", "json"
        except json.JSONDecodeError:
            pass
    if _looks_like_csv(text):
        return "csv", "csv"
    return "text", "txt"


def _select_loader(data: bytes, filename: str, extension: str | None) -> dict[str, Any]:
    declared_extension = _normalized_extension(filename, extension)
    declared_loader = EXTENSION_TO_LOADER.get(declared_extension)
    content_loader, content_extension = _detect_from_content(data)

    if content_loader == "binary_doc":
        raise DocumentProcessingError("\u4f20\u7edf\u4e8c\u8fdb\u5236 .doc \u6682\u4e0d\u652f\u6301\uff0c\u8bf7\u5148\u8f6c\u6362\u4e3a .docx \u540e\u518d\u4e0a\u4f20")

    if content_loader in {"pdf", "docx"} and declared_loader != content_loader:
        loader_type = content_loader
        detection_source = "content"
    elif declared_loader:
        loader_type = declared_loader
        detection_source = "extension"
    elif content_loader:
        loader_type = content_loader
        detection_source = "content"
    else:
        suffix = f".{declared_extension}" if declared_extension else "\u672a\u77e5\u540e\u7f00"
        raise DocumentProcessingError(f"\u6682\u4e0d\u652f\u6301\u8be5\u6587\u4ef6\u7c7b\u578b\uff1a{suffix}")

    return {
        "loader_type": loader_type,
        "declared_extension": declared_extension or None,
        "detected_extension": content_extension or declared_extension or PRIMARY_EXTENSION[loader_type],
        "detection_source": detection_source,
        "display_name_zh": LOADER_SPECS[loader_type].display_name_zh,
    }


def _csv_args(options: LoaderOptions) -> dict[str, Any]:
    args = dict(options.csv_args or {})
    args.setdefault("delimiter", options.csv_delimiter)
    args.setdefault("skipinitialspace", options.csv_skipinitialspace)
    return args


def _json_text_content(options: LoaderOptions) -> bool:
    if options.text_content is not None:
        return options.text_content
    return bool(options.content_key)


def _validate_pdf_options(options: LoaderOptions) -> None:
    if options.mode not in {"page", "single"}:
        raise DocumentProcessingError("PDF \u53c2\u6570 mode \u4ec5\u652f\u6301 page \u6216 single")
    if options.images_inner_format not in {"text", "markdown-img", "html-img"}:
        raise DocumentProcessingError("PDF \u53c2\u6570 images_inner_format \u4ec5\u652f\u6301 text\u3001markdown-img\u3001html-img")
    if options.extraction_mode not in {"plain", "layout"}:
        raise DocumentProcessingError("PDF \u53c2\u6570 extraction_mode \u4ec5\u652f\u6301 plain \u6216 layout")


def _loader_result_info(loader_info: dict[str, Any], options: LoaderOptions) -> dict[str, Any]:
    return {**loader_info, "options": options.as_public_dict()}



def _strategy_get(strategy: Any, key: str, default: Any = None) -> Any:
    if isinstance(strategy, dict):
        return strategy.get(key, default)
    return getattr(strategy, key, default)


def strategy_to_dict(strategy: Any) -> dict[str, Any]:
    fields = [
        "id",
        "name",
        "description",
        "file_types",
        "splitter_type",
        "chunk_size",
        "chunk_overlap",
        "count_by_token",
        "table_policy",
        "code_policy",
        "heading_policy",
        "keep_page_number",
        "keep_paragraph_number",
        "keep_title_path",
        "metadata_rules",
    ]
    return {field: _strategy_get(strategy, field) for field in fields if _strategy_get(strategy, field) is not None}


def _with_tempfile(data: bytes, extension: str, loader_factory) -> list[Document]:
    suffix = f".{extension.lstrip('.') or 'txt'}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        loader = loader_factory(path)
        return list(loader.load())
    finally:
        Path(path).unlink(missing_ok=True)


def _default_json_schema(data: bytes, options: LoaderOptions) -> str:
    if options.jq_schema:
        return options.jq_schema
    if options.json_lines:
        return "."
    try:
        parsed = json.loads(data.decode(options.encoding or "utf-8"))
    except Exception:
        return ".[]"
    return ".[]" if isinstance(parsed, list) else "."


def _fallback_json_documents(data: bytes, filename: str, options: LoaderOptions) -> list[LoadedDocument]:
    encoding = options.encoding or "utf-8"
    raw = data.decode(encoding, errors="replace")
    parsed = [json.loads(line) for line in raw.splitlines() if line.strip()] if options.json_lines else json.loads(raw)

    schema = options.jq_schema or (".[]" if isinstance(parsed, list) else ".")
    if schema in {".[]", ".[ ]"} and isinstance(parsed, list):
        items = parsed
    elif schema == ".data[]" and isinstance(parsed, dict):
        items = parsed.get("data") or []
    elif schema == ".":
        items = [parsed]
    else:
        raise DocumentProcessingError("JSONLoader 依赖 jq，当前环境未安装 jq；仅支持 .、.[]、.data[] 的内置兜底解析")

    docs: list[LoadedDocument] = []
    for index, item in enumerate(items):
        content: Any = item
        if options.content_key:
            key = options.content_key.strip(".")
            for part in key.split("."):
                if isinstance(content, dict):
                    content = content.get(part)
                else:
                    content = None
                    break
        page_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        docs.append(
            LoadedDocument(
                page_content=page_content,
                metadata={"source": filename, "row": index, "json_schema": schema},
            )
        )
    return docs


def _from_langchain(documents: list[Document], filename: str) -> list[LoadedDocument]:
    loaded: list[LoadedDocument] = []
    for index, doc in enumerate(documents):
        metadata = dict(doc.metadata or {})
        metadata.setdefault("source", metadata.get("source") or filename)
        metadata.setdefault("document_index", index)
        if "page" in metadata and "page_number" not in metadata:
            try:
                metadata["page_number"] = int(metadata["page"]) + 1
            except Exception:
                metadata["page_number"] = metadata["page"]
        loaded.append(LoadedDocument(page_content=doc.page_content or "", metadata=metadata))
    return loaded


def load_documents_result(
    data: bytes,
    filename: str,
    extension: str,
    options: dict[str, Any] | LoaderOptions | None = None,
) -> LoadedDocumentsResult:
    opts = options if isinstance(options, LoaderOptions) else LoaderOptions.from_mapping(options)
    loader_info = _select_loader(data, filename, extension)
    loader_type = loader_info["loader_type"]
    temp_extension = loader_info.get("detected_extension") or loader_info.get("declared_extension") or PRIMARY_EXTENSION[loader_type]

    try:
        if loader_type == "text":
            from langchain_community.document_loaders import TextLoader

            docs = _with_tempfile(
                data,
                temp_extension,
                lambda path: TextLoader(
                    file_path=path,
                    encoding=opts.encoding,
                    autodetect_encoding=opts.autodetect_encoding,
                ),
            )
            return LoadedDocumentsResult(_from_langchain(docs, filename), _loader_result_info(loader_info, opts))

        if loader_type == "csv":
            from langchain_community.document_loaders import CSVLoader

            docs = _with_tempfile(
                data,
                temp_extension,
                lambda path: CSVLoader(
                    file_path=path,
                    source_column=opts.source_column,
                    metadata_columns=opts.metadata_columns,
                    csv_args=_csv_args(opts),
                    encoding=opts.encoding,
                    autodetect_encoding=opts.autodetect_encoding,
                    content_columns=opts.content_columns,
                ),
            )
            return LoadedDocumentsResult(_from_langchain(docs, filename), _loader_result_info(loader_info, opts))

        if loader_type == "json":
            from langchain_community.document_loaders import JSONLoader

            try:
                docs = _with_tempfile(
                    data,
                    temp_extension,
                    lambda path: JSONLoader(
                        file_path=path,
                        jq_schema=_default_json_schema(data, opts),
                        content_key=opts.content_key,
                        is_content_key_jq_parsable=opts.is_content_key_jq_parsable,
                        text_content=_json_text_content(opts),
                        json_lines=opts.json_lines,
                    ),
                )
                return LoadedDocumentsResult(_from_langchain(docs, filename), _loader_result_info(loader_info, opts))
            except ImportError:
                return LoadedDocumentsResult(_fallback_json_documents(data, filename, opts), _loader_result_info(loader_info, opts))

        if loader_type == "pdf":
            from langchain_community.document_loaders import PyPDFLoader

            _validate_pdf_options(opts)
            docs = _with_tempfile(
                data,
                temp_extension,
                lambda path: PyPDFLoader(
                    file_path=path,
                    password=opts.password,
                    extract_images=opts.extract_images,
                    mode=opts.mode,
                    images_inner_format=opts.images_inner_format,
                    pages_delimiter=opts.pages_delimiter,
                    extraction_mode=opts.extraction_mode,
                    extraction_kwargs=opts.extraction_kwargs,
                ),
            )
            return LoadedDocumentsResult(_from_langchain(docs, filename), _loader_result_info(loader_info, opts))

        if loader_type == "docx":
            from langchain_community.document_loaders import Docx2txtLoader

            docs = _with_tempfile(data, temp_extension, lambda path: Docx2txtLoader(file_path=path))
            return LoadedDocumentsResult(_from_langchain(docs, filename), _loader_result_info(loader_info, opts))
    except DocumentProcessingError:
        raise
    except Exception as exc:
        ext = loader_info.get("declared_extension") or loader_info.get("detected_extension") or extension
        raise DocumentProcessingError(f"\u52a0\u8f7d {filename} \u5931\u8d25\uff08{ext}\uff09\uff1a{exc}") from exc

    raise DocumentProcessingError(f"\u6682\u4e0d\u652f\u6301\u8be5\u6587\u4ef6\u7c7b\u578b\uff1a{loader_type}")


def load_documents(
    data: bytes,
    filename: str,
    extension: str,
    options: dict[str, Any] | LoaderOptions | None = None,
) -> list[LoadedDocument]:
    return load_documents_result(data, filename, extension, options).documents



def _simple_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    size = max(chunk_size, 1)
    overlap = max(min(chunk_overlap, size - 1), 0)
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _metadata_rules(strategy: Any) -> dict[str, Any]:
    rules = _strategy_get(strategy, "metadata_rules", {}) or {}
    return dict(rules) if isinstance(rules, dict) else {}


def _bool_rule(rules: dict[str, Any], key: str, default: bool) -> bool:
    value = rules.get(key, default)
    return default if value is None else bool(value)


def _list_rule(rules: dict[str, Any], key: str, default: list[Any]) -> list[Any]:
    value = rules.get(key)
    if isinstance(value, list):
        return value
    return default


def _headers_rule(rules: dict[str, Any], default: list[tuple[str, str]]) -> list[tuple[str, str]]:
    headers = rules.get("headers_to_split_on", rules.get("headers"))
    if not isinstance(headers, list):
        return default
    normalized: list[tuple[str, str]] = []
    for item in headers:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append((str(item[0]), str(item[1])))
    return normalized or default


def _set_or_all(value: Any, default: Any) -> Any:
    if value == "all":
        return "all"
    if isinstance(value, list):
        return set(str(item) for item in value)
    return default


def _chunk_size(strategy: Any) -> int:
    return int(_strategy_get(strategy, "chunk_size", 800) or 800)


def _chunk_overlap(strategy: Any) -> int:
    size = _chunk_size(strategy)
    overlap = int(_strategy_get(strategy, "chunk_overlap", 0) or 0)
    return max(0, min(overlap, size - 1))


def _int_rule(rules: dict[str, Any], key: str, default: int) -> int:
    value = rules.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _text_splitter_kwargs(strategy: Any) -> dict[str, Any]:
    rules = _metadata_rules(strategy)
    return {
        "chunk_size": _chunk_size(strategy),
        "chunk_overlap": _chunk_overlap(strategy),
        "add_start_index": _bool_rule(rules, "add_start_index", False),
        "strip_whitespace": _bool_rule(rules, "strip_whitespace", True),
    }


def _tiktoken_kwargs(strategy: Any) -> dict[str, Any]:
    rules = _metadata_rules(strategy)
    kwargs = _text_splitter_kwargs(strategy)
    kwargs.update(
        {
            "encoding_name": rules.get("encoding_name", rules.get("token_encoding", "gpt2")),
            "model_name": rules.get("model_name") or rules.get("tiktoken_model_name"),
            "allowed_special": _set_or_all(rules.get("allowed_special"), set()),
            "disallowed_special": _set_or_all(rules.get("disallowed_special"), "all"),
        }
    )
    if not kwargs["model_name"]:
        kwargs.pop("model_name")
    return kwargs


def _tokenizer_enabled(strategy: Any) -> bool:
    rules = _metadata_rules(strategy)
    if "tokenizer_enabled" in rules:
        return bool(rules.get("tokenizer_enabled"))
    return bool(_strategy_get(strategy, "count_by_token", False))


def _tokenizer_type(strategy: Any) -> str:
    rules = _metadata_rules(strategy)
    value = str(rules.get("tokenizer_type") or "tiktoken").lower()
    aliases = {
        "gpt2": "gpt2_fast",
        "gpt2tokenizerfast": "gpt2_fast",
        "gpt2_tokenizer_fast": "gpt2_fast",
        "huggingface_gpt2": "gpt2_fast",
        "tik_token": "tiktoken",
    }
    value = aliases.get(value, value)
    if value in {"spacy", "nltk"}:
        raise DocumentProcessingError(f"{value} tokenizer 暂未启用；当前仅支持 tiktoken 和 GPT2TokenizerFast")
    if value not in {"tiktoken", "gpt2_fast"}:
        raise DocumentProcessingError(f"暂不支持 tokenizer_type：{value}")
    return value


def _gpt2_tokenizer(strategy: Any):
    rules = _metadata_rules(strategy)
    model_name = str(rules.get("hf_model_name") or "gpt2")
    try:
        from transformers import GPT2TokenizerFast
    except ImportError as exc:
        raise DocumentProcessingError("GPT2TokenizerFast 需要安装 transformers 依赖") from exc
    try:
        return GPT2TokenizerFast.from_pretrained(model_name)
    except Exception as exc:
        raise DocumentProcessingError(f"加载 GPT2TokenizerFast 失败（{model_name}）：{exc}") from exc


def _tokenized_text_splitter(splitter_cls: Any, strategy: Any, splitter_kwargs: dict[str, Any]):
    tokenizer_type = _tokenizer_type(strategy)
    if tokenizer_type == "tiktoken":
        kwargs = _tiktoken_kwargs(strategy)
        kwargs.update(splitter_kwargs)
        return splitter_cls.from_tiktoken_encoder(**kwargs)
    kwargs = _text_splitter_kwargs(strategy)
    kwargs.update(splitter_kwargs)
    return splitter_cls.from_huggingface_tokenizer(_gpt2_tokenizer(strategy), **kwargs)


class _HuggingFaceTokenTextSplitter:
    def __init__(self, strategy: Any) -> None:
        self.strategy = strategy
        self.tokenizer = _gpt2_tokenizer(strategy)

    def split_text(self, text: str) -> list[str]:
        from langchain_text_splitters.base import Tokenizer, split_text_on_tokens

        def encode(value: str) -> list[int]:
            return self.tokenizer.encode(value, add_special_tokens=False)

        tokenizer = Tokenizer(
            chunk_overlap=_chunk_overlap(self.strategy),
            tokens_per_chunk=_chunk_size(self.strategy),
            decode=self.tokenizer.decode,
            encode=encode,
        )
        return split_text_on_tokens(text=text, tokenizer=tokenizer)

    def create_documents(self, texts: list[str]) -> list[Document]:
        return [Document(page_content=chunk) for text in texts for chunk in self.split_text(text)]

    def split_documents(self, docs: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in docs:
            for content in self.split_text(doc.page_content):
                chunks.append(Document(page_content=content, metadata=dict(doc.metadata or {})))
        return chunks


def _recursive_splitter(strategy: Any):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    rules = _metadata_rules(strategy)
    splitter_kwargs = {
        "separators": _list_rule(rules, "separators", ["\n\n", "\n", "。", "！", "？", ";", "；", ",", "，", " ", ""]),
        "keep_separator": rules.get("keep_separator", "end"),
        "is_separator_regex": _bool_rule(rules, "is_separator_regex", False),
    }
    if _tokenizer_enabled(strategy):
        return _tokenized_text_splitter(RecursiveCharacterTextSplitter, strategy, splitter_kwargs)
    return RecursiveCharacterTextSplitter(
        **splitter_kwargs,
        **_text_splitter_kwargs(strategy),
    )


def _character_splitter(strategy: Any):
    from langchain_text_splitters import CharacterTextSplitter

    rules = _metadata_rules(strategy)
    splitter_kwargs = {
        "separator": str(rules.get("separator", "\n\n")),
        "is_separator_regex": _bool_rule(rules, "is_separator_regex", False),
    }
    if _tokenizer_enabled(strategy):
        return _tokenized_text_splitter(CharacterTextSplitter, strategy, splitter_kwargs)
    return CharacterTextSplitter(
        **splitter_kwargs,
        **_text_splitter_kwargs(strategy),
    )


def _token_splitter(strategy: Any):
    from langchain_text_splitters import TokenTextSplitter

    if _tokenizer_type(strategy) == "gpt2_fast":
        return _HuggingFaceTokenTextSplitter(strategy)
    return TokenTextSplitter(**_tiktoken_kwargs(strategy))


def _split_with_text_splitter(docs: list[Document], strategy: Any) -> list[Document]:
    mode = _split_mode(strategy, "")
    try:
        if mode == "character":
            splitter = _character_splitter(strategy)
        elif mode == "token":
            splitter = _token_splitter(strategy)
        else:
            splitter = _recursive_splitter(strategy)
        return splitter.split_documents(docs)
    except DocumentProcessingError:
        raise
    except Exception:
        chunks: list[Document] = []
        for doc in docs:
            for content in _simple_split(doc.page_content, _chunk_size(strategy), _chunk_overlap(strategy)):
                chunks.append(Document(page_content=content, metadata=dict(doc.metadata or {})))
        return chunks


def _create_with_text_splitter(text: str, strategy: Any) -> list[Document]:
    mode = _split_mode(strategy, "")
    if mode == "character":
        return _character_splitter(strategy).create_documents([text])
    if mode == "token":
        return _token_splitter(strategy).create_documents([text])
    return _recursive_splitter(strategy).create_documents([text])


def _recursive_after_split(strategy: Any) -> bool:
    return _bool_rule(_metadata_rules(strategy), "recursive_after_split", False)


def _maybe_recursive_after_split(docs: list[Document], strategy: Any) -> list[Document]:
    if _recursive_after_split(strategy):
        return _split_with_text_splitter(docs, strategy)
    return docs


def _split_structured_documents(docs: list[Document], strategy: Any, splitter) -> list[Document]:
    chunks: list[Document] = []
    for doc in docs:
        for item in splitter(doc.page_content, strategy):
            item.metadata = {**dict(doc.metadata or {}), **dict(item.metadata or {})}
            chunks.append(item)
    return _maybe_recursive_after_split(chunks, strategy)


def _split_markdown_header(docs: list[Document], strategy: Any) -> list[Document]:
    return _split_structured_documents(docs, strategy, _split_markdown_text)


def _split_markdown_text(text: str, strategy: Any) -> list[Document]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    rules = _metadata_rules(strategy)
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_headers_rule(rules, [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]),
        return_each_line=_bool_rule(rules, "return_each_line", False),
        strip_headers=bool(rules.get("strip_headers", _strategy_get(strategy, "heading_policy", "preserve_path") != "keep_inline")),
        custom_header_patterns=dict(rules.get("custom_header_patterns") or {}) or None,
    )
    return splitter.split_text(text)


def _split_html_header(docs: list[Document], strategy: Any) -> list[Document]:
    return _split_structured_documents(docs, strategy, _split_html_text)


def _split_html_text(text: str, strategy: Any) -> list[Document]:
    from langchain_text_splitters import HTMLHeaderTextSplitter, HTMLSectionSplitter, HTMLSemanticPreservingSplitter

    rules = _metadata_rules(strategy)
    headers = _headers_rule(rules, [("h1", "h1"), ("h2", "h2"), ("h3", "h3"), ("h4", "h4")])
    mode = _split_mode(strategy, "")
    if mode == "html_section":
        splitter = HTMLSectionSplitter(headers_to_split_on=headers)
    elif mode == "html_semantic_preserving":
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=headers,
            max_chunk_size=_int_rule(rules, "max_chunk_size", 1000),
            chunk_overlap=_int_rule(rules, "chunk_overlap", 0),
            separators=rules.get("separators"),
            elements_to_preserve=rules.get("elements_to_preserve"),
            preserve_links=_bool_rule(rules, "preserve_links", False),
            preserve_images=_bool_rule(rules, "preserve_images", False),
            preserve_videos=_bool_rule(rules, "preserve_videos", False),
            preserve_audio=_bool_rule(rules, "preserve_audio", False),
            stopword_removal=_bool_rule(rules, "stopword_removal", False),
            stopword_lang=str(rules.get("stopword_lang", "english") or "english"),
            normalize_text=_bool_rule(rules, "normalize_text", False),
            external_metadata=dict(rules.get("external_metadata") or {}) or None,
            allowlist_tags=rules.get("allowlist_tags"),
            denylist_tags=rules.get("denylist_tags"),
            preserve_parent_metadata=_bool_rule(rules, "preserve_parent_metadata", False),
            keep_separator=rules.get("keep_separator", True),
        )
    else:
        splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=headers,
            return_each_element=_bool_rule(rules, "return_each_element", False),
        )
    return splitter.split_text(text)


def _split_json(docs: list[Document], strategy: Any) -> list[Document]:
    return _split_structured_documents(docs, strategy, _split_json_text)


def _split_json_text(text: str, strategy: Any) -> list[Document]:
    from langchain_text_splitters import RecursiveJsonSplitter

    rules = _metadata_rules(strategy)
    splitter = RecursiveJsonSplitter(
        max_chunk_size=_int_rule(rules, "max_chunk_size", 2000),
        min_chunk_size=rules.get("min_chunk_size"),
    )
    try:
        parsed = json.loads(text)
        parts = splitter.split_text(
            parsed,
            convert_lists=_bool_rule(rules, "convert_lists", True),
            ensure_ascii=_bool_rule(rules, "ensure_ascii", False),
        )
    except Exception:
        parts = _simple_split(text, _int_rule(rules, "max_chunk_size", 2000), 0)
    return [Document(page_content=content, metadata={"json_chunk_index": index}) for index, content in enumerate(parts)]


def _language_for_code(extension: str, strategy: Any):
    from langchain_text_splitters import Language

    rules = _metadata_rules(strategy)
    requested = rules.get("language")
    by_ext = {
        "py": "python",
        "js": "js",
        "jsx": "js",
        "ts": "ts",
        "tsx": "ts",
        "java": "java",
        "go": "go",
        "rs": "rust",
        "rb": "ruby",
        "php": "php",
        "cpp": "cpp",
        "c": "cpp",
        "cs": "csharp",
        "html": "html",
        "md": "markdown",
    }
    value = requested or by_ext.get(extension, "python")
    for item in Language:
        if item.value == value:
            return item
    return Language.PYTHON


def _split_code(docs: list[Document], strategy: Any, extension: str) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=_language_for_code(extension, strategy),
        chunk_size=_chunk_size(strategy),
        chunk_overlap=_chunk_overlap(strategy),
    )
    return splitter.split_documents(docs)


def _split_code_text(text: str, strategy: Any, extension: str) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=_language_for_code(extension, strategy),
        chunk_size=_chunk_size(strategy),
        chunk_overlap=_chunk_overlap(strategy),
    )
    return splitter.create_documents([text])


def _split_mode(strategy: Any, extension: str) -> str:
    splitter_type = str(_strategy_get(strategy, "splitter_type", "recursive") or "recursive")
    aliases = {
        "markdown": "markdown_header",
        "html": "html_header",
        "html_semantic": "html_semantic_preserving",
    }
    splitter_type = aliases.get(splitter_type, splitter_type)
    if splitter_type == "markdown":
        return "markdown_header"
    if splitter_type == "structured":
        if extension == "md":
            return "markdown_header"
        if extension in {"html", "htm"}:
            return "html_header"
        if extension == "json":
            return "json"
        return "recursive"
    if splitter_type == "token":
        return "token"
    if splitter_type in {
        "character",
        "code",
        "html_header",
        "html_section",
        "html_semantic_preserving",
        "json",
        "markdown_header",
        "recursive",
    }:
        return splitter_type
    return "recursive"


def _title_path(metadata: dict[str, Any]) -> str | None:
    titles = [str(metadata[key]) for key in ("h1", "h2", "h3", "h4", "title1", "title2", "title3") if metadata.get(key)]
    return " / ".join(titles) if titles else metadata.get("title_path")


def _clean_metadata(metadata: dict[str, Any], strategy: Any) -> dict[str, Any]:
    rules = _metadata_rules(strategy)
    cleaned = dict(metadata)
    include = rules.get("include_keys")
    exclude = set(rules.get("exclude_keys") or [])
    if include:
        cleaned = {key: cleaned[key] for key in include if key in cleaned}
    for key in exclude:
        cleaned.pop(key, None)
    cleaned.update(rules.get("extra") or {})
    if not _strategy_get(strategy, "keep_page_number", True):
        cleaned.pop("page", None)
        cleaned.pop("page_number", None)
    if not _strategy_get(strategy, "keep_title_path", True):
        for key in ("h1", "h2", "h3", "h4", "title1", "title2", "title3", "title_path"):
            cleaned.pop(key, None)
    return cleaned


def _format_chunks(chunk_docs: list[Document], strategy: Any, limit: int | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for doc in chunk_docs:
        content = doc.page_content or ""
        metadata = _clean_metadata(dict(doc.metadata or {}), strategy)
        page_number = metadata.get("page_number") or metadata.get("page")
        result.append(
            {
                "chunk_index": len(result),
                "content": content,
                "char_count": len(content),
                "token_count": _estimate_tokens(content),
                "title_path": _title_path(metadata) if _strategy_get(strategy, "keep_title_path", True) else None,
                "page_number": page_number if _strategy_get(strategy, "keep_page_number", True) else None,
                "metadata": metadata,
            }
        )
        if limit and len(result) >= limit:
            break
    return result


def split_texts(
    texts: list[str],
    strategy: Any,
    extension: str,
    metadatas: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    ext = extension.lower().lstrip(".")
    mode = _split_mode(strategy, ext)
    chunk_docs: list[Document] = []
    try:
        for index, text in enumerate(texts):
            source_metadata = dict((metadatas or [{}])[index] if metadatas and index < len(metadatas) else {})
            if mode in {"recursive", "character", "token"}:
                docs = _create_with_text_splitter(text, strategy)
            elif mode == "markdown_header":
                docs = _maybe_recursive_after_split(_split_markdown_text(text, strategy), strategy)
            elif mode in {"html_header", "html_section", "html_semantic_preserving"}:
                docs = _maybe_recursive_after_split(_split_html_text(text, strategy), strategy)
            elif mode == "json":
                docs = _maybe_recursive_after_split(_split_json_text(text, strategy), strategy)
            elif mode == "code":
                docs = _split_code_text(text, strategy, ext)
            else:
                docs = _create_with_text_splitter(text, strategy)
            for doc in docs:
                doc.metadata = {**source_metadata, **dict(doc.metadata or {})}
                chunk_docs.append(doc)
                if limit and len(chunk_docs) >= limit:
                    return _format_chunks(chunk_docs, strategy, limit=limit)
    except Exception as exc:
        raise DocumentProcessingError(f"分块预览失败（{mode}）：{exc}") from exc
    return _format_chunks(chunk_docs, strategy, limit=limit)


def split_documents(
    documents: list[LoadedDocument],
    strategy: Any,
    extension: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    ext = extension.lower().lstrip(".")
    langchain_docs = [Document(page_content=doc.page_content, metadata=dict(doc.metadata or {})) for doc in documents]
    mode = _split_mode(strategy, ext)

    try:
        if mode == "token":
            chunk_docs = _token_splitter(strategy).split_documents(langchain_docs)
        elif mode == "markdown_header":
            chunk_docs = _split_markdown_header(langchain_docs, strategy)
        elif mode == "html_header":
            chunk_docs = _split_html_header(langchain_docs, strategy)
        elif mode == "json":
            chunk_docs = _split_json(langchain_docs, strategy)
        elif mode == "code":
            chunk_docs = _split_code(langchain_docs, strategy, ext)
        elif mode in {"html_section", "html_semantic_preserving"}:
            chunk_docs = _split_html_header(langchain_docs, strategy)
        elif mode == "character":
            chunk_docs = _split_with_text_splitter(langchain_docs, strategy)
        else:
            chunk_docs = _split_with_text_splitter(langchain_docs, strategy)
    except Exception as exc:
        raise DocumentProcessingError(f"分块失败（{mode}）：{exc}") from exc

    return _format_chunks(chunk_docs, strategy, limit=limit)
