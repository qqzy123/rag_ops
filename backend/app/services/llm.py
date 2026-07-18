# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.core.security import decrypt_secret
from app.models import ModelConfig


DEFAULT_GRAPH_EXTRACTION_PROMPT = """
你是知识图谱事实抽取器。请从片段中抽取实体和事实关系。
返回纯 JSON，不要 Markdown：
{"entities":[{"name":"...","type":"...","aliases":[],"evidence":"...","confidence":0.0}],"relations":[{"subject":"...","predicate":"...","object":"...","evidence":"...","confidence":0.0}]}

规则：
- 只抽原文明确表达的事实，不要使用常识补全。
- evidence 必须是片段中的原文连续片段。
- 实体名使用规范短名；同义名称放入 aliases。
- predicate 只能从这些词中选择：定义、属于、包含、组成、依赖、导致、影响、用于、实现、位于、发生于、负责、等价于、别名、引用、约束。
- 无法确定时返回 {"entities":[],"relations":[]}。

片段：
{text}
""".strip()


def _chat_endpoint(model: ModelConfig) -> str:
    endpoint = (model.base_url or "").rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    return endpoint


def _complete_text_with_client(model: ModelConfig, prompt: str, client: httpx.Client) -> str:
    headers = {"Content-Type": "application/json"}
    api_key = decrypt_secret(model.api_key_encrypted)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    response = client.post(_chat_endpoint(model), headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def complete_text(model: ModelConfig, prompt: str, client: httpx.Client | None = None) -> str:
    if not model.base_url or model.provider == "mock":
        return '{"entities":[],"relations":[]}'
    if client is not None:
        return _complete_text_with_client(model, prompt, client)
    with httpx.Client(timeout=model.timeout or 60) as client:
        return _complete_text_with_client(model, prompt, client)


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def extract_triples(model: ModelConfig, text: str) -> list[dict[str, Any]]:
    if not text.strip():
        return []
    if not model.base_url or model.provider == "mock":
        return []
    template = model.prompt_template or DEFAULT_GRAPH_EXTRACTION_PROMPT
    prompt = template.replace("{text}", text[:12000])
    data = _json_object(complete_text(model, prompt))
    triples = data.get("triples") if isinstance(data, dict) else []
    relations = data.get("relations") if isinstance(data, dict) else []
    triples = [*(triples or []), *(relations or [])]
    result: list[dict[str, Any]] = []
    for item in triples or []:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue
        result.append(
            {
                "subject": subject[:256],
                "predicate": predicate[:128],
                "object": obj[:256],
                "evidence": str(item.get("evidence") or "")[:1000],
                "confidence": float(item.get("confidence") or 0),
            }
        )
    return result[:12]


def test_llm_connectivity(model: ModelConfig) -> dict[str, Any]:
    if not model.base_url or model.provider == "mock":
        return {"ok": True, "mode": "mock"}
    content = complete_text(model, 'Return only this JSON: {"ok": true}')
    return {"ok": True, "sample": content[:120]}
