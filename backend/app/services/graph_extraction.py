# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import get_settings
from app.models import ModelConfig
from app.services.llm import DEFAULT_GRAPH_EXTRACTION_PROMPT, complete_text


PREDICATE_WHITELIST = {
    "定义",
    "属于",
    "包含",
    "组成",
    "依赖",
    "导致",
    "影响",
    "用于",
    "实现",
    "位于",
    "发生于",
    "负责",
    "等价于",
    "别名",
    "引用",
    "约束",
}
PREDICATE_ALIASES = {
    "用途": "用于",
    "用来": "用于",
    "使用": "用于",
    "类型": "属于",
    "类别": "属于",
    "技术类型": "属于",
    "等同于": "等价于",
    "又称": "别名",
    "相关": "引用",
    "related_to": "引用",
}
DEFAULT_ENTITY_TYPE = "概念"
MIN_CONFIDENCE = 0.65
CROSS_CHUNK_MIN_CONFIDENCE = 0.8
MAX_LOCAL_RELATIONS_PER_CHUNK = 24
MAX_CROSS_CHUNK_PAIRS = 20
MAX_GRAPH_EXTRACTION_CONCURRENCY = 8

DEFAULT_CROSS_CHUNK_PROMPT = """
你是知识图谱事实抽取器。请只基于下面两个相邻或相关片段抽取跨片段才能确认的事实关系。
返回纯 JSON，不要 Markdown：
{"entities":[{"name":"...","type":"...","aliases":[],"evidence":"...","confidence":0.0}],"relations":[{"subject":"...","predicate":"...","object":"...","evidence":"...","confidence":0.0}]}

规则：
- 只抽原文明确表达的事实，不要使用常识补全。
- evidence 必须是下方片段中的原文连续片段。
- predicate 只能从这些词中选择：定义、属于、包含、组成、依赖、导致、影响、用于、实现、位于、发生于、负责、等价于、别名、引用、约束。
- 无法确定时返回 {"entities":[],"relations":[]}。

片段：
{text}
""".strip()


class GraphExtractionError(RuntimeError):
    pass


@dataclass
class GraphExtractionResult:
    entities_by_chunk: dict[str, list[dict[str, Any]]]
    relations_by_chunk: dict[str, list[dict[str, Any]]]
    entity_count: int
    relation_count: int
    cross_chunk_relation_count: int
    rejected_count: int
    rejection_reasons: dict[str, int]

    def summary(self) -> dict[str, Any]:
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "cross_chunk_relation_count": self.cross_chunk_relation_count,
            "rejected_count": self.rejected_count,
            "rejection_reasons": dict(self.rejection_reasons),
        }


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
            return json.loads(cleaned[start: end + 1])
        raise


def parse_graph_response(text: str) -> dict[str, list[dict[str, Any]]]:
    data = _json_object(text)
    if not isinstance(data, dict):
        return {"entities": [], "relations": []}

    entities = data.get("entities") if isinstance(data.get("entities"), list) else []
    relations = data.get("relations") if isinstance(data.get("relations"), list) else []
    triples = data.get("triples") if isinstance(data.get("triples"), list) else []
    for item in triples:
        if not isinstance(item, dict):
            continue
        relations.append(
            {
                "subject": item.get("subject"),
                "predicate": item.get("predicate"),
                "object": item.get("object"),
                "evidence": item.get("evidence"),
                "confidence": item.get("confidence"),
            }
        )
    return {
        "entities": [item for item in entities if isinstance(item, dict)],
        "relations": [item for item in relations if isinstance(item, dict)],
    }


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _canonical_key(value: Any) -> str:
    return _norm(value).casefold()


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _contains_evidence(source: str, evidence: str) -> bool:
    if not evidence:
        return False
    return evidence in source or _compact(evidence) in _compact(source)


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence < 0:
        return 0.0
    if confidence > 1:
        return 1.0
    return confidence


def _predicate(value: Any) -> str:
    predicate = _norm(value)
    predicate = PREDICATE_ALIASES.get(predicate, predicate)
    return predicate if predicate in PREDICATE_WHITELIST else ""


def _entity_type(value: Any) -> str:
    entity_type = _norm(value)
    return entity_type[:64] if entity_type else DEFAULT_ENTITY_TYPE


def _reject(reasons: Counter[str], reason: str) -> None:
    reasons[reason] += 1


def _clean_entity(item: dict[str, Any], source_text: str, reasons: Counter[str]) -> dict[str, Any] | None:
    name = _norm(item.get("name"))
    if not name:
        _reject(reasons, "entity_missing_name")
        return None
    evidence = _norm(item.get("evidence"))
    if not _contains_evidence(source_text, evidence):
        _reject(reasons, "entity_evidence_missing")
        return None
    confidence = _confidence(item.get("confidence"))
    if confidence < MIN_CONFIDENCE:
        _reject(reasons, "entity_low_confidence")
        return None
    aliases = []
    for alias in item.get("aliases") or []:
        alias_text = _norm(alias)
        if alias_text and alias_text != name:
            aliases.append(alias_text[:256])
    return {
        "name": name[:256],
        "type": _entity_type(item.get("type")),
        "aliases": list(dict.fromkeys(aliases)),
        "evidence": evidence[:1000],
        "confidence": confidence,
    }


def _clean_relation(
    item: dict[str, Any],
    source_text: str,
    source_chunk_ids: list[str],
    reasons: Counter[str],
    *,
    cross_chunk: bool,
) -> dict[str, Any] | None:
    subject = _norm(item.get("subject"))
    obj = _norm(item.get("object"))
    predicate = _predicate(item.get("predicate"))
    if not subject or not obj:
        _reject(reasons, "relation_missing_endpoint")
        return None
    if not predicate:
        _reject(reasons, "relation_predicate_not_allowed")
        return None
    evidence = _norm(item.get("evidence"))
    if not _contains_evidence(source_text, evidence):
        _reject(reasons, "relation_evidence_missing")
        return None
    confidence = _confidence(item.get("confidence"))
    threshold = CROSS_CHUNK_MIN_CONFIDENCE if cross_chunk else MIN_CONFIDENCE
    if confidence < threshold:
        _reject(reasons, "relation_low_confidence")
        return None
    return {
        "subject": subject[:256],
        "predicate": predicate,
        "predicate_key": predicate,
        "object": obj[:256],
        "evidence": evidence[:1000],
        "confidence": confidence,
        "chunk_id": source_chunk_ids[0],
        "source_chunk_ids": list(dict.fromkeys(source_chunk_ids)),
        "cross_chunk": cross_chunk,
    }


def _row_source_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    title_path = row.get("title_path") or metadata.get("title_path")
    parts = []
    if title_path:
        parts.append(f"标题路径：{title_path}")
    parts.append(str(row.get("content") or ""))
    return "\n".join(parts)


def _chunk_prompt_text(rows: list[dict[str, Any]], index: int) -> str:
    row = rows[index]
    parts = [_row_source_text(row)]
    if index > 0:
        previous = str(rows[index - 1].get("content") or "")
        parts.append(f"上文片段摘要：{previous[-800:]}")
    if index + 1 < len(rows):
        following = str(rows[index + 1].get("content") or "")
        parts.append(f"下文片段摘要：{following[:800]}")
    return "\n\n".join(parts)


def _render_prompt(template: str, text: str) -> str:
    limited = text[:12000]
    if "{text}" in template:
        return template.replace("{text}", limited)
    return f"{template.rstrip()}\n\n{limited}"


def _merge_entity(
    target: dict[tuple[str, str], dict[str, Any]],
    entity: dict[str, Any],
) -> dict[str, Any]:
    key = (entity["type"], entity["canonical_name"])
    existing = target.get(key)
    if existing is None:
        target[key] = dict(entity)
        return target[key]
    aliases = list(dict.fromkeys([*(existing.get("aliases") or []), *(entity.get("aliases") or [])]))
    existing["aliases"] = aliases[:50]
    if entity.get("confidence", 0) > existing.get("confidence", 0):
        existing["name"] = entity["name"]
        existing["evidence"] = entity.get("evidence", "")
        existing["confidence"] = entity.get("confidence", 0.0)
    return existing


def build_graph_records(
    rows: list[dict[str, Any]],
    extracted_by_chunk: dict[str, dict[str, Any]],
    cross_extracted: list[dict[str, Any]] | None = None,
) -> GraphExtractionResult:
    reasons: Counter[str] = Counter()
    raw_entities_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_relations: list[dict[str, Any]] = []

    row_by_id = {row["chunk_id"]: row for row in rows}
    for row in rows:
        chunk_id = row["chunk_id"]
        extracted = extracted_by_chunk.get(chunk_id) or {"entities": [], "relations": []}
        source_text = str(extracted.get("_source_text") or _row_source_text(row))
        for item in extracted.get("entities") or []:
            entity = _clean_entity(item, source_text, reasons)
            if entity:
                raw_entities_by_chunk[chunk_id].append(entity)
        for item in (extracted.get("relations") or [])[:MAX_LOCAL_RELATIONS_PER_CHUNK]:
            relation = _clean_relation(item, source_text, [chunk_id], reasons, cross_chunk=False)
            if relation:
                raw_relations.append(relation)

    for extracted in cross_extracted or []:
        source_chunk_ids = [chunk_id for chunk_id in extracted.get("source_chunk_ids") or [] if chunk_id in row_by_id]
        if len(source_chunk_ids) < 2:
            continue
        source_text = str(extracted.get("_source_text") or "")
        for item in extracted.get("relations") or []:
            relation = _clean_relation(item, source_text, source_chunk_ids, reasons, cross_chunk=True)
            if relation:
                raw_relations.append(relation)

    hints_by_key: dict[str, dict[str, Any]] = {}
    alias_to_key: dict[str, str] = {}
    for entities in raw_entities_by_chunk.values():
        for entity in entities:
            key = _canonical_key(entity["name"])
            hints_by_key.setdefault(key, entity)
            for alias in entity.get("aliases") or []:
                alias_key = _canonical_key(alias)
                if alias_key:
                    alias_to_key.setdefault(alias_key, key)

    def resolve_entity(name: str) -> dict[str, Any]:
        raw_key = _canonical_key(name)
        canonical_key = alias_to_key.get(raw_key, raw_key)
        hint = hints_by_key.get(canonical_key) or hints_by_key.get(raw_key)
        entity_type = _entity_type((hint or {}).get("type"))
        display_name = (hint or {}).get("name") or _norm(name)[:256]
        aliases = list((hint or {}).get("aliases") or [])
        if raw_key != canonical_key and _norm(name) not in aliases:
            aliases.append(_norm(name))
        return {
            "name": display_name,
            "canonical_name": canonical_key,
            "type": entity_type,
            "aliases": list(dict.fromkeys(aliases)),
        }

    entities_by_chunk: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    all_entities: dict[tuple[str, str], dict[str, Any]] = {}

    for chunk_id, entities in raw_entities_by_chunk.items():
        for entity in entities:
            resolved = resolve_entity(entity["name"])
            record = {
                **resolved,
                "evidence": entity.get("evidence", ""),
                "confidence": entity.get("confidence", 0.0),
            }
            _merge_entity(entities_by_chunk[chunk_id], record)
            _merge_entity(all_entities, record)

    relations_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relation_seen: set[tuple[str, str, str, str, str]] = set()
    for relation in raw_relations:
        subject = resolve_entity(relation["subject"])
        obj = resolve_entity(relation["object"])
        chunk_id = relation["chunk_id"]
        for endpoint in (subject, obj):
            entity_record = {
                **endpoint,
                "evidence": relation["evidence"],
                "confidence": relation["confidence"],
            }
            _merge_entity(entities_by_chunk[chunk_id], entity_record)
            _merge_entity(all_entities, entity_record)
        relation_record = {
            **relation,
            "subject_name": subject["name"],
            "subject_canonical_name": subject["canonical_name"],
            "subject_type": subject["type"],
            "object_name": obj["name"],
            "object_canonical_name": obj["canonical_name"],
            "object_type": obj["type"],
        }
        key = (
            relation_record["subject_type"],
            relation_record["subject_canonical_name"],
            relation_record["predicate_key"],
            relation_record["object_type"],
            relation_record["object_canonical_name"],
        )
        scoped_key = (*key, chunk_id)
        if scoped_key in relation_seen:
            continue
        relation_seen.add(scoped_key)
        relations_by_chunk[chunk_id].append(relation_record)

    return GraphExtractionResult(
        entities_by_chunk={chunk_id: list(items.values()) for chunk_id, items in entities_by_chunk.items()},
        relations_by_chunk=dict(relations_by_chunk),
        entity_count=len(all_entities),
        relation_count=sum(len(items) for items in relations_by_chunk.values()),
        cross_chunk_relation_count=sum(1 for items in relations_by_chunk.values() for item in items if item.get("cross_chunk")),
        rejected_count=sum(reasons.values()),
        rejection_reasons=dict(reasons),
    )


def _cross_candidate_pairs(
    rows: list[dict[str, Any]],
    extracted_by_chunk: dict[str, dict[str, Any]],
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(left: int, right: int) -> None:
        if left == right:
            return
        pair = (min(left, right), max(left, right))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    for index in range(len(rows) - 1):
        add(index, index + 1)

    title_groups: dict[str, list[int]] = defaultdict(list)
    entity_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        title_path = _norm(row.get("title_path") or metadata.get("title_path"))
        if title_path:
            title_groups[title_path].append(index)
        chunk_id = row["chunk_id"]
        for entity in extracted_by_chunk.get(chunk_id, {}).get("entities") or []:
            key = _canonical_key(entity.get("name"))
            if key:
                entity_groups[key].append(index)

    for group in title_groups.values():
        for left, right in zip(group, group[1:]):
            add(left, right)
    for group in entity_groups.values():
        unique = list(dict.fromkeys(group))
        for left, right in zip(unique, unique[1:]):
            add(left, right)

    return pairs[:MAX_CROSS_CHUNK_PAIRS]


def _graph_extraction_concurrency() -> int:
    configured = get_settings().GRAPH_EXTRACTION_CONCURRENCY
    return max(1, min(MAX_GRAPH_EXTRACTION_CONCURRENCY, int(configured or 1)))


def extract_graph(model: ModelConfig, rows: list[dict[str, Any]]) -> GraphExtractionResult:
    if not rows:
        return GraphExtractionResult({}, {}, 0, 0, 0, 0, {})
    if not model.base_url or model.provider == "mock":
        raise GraphExtractionError("Neo4j graph extraction requires a real enabled LLM model")

    template = model.prompt_template or DEFAULT_GRAPH_EXTRACTION_PROMPT
    extracted_by_chunk: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    concurrency = _graph_extraction_concurrency()

    def extract_chunk(index: int, client: httpx.Client) -> tuple[str, dict[str, Any], str | None]:
        row = rows[index]
        prompt_text = _chunk_prompt_text(rows, index)
        evidence_text = _row_source_text(row)
        prompt = _render_prompt(template, prompt_text)
        try:
            parsed = parse_graph_response(complete_text(model, prompt, client=client))
        except Exception:
            parsed = {"entities": [], "relations": []}
            return row["chunk_id"], parsed | {"_source_text": evidence_text}, "invalid_json"
        return row["chunk_id"], parsed | {"_source_text": evidence_text}, None

    def extract_cross(left: int, right: int, client: httpx.Client) -> tuple[dict[str, Any] | None, str | None]:
        source_text = "\n\n".join([_row_source_text(rows[left]), _row_source_text(rows[right])])
        try:
            parsed = parse_graph_response(
                complete_text(model, _render_prompt(DEFAULT_CROSS_CHUNK_PROMPT, source_text), client=client)
            )
        except Exception:
            return None, "invalid_cross_json"
        parsed["_source_text"] = source_text
        parsed["source_chunk_ids"] = [rows[left]["chunk_id"], rows[right]["chunk_id"]]
        return parsed, None

    with httpx.Client(timeout=model.timeout or 60) as client:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_index = {executor.submit(extract_chunk, index, client): index for index in range(len(rows))}
            chunk_results: list[tuple[int, str, dict[str, Any], str | None]] = []
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                chunk_id, parsed, reason = future.result()
                chunk_results.append((index, chunk_id, parsed, reason))
            for _index, chunk_id, parsed, reason in sorted(chunk_results, key=lambda item: item[0]):
                if reason:
                    rejected[reason] += 1
                extracted_by_chunk[chunk_id] = parsed

        cross_extracted: list[dict[str, Any]] = []
        cross_pairs = _cross_candidate_pairs(rows, extracted_by_chunk)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_order = {
                executor.submit(extract_cross, left, right, client): order
                for order, (left, right) in enumerate(cross_pairs)
            }
            cross_results: list[tuple[int, dict[str, Any] | None, str | None]] = []
            for future in as_completed(future_to_order):
                order = future_to_order[future]
                parsed, reason = future.result()
                cross_results.append((order, parsed, reason))
            for _order, parsed, reason in sorted(cross_results, key=lambda item: item[0]):
                if reason:
                    rejected[reason] += 1
                    continue
                if parsed:
                    cross_extracted.append(parsed)

    result = build_graph_records(rows, extracted_by_chunk, cross_extracted)
    merged_reasons = Counter(result.rejection_reasons)
    merged_reasons.update(rejected)
    result.rejection_reasons = dict(merged_reasons)
    result.rejected_count = sum(merged_reasons.values())
    return result
