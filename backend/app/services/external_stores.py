# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any

import unicodedata

from app.core.config import get_settings


def _graph_query_terms(query_text: str, limit: int = 20) -> list[str]:
    normalized = unicodedata.normalize("NFKC", query_text or "").strip().lower()
    candidates: list[str] = []
    if len(normalized) >= 2:
        candidates.append(normalized)
    candidates.extend(re.findall(r"[a-z0-9][a-z0-9_+\-.#]{1,}", normalized))
    candidates.extend(re.findall(r"[\u4e00-\u9fff]{2,}", normalized))

    terms: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        term = term.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


class ElasticsearchStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None

    def client(self):
        if self._client is None:
            from elasticsearch import Elasticsearch
            self._client = Elasticsearch(self.settings.ELASTICSEARCH_URL)
        return self._client

    def ensure_index(self) -> None:
        client = self.client()
        index = self.settings.ELASTICSEARCH_INDEX
        if client.indices.exists(index=index):
            return
        client.indices.create(
            index=index,
            mappings={
                "properties": {
                    "user_id": {"type": "keyword"},
                    "knowledge_base_id": {"type": "keyword"},
                    "file_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word"
                    },
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        )

    # 批量写入 ES
    def index_chunks(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        from elasticsearch.helpers import bulk

        self.ensure_index()
        actions = [
            {
                "_index": self.settings.ELASTICSEARCH_INDEX,
                "_id": row["chunk_id"],  # `chunk_id` 作为唯一 `_id`
                "_source": row,
            }
            for row in rows
        ]
        bulk(self.client(), actions)

    # 按范围批量删除 ES 文档
    def delete_by_scope(
            self,
            user_id: str,
            knowledge_base_id: str,
            file_id: str | None = None,
    ) -> None:
        client = self.client()
        index = self.settings.ELASTICSEARCH_INDEX
        if not client.indices.exists(index=index):
            return
        if not file_id or not user_id or not knowledge_base_id:
            raise ValueError("删除异常")
        filters: list[dict[str, Any]] = [
            {"term": {"user_id": user_id}},
            {"term": {"knowledge_base_id": knowledge_base_id}},
            {"term": {"file_id": file_id}}
        ]
        query = {"query": {"bool": {"filter": filters}}}
        print(json.dumps(query))
        self.client().delete_by_query(
            index=self.settings.ELASTICSEARCH_INDEX,
            body=query,
            conflicts="proceed",  # 版本冲突时继续执行删除，不中断
            refresh=False,  # 删除后不立刻刷新索引，查询不实时生效，提高性能
        )

    # bm25 检索
    def bm25_search(
            self,
            user_id: str,
            knowledge_base_id: str,
            query: str,
            top_k: int
    ) -> list[dict[str, Any]]:
        try:
            body = {
                "query": {
                    "bool": {
                        "must": [{"match": {"content": query}}],
                        "filter": [
                            {"term": {"user_id": user_id}},
                            {"term": {"knowledge_base_id": knowledge_base_id}},
                        ],
                    }
                },
                "size": top_k,
            }
            data = self.client().search(index=self.settings.ELASTICSEARCH_INDEX, body=body)
        except Exception as e:
            raise RuntimeError("ES 检索异常") from e
        results = []
        for hit in data.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append(
                {
                    "chunk_id": source.get("chunk_id"),
                    "file_id": source.get("file_id"),
                    "chunk_index": source.get("chunk_index"),
                    "content": source.get("content", ""),
                    "metadata": source.get("metadata") or {},
                    "bm25_score": float(hit.get("_score") or 0.0),
                    "recall_types": ["bm25"],
                }
            )
        return results

    # 预览数据
    def preview_file(self, user_id: str, knowledge_base_id: str, file_id: str) -> list[dict[str, Any]]:
        try:
            body = {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"user_id": user_id}},
                            {"term": {"knowledge_base_id": knowledge_base_id}},
                            {"term": {"file_id": file_id}},
                        ]
                    }
                },
                "size": 100,
                "sort": [{"chunk_index": "asc"}],
            }
            data = self.client().search(index=self.settings.ELASTICSEARCH_INDEX, body=body)
        except Exception as e:
            raise RuntimeError("ES 检索异常") from e
        return [hit.get("_source", {}) for hit in data.get("hits", {}).get("hits", [])]


class MilvusStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None

    def collection_name(self, dimension: int) -> str:
        return f"{self.settings.MILVUS_COLLECTION_PREFIX}_{dimension}"

    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=f"http://{self.settings.MILVUS_HOST}:{self.settings.MILVUS_PORT}")
        return self._client

    @staticmethod
    def _expr_eq(field_name: str, value: str) -> str:
        return f"{field_name} == {json.dumps(value)}"

    def _scope_filter(self, user_id: str, knowledge_base_id: str, file_id: str | None = None) -> str:
        if not user_id or not knowledge_base_id:
            raise ValueError("Invalid Milvus scope")
        filters = [
            self._expr_eq("user_id", user_id),
            self._expr_eq("knowledge_base_id", knowledge_base_id),
        ]
        if file_id is not None:
            if not file_id:
                raise ValueError("Invalid Milvus scope")
            filters.append(self._expr_eq("file_id", file_id))
        return " and ".join(filters)

    def _collection_names(self) -> list[str]:
        return [
            name
            for name in self.client().list_collections()
            if name.startswith(self.settings.MILVUS_COLLECTION_PREFIX)
        ]

    def _load_collection(self, name: str) -> None:
        client = self.client()
        try:
            state = client.get_load_state(collection_name=name).get("state")
            if "Loaded" in str(state):
                return
        except Exception:
            pass
        client.load_collection(collection_name=name)

    def ensure_collection(self, dimension: int) -> str:
        from pymilvus import DataType, MilvusClient

        client = self.client()
        name = self.collection_name(dimension)
        if client.has_collection(collection_name=name):
            self._load_collection(name)
            return name

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="knowledge_base_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="file_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        client.create_collection(collection_name=name, schema=schema, index_params=index_params)
        self._load_collection(name)
        return name

    def insert_chunks(self, dimension: int, rows: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        if not rows:
            return
        if len(rows) != len(vectors):
            raise ValueError(f"Vector count does not match chunk count: {len(vectors)} / {len(rows)}")
        name = self.ensure_collection(dimension)
        data = [
            {
                "pk": row["chunk_id"],
                "user_id": row["user_id"],
                "knowledge_base_id": row["knowledge_base_id"],
                "file_id": row["file_id"],
                "chunk_id": row["chunk_id"],
                "metadata": row.get("metadata") or {},
                "vector": vector,
            }
            # 提前校验行数和向量数量完全匹配，保障**每一条原始 chunk 都精确对应一条 embedding 向量**
            for row, vector in zip(rows, vectors, strict=True)
        ]
        client = self.client()
        client.insert(collection_name=name, data=data)
        client.flush(collection_name=name)

    def delete_by_scope(
            self,
            user_id: str,
            knowledge_base_id: str,
            file_id: str | None = None,
    ) -> None:
        client = self.client()
        expr = self._scope_filter(user_id, knowledge_base_id, file_id)
        for name in self._collection_names():
            self._load_collection(name)
            client.delete(collection_name=name, filter=expr)
            client.flush(collection_name=name)

    def vector_search(
            self,
            user_id: str,
            knowledge_base_id: str,
            vector: list[float],
            top_k: int,
            dimension: int,
    ) -> list[dict[str, Any]]:
        try:
            name = self.ensure_collection(dimension)
            expr = self._scope_filter(user_id, knowledge_base_id)
            hits = self.client().search(
                collection_name=name,
                data=[vector],
                anns_field="vector",
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                filter=expr,
                output_fields=["user_id", "knowledge_base_id", "file_id", "chunk_id", "metadata"],
            )
        except Exception as e:
            raise RuntimeError("Milvus search failed") from e
        results = []
        for hit in hits[0] if hits else []:
            entity = hit.get("entity") or {}
            results.append(
                {
                    "chunk_id": entity.get("chunk_id"),
                    "file_id": entity.get("file_id"),
                    "metadata": entity.get("metadata") or {},
                    "vector_score": float(hit.get("distance", hit.get("score", 0.0))),
                    "recall_types": ["vector"],
                }
            )
        return results

    def preview_file(self, user_id: str, knowledge_base_id: str, file_id: str) -> list[dict[str, Any]]:
        try:
            client = self.client()
            rows: list[dict[str, Any]] = []
            expr = self._scope_filter(user_id, knowledge_base_id, file_id)
            for name in self._collection_names():
                self._load_collection(name)
                rows.extend(
                    client.query(
                        collection_name=name,
                        filter=expr,
                        output_fields=["chunk_id", "file_id", "metadata"],
                        limit=100,
                    )
                )
            return rows
        except Exception as e:
            raise RuntimeError("Milvus search failed") from e


class Neo4jStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._driver = None
        self._constraints_ready = False

    def enabled(self) -> bool:
        return bool(self.settings.NEO4J_URI and self.settings.NEO4J_USER and self.settings.NEO4J_PASSWORD)

    # 懒加载模式：首次调用才创建驱动，全局复用同一个驱动
    def driver(self):
        if not self.enabled():
            return None
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.settings.NEO4J_URI,
                auth=(self.settings.NEO4J_USER, self.settings.NEO4J_PASSWORD),
            )
            self.ensure_constraints()
        return self._driver

    def ensure_constraints(self) -> None:
        if self._constraints_ready or self._driver is None:
            return
        queries = [
            """
            CREATE CONSTRAINT user_id_unique IF NOT EXISTS
            FOR (u:User) REQUIRE u.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT knowledge_base_scope_unique IF NOT EXISTS
            FOR (kb:KnowledgeBase) REQUIRE (kb.id, kb.user_id) IS UNIQUE
            """,
            """
            CREATE CONSTRAINT file_scope_unique IF NOT EXISTS
            FOR (f:File) REQUIRE (f.id, f.user_id, f.knowledge_base_id) IS UNIQUE
            """,
            """
            CREATE CONSTRAINT chunk_scope_unique IF NOT EXISTS
            FOR (c:Chunk) REQUIRE (c.id, c.user_id, c.knowledge_base_id) IS UNIQUE
            """,
            """
            CREATE CONSTRAINT entity_scope_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE (e.user_id, e.knowledge_base_id, e.type, e.canonical_name) IS UNIQUE
            """,
        ]
        optional_queries = [
            """
            CREATE INDEX related_to_scope_index IF NOT EXISTS
                FOR ()-[r:RELATED_TO]-() ON (r.user_id, r.knowledge_base_id, r.file_id)
            """,
        ]
        try:
            with self._driver.session() as session:
                for query in queries:
                    session.run(query)
                for query in optional_queries:
                    try:
                        session.run(query)
                    except Exception:
                        pass
            self._constraints_ready = True
        except Exception:
            self._constraints_ready = False

    def upsert_chunks(
            self,
            rows: list[dict[str, Any]],
            entities_by_chunk: dict[str, list[dict[str, Any]]] | None = None,
            relations_by_chunk: dict[str, list[dict[str, Any]]] | None = None,
            triples_by_chunk: dict[str, list[dict[str, Any]]] | None = None,
            raise_errors: bool = False,
    ) -> None:
        driver = self.driver()
        if driver is None:
            if raise_errors:
                raise RuntimeError("Neo4j connection settings are not configured")
            return
        if triples_by_chunk and relations_by_chunk is None:
            relations_by_chunk = {}
            entities_by_chunk = entities_by_chunk or {}
            for chunk_id, triples in triples_by_chunk.items():
                relations = []
                entities = entities_by_chunk.setdefault(chunk_id, [])
                for triple in triples:
                    subject = str(triple.get("subject") or "").strip()
                    obj = str(triple.get("object") or "").strip()
                    predicate = str(triple.get("predicate") or "").strip()
                    if not subject or not obj or not predicate:
                        continue
                    for name in (subject, obj):
                        entities.append(
                            {
                                "name": name,
                                "canonical_name": name.casefold(),
                                "type": "概念",
                                "aliases": [],
                                "evidence": str(triple.get("evidence") or "")[:1000],
                                "confidence": float(triple.get("confidence") or 0),
                            }
                        )
                    relations.append(
                        {
                            "subject_name": subject,
                            "subject_canonical_name": subject.casefold(),
                            "subject_type": "概念",
                            "object_name": obj,
                            "object_canonical_name": obj.casefold(),
                            "object_type": "概念",
                            "predicate": predicate,
                            "predicate_key": predicate,
                            "evidence": str(triple.get("evidence") or "")[:1000],
                            "confidence": float(triple.get("confidence") or 0),
                            "source_chunk_ids": [chunk_id],
                            "cross_chunk": False,
                        }
                    )
                relations_by_chunk[chunk_id] = relations

        prepared_rows: list[dict[str, Any]] = []
        has_entities = False
        has_relationships = False
        for row in rows:
            chunk_id = row["chunk_id"]
            entities = (entities_by_chunk or {}).get(chunk_id) or []
            relationships = (relations_by_chunk or {}).get(chunk_id) or []
            has_entities = has_entities or bool(entities)
            has_relationships = has_relationships or bool(relationships)
            prepared_rows.append(
                {
                    "user_id": row["user_id"],
                    "knowledge_base_id": row["knowledge_base_id"],
                    "file_id": row["file_id"],
                    "chunk_id": chunk_id,
                    "chunk_index": row.get("chunk_index"),
                    "content": row.get("content") or "",
                    "metadata_json": json.dumps(row.get("metadata") or {}, ensure_ascii=False),
                    "entities": entities,
                    "relationships": relationships,
                }
            )
        if not prepared_rows:
            return

        chunk_query = """
        UNWIND $rows AS row
        MERGE (u:User {id: row.user_id})
        MERGE (kb:KnowledgeBase {id: row.knowledge_base_id, user_id: row.user_id})
        MERGE (f:File {id: row.file_id, user_id: row.user_id, knowledge_base_id: row.knowledge_base_id})
        MERGE (u)-[:OWNS]->(kb)
        MERGE (kb)-[:HAS_FILE]->(f)
        MERGE (c:Chunk {
            id: row.chunk_id,
            user_id: row.user_id,
            knowledge_base_id: row.knowledge_base_id,
            file_id: row.file_id
        })
        SET c.chunk_index = row.chunk_index,
            c.content = row.content,
            c.metadata_json = row.metadata_json
        MERGE (f)-[:HAS_CHUNK]->(c)
        """
        entity_query = """
        UNWIND $rows AS row
        MATCH (c:Chunk {
            id: row.chunk_id,
            user_id: row.user_id,
            knowledge_base_id: row.knowledge_base_id,
            file_id: row.file_id
        })
        UNWIND row.entities AS entity
        MERGE (e:Entity {
            user_id: row.user_id,
            knowledge_base_id: row.knowledge_base_id,
            type: entity.type,
            canonical_name: entity.canonical_name
        })
        ON CREATE SET e.created_at = datetime()
        SET e.name = entity.name,
            e.aliases = coalesce(entity.aliases, []),
            e.updated_at = datetime()
        MERGE (c)-[m:MENTIONS {
            user_id: row.user_id,
            knowledge_base_id: row.knowledge_base_id,
            file_id: row.file_id,
            chunk_id: row.chunk_id
        }]->(e)
        SET m.evidence = entity.evidence, m.confidence = entity.confidence
        """
        relation_query = """
        UNWIND $rows AS row
        MATCH (c:Chunk {
            id: row.chunk_id,
            user_id: row.user_id,
            knowledge_base_id: row.knowledge_base_id,
            file_id: row.file_id
        })
        UNWIND row.relationships AS rel
        MERGE (s:Entity {
            user_id: row.user_id,
            knowledge_base_id: row.knowledge_base_id,
            type: rel.subject_type,
            canonical_name: rel.subject_canonical_name
        })
        SET s.name = rel.subject_name
        MERGE (o:Entity {
            user_id: row.user_id,
            knowledge_base_id: row.knowledge_base_id,
            type: rel.object_type,
            canonical_name: rel.object_canonical_name
        })
        SET o.name = rel.object_name
        MERGE (c)-[sm:MENTIONS {
            user_id: row.user_id,
            knowledge_base_id: row.knowledge_base_id,
            file_id: row.file_id,
            chunk_id: row.chunk_id
        }]->(s)
        SET sm.evidence = rel.evidence, sm.confidence = rel.confidence
        MERGE (c)-[om:MENTIONS {
            user_id: row.user_id,
            knowledge_base_id: row.knowledge_base_id,
            file_id: row.file_id,
            chunk_id: row.chunk_id
        }]->(o)
        SET om.evidence = rel.evidence, om.confidence = rel.confidence
        MERGE (s)-[r:RELATED_TO {
            user_id: row.user_id,
            knowledge_base_id: row.knowledge_base_id,
            file_id: row.file_id,
            chunk_id: row.chunk_id,
            predicate_key: rel.predicate_key
        }]->(o)
        SET r.predicate = rel.predicate,
            r.evidence = rel.evidence,
            r.confidence = rel.confidence,
            r.source_chunk_ids = coalesce(rel.source_chunk_ids, [row.chunk_id]),
            r.cross_chunk = coalesce(rel.cross_chunk, false)
        """
        with driver.session() as session:
            session.run(chunk_query, rows=prepared_rows)
            if has_entities:
                session.run(entity_query, rows=prepared_rows)
            if has_relationships:
                session.run(relation_query, rows=prepared_rows)

    def delete_by_scope(
            self,
            user_id: str,
            knowledge_base_id: str,
            file_id: str | None = None,
            raise_errors: bool = False,
    ) -> None:
        driver = self.driver()
        if driver is None:
            if raise_errors:
                raise RuntimeError("Neo4j connection settings are not configured")
            return
        with driver.session() as session:
            if file_id:
                params = {"user_id": user_id, "knowledge_base_id": knowledge_base_id, "file_id": file_id}
                session.run(
                    """
                    MATCH (f:File {id: $file_id, user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                        -[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (c)-[m:MENTIONS]->()
                    DELETE m
                    """,
                    **params,
                )
                session.run(
                    """
                    MATCH (f:File {id: $file_id, user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                        -[:HAS_CHUNK]->(c:Chunk)
                    MATCH ()-[r:RELATED_TO {user_id: $user_id, knowledge_base_id: $knowledge_base_id}]-()
                    WHERE r.file_id = $file_id OR c.id IN coalesce(r.source_chunk_ids, [])
                    DELETE r
                    """,
                    **params,
                )
                session.run(
                    """
                    MATCH (f:File {id: $file_id, user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                        -[:HAS_CHUNK]->(c:Chunk)
                    DETACH DELETE c
                    """,
                    **params,
                )
                session.run(
                    """
                    MATCH (f:File {id: $file_id, user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                    DETACH DELETE f
                    """,
                    **params,
                )
                session.run(
                    """
                    MATCH (e:Entity {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                    WHERE NOT (e)--()
                    DELETE e
                    """,
                    **params,
                )
            else:
                session.run(
                    """
                    MATCH (kb:KnowledgeBase {id: $knowledge_base_id, user_id: $user_id})
                        -[:HAS_FILE]->(:File)-[:HAS_CHUNK]->(c:Chunk)
                    DETACH DELETE c
                    """,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )
                session.run(
                    """
                    MATCH (kb:KnowledgeBase {id: $knowledge_base_id, user_id: $user_id})
                        -[:HAS_FILE]->(f:File)
                    DETACH DELETE f
                    """,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )
                session.run(
                    """
                    MATCH (e:Entity {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                    DETACH DELETE e
                    """,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )
                session.run(
                    """
                    MATCH (kb:KnowledgeBase {id: $knowledge_base_id, user_id: $user_id})
                    DETACH DELETE kb
                    """,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )

    def graph_search(
            self,
            user_id: str,
            knowledge_base_id: str,
            query_text: str,
            top_k: int,
            candidate_chunk_ids: list[str] | None = None,
            query_terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        driver = self.driver()
        if driver is None:
            return []
        query_terms = query_terms if query_terms is not None else _graph_query_terms(query_text)
        if candidate_chunk_ids:
            query = """
            MATCH (seed:Chunk {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
            WHERE seed.id IN $candidate_chunk_ids
            CALL (seed) {
                MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c:Chunk {
                    user_id: $user_id,
                    knowledge_base_id: $knowledge_base_id
                })
                WITH c, e,
                    CASE WHEN any(term IN $query_terms WHERE
                        toLower(coalesce(e.name, "")) CONTAINS term
                        OR any(alias IN coalesce(e.aliases, []) WHERE toLower(alias) CONTAINS term)
                    ) THEN 2.0 ELSE 0.5 END AS score
                RETURN c, score

                UNION ALL

                MATCH (seed)-[:MENTIONS]->(:Entity)
                    -[r:RELATED_TO {user_id: $user_id, knowledge_base_id: $knowledge_base_id}]-
                    (neighbor:Entity)<-[:MENTIONS]-(c:Chunk {
                        user_id: $user_id,
                        knowledge_base_id: $knowledge_base_id
                    })
                WITH c, r, neighbor,
                    1.0 + coalesce(r.confidence, 0.5)
                    + CASE WHEN any(term IN $query_terms WHERE
                        toLower(coalesce(neighbor.name, "")) CONTAINS term
                        OR any(alias IN coalesce(neighbor.aliases, []) WHERE toLower(alias) CONTAINS term)
                    ) THEN 1.0 ELSE 0.0 END
                    + CASE WHEN any(term IN $query_terms WHERE
                        toLower(coalesce(r.predicate, "")) CONTAINS term
                        OR toLower(coalesce(r.evidence, "")) CONTAINS term
                    ) THEN 1.0 ELSE 0.0 END AS score
                RETURN c, score

                UNION ALL

                MATCH (seed)-[:MENTIONS]->(:Entity)
                    -[r:RELATED_TO {user_id: $user_id, knowledge_base_id: $knowledge_base_id}]-
                    (:Entity)
                WITH DISTINCT r
                MATCH (c:Chunk {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                WHERE c.id = r.chunk_id OR c.id IN coalesce(r.source_chunk_ids, [])
                WITH c, r, 1.0 + coalesce(r.confidence, 0.5) AS score
                RETURN c, score
            }
            WITH c, sum(score) AS graph_score
            WHERE graph_score > 0
            RETURN c.id AS chunk_id, c.file_id AS file_id, c.chunk_index AS chunk_index,
                   c.content AS content, graph_score AS graph_score
            ORDER BY graph_score DESC
            LIMIT $top_k
            """
        else:
            query = """
            CALL () {
                MATCH (c:Chunk {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                    -[:MENTIONS]->(e:Entity {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                WHERE any(term IN $query_terms WHERE
                    toLower(coalesce(e.name, "")) CONTAINS term
                    OR any(alias IN coalesce(e.aliases, []) WHERE toLower(alias) CONTAINS term)
                )
                WITH c, count(DISTINCT e) * 2.0 AS score
                RETURN c, score

                UNION ALL

                MATCH (s:Entity {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                    -[r:RELATED_TO {user_id: $user_id, knowledge_base_id: $knowledge_base_id}]-
                    (o:Entity {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                WHERE any(term IN $query_terms WHERE
                    toLower(coalesce(s.name, "")) CONTAINS term
                    OR any(alias IN coalesce(s.aliases, []) WHERE toLower(alias) CONTAINS term)
                    OR toLower(coalesce(o.name, "")) CONTAINS term
                    OR any(alias IN coalesce(o.aliases, []) WHERE toLower(alias) CONTAINS term)
                    OR toLower(coalesce(r.predicate, "")) CONTAINS term
                    OR toLower(coalesce(r.evidence, "")) CONTAINS term
                )
                MATCH (c:Chunk {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
                WHERE c.id = r.chunk_id OR c.id IN coalesce(r.source_chunk_ids, [])
                WITH c, r, 1.0 + coalesce(r.confidence, 0.5) AS score
                RETURN c, score
            }
            WITH c, sum(score) AS graph_score
            WHERE graph_score > 0
            RETURN c.id AS chunk_id, c.file_id AS file_id, c.chunk_index AS chunk_index,
                   c.content AS content, graph_score AS graph_score
            ORDER BY graph_score DESC
            LIMIT $top_k
            """
        try:
            with driver.session() as session:
                records = session.run(query,
                                      user_id=user_id,
                                      knowledge_base_id=knowledge_base_id,
                                      query_text=query_text, top_k=top_k,
                                      candidate_chunk_ids=candidate_chunk_ids or [],
                                      query_terms=query_terms)
                return [
                    {
                        "chunk_id": row["chunk_id"],
                        "file_id": row["file_id"],
                        "chunk_index": row["chunk_index"],
                        "content": row["content"],
                        "metadata": {},
                        "graph_score": float(row["graph_score"]),
                        "recall_types": ["graph"],
                    }
                    for row in records
                ]
        except Exception as e:
            raise RuntimeError("Neo4j search failed") from e

    def preview_file(self, user_id: str, knowledge_base_id: str, file_id: str) -> dict[str, Any]:
        empty = {
            "graph": {"nodes": [], "edges": [], "stats": {"node_count": 0, "edge_count": 0}},
            "nodes": [],
            "relations": [],
            "raw": [],
            "truncated": False,
        }
        driver = self.driver()
        if driver is None:
            return empty
        graph_query = """
        MATCH (source:Entity {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
            -[r:RELATED_TO {user_id: $user_id, knowledge_base_id: $knowledge_base_id}]->
            (target:Entity {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
        WHERE r.file_id = $file_id OR $file_id IN coalesce(r.source_chunk_ids, [])
        WITH DISTINCT source, r, target
        ORDER BY coalesce(r.confidence, 0.0) DESC
        LIMIT $edge_limit
        RETURN
            source {.name, .canonical_name, .type, .aliases} AS source,
            target {.name, .canonical_name, .type, .aliases} AS target,
            {
                id: elementId(r),
                predicate: r.predicate,
                predicate_key: r.predicate_key,
                confidence: r.confidence,
                evidence: r.evidence,
                chunk_id: r.chunk_id,
                source_chunk_ids: coalesce(r.source_chunk_ids, []),
                cross_chunk: coalesce(r.cross_chunk, false)
            } AS relation
        """
        mention_query = """
        MATCH (c:Chunk {user_id: $user_id, knowledge_base_id: $knowledge_base_id, file_id: $file_id})
            -[m:MENTIONS]->(entity:Entity {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
        RETURN
            entity.name AS name,
            entity.type AS type,
            entity.canonical_name AS canonical_name,
            entity.aliases AS aliases,
            collect(DISTINCT c.id)[..5] AS chunk_ids,
            max(coalesce(m.confidence, 0.0)) AS confidence,
            head([item IN collect(m.evidence) WHERE item IS NOT NULL AND item <> ""]) AS evidence
        LIMIT $node_limit
        """
        raw_query = """
        CALL () {
            MATCH (n:Entity {user_id: $user_id, knowledge_base_id: $knowledge_base_id})
            WHERE EXISTS {
                MATCH (:Chunk {user_id: $user_id, knowledge_base_id: $knowledge_base_id, file_id: $file_id})--(n)
            }
            OPTIONAL MATCH (n)-[r:RELATED_TO {user_id: $user_id, knowledge_base_id: $knowledge_base_id}]-(m:Entity)
            WHERE r IS NULL OR r.file_id = $file_id OR $file_id IN coalesce(r.source_chunk_ids, [])
            RETURN labels(n) AS labels, properties(n) AS node, type(r) AS relation, properties(r) AS relation_props,
                   properties(m) AS target
        }
        RETURN labels, node, relation, relation_props, target
        LIMIT $raw_limit
        """
        node_limit = 80
        edge_limit = 120
        raw_limit = 100
        try:
            with driver.session() as session:
                relation_records = [
                    dict(row)
                    for row in session.run(
                        graph_query,
                        user_id=user_id,
                        knowledge_base_id=knowledge_base_id,
                        file_id=file_id,
                        edge_limit=edge_limit + 1,
                    )
                ]
                mention_records = [
                    dict(row)
                    for row in session.run(
                        mention_query,
                        user_id=user_id,
                        knowledge_base_id=knowledge_base_id,
                        file_id=file_id,
                        node_limit=node_limit + 1,
                    )
                ]
                raw = [
                    dict(row)
                    for row in session.run(
                        raw_query,
                        user_id=user_id,
                        knowledge_base_id=knowledge_base_id,
                        file_id=file_id,
                        raw_limit=raw_limit,
                    )
                ]
            truncated = len(relation_records) > edge_limit or len(mention_records) > node_limit
            relation_records = relation_records[:edge_limit]
            mention_records = mention_records[:node_limit]

            nodes_by_id: dict[str, dict[str, Any]] = {}
            for record in mention_records:
                node_id = f"{record.get('type') or 'Entity'}:{record.get('canonical_name')}"
                nodes_by_id[node_id] = {
                    "id": node_id,
                    "label": record.get("name") or record.get("canonical_name") or "",
                    "name": record.get("name") or record.get("canonical_name") or "",
                    "type": record.get("type") or "Entity",
                    "aliases": record.get("aliases") or [],
                    "confidence": float(record.get("confidence") or 0.0),
                    "evidence": record.get("evidence") or "",
                    "chunk_ids": record.get("chunk_ids") or [],
                }

            edges: list[dict[str, Any]] = []
            seen_edge_ids: set[str] = set()
            for record in relation_records:
                source = record.get("source") or {}
                target = record.get("target") or {}
                relation = record.get("relation") or {}
                source_id = f"{source.get('type') or 'Entity'}:{source.get('canonical_name')}"
                target_id = f"{target.get('type') or 'Entity'}:{target.get('canonical_name')}"
                predicate_key = relation.get("predicate_key") or relation.get("predicate") or "RELATED_TO"
                edge_id = str(relation.get("id") or f"{source_id}->{target_id}:{predicate_key}:{relation.get('chunk_id') or ''}")
                if edge_id in seen_edge_ids:
                    continue
                seen_edge_ids.add(edge_id)
                nodes_by_id.setdefault(
                    source_id,
                    {
                        "id": source_id,
                        "label": source.get("name") or source.get("canonical_name") or "",
                        "name": source.get("name") or source.get("canonical_name") or "",
                        "type": source.get("type") or "Entity",
                        "aliases": source.get("aliases") or [],
                        "confidence": 0.0,
                        "evidence": "",
                        "chunk_ids": [],
                    },
                )
                nodes_by_id.setdefault(
                    target_id,
                    {
                        "id": target_id,
                        "label": target.get("name") or target.get("canonical_name") or "",
                        "name": target.get("name") or target.get("canonical_name") or "",
                        "type": target.get("type") or "Entity",
                        "aliases": target.get("aliases") or [],
                        "confidence": 0.0,
                        "evidence": "",
                        "chunk_ids": [],
                    },
                )
                chunk_ids = relation.get("source_chunk_ids") or []
                if not chunk_ids and relation.get("chunk_id"):
                    chunk_ids = [relation.get("chunk_id")]
                edges.append(
                    {
                        "id": edge_id,
                        "source": source_id,
                        "target": target_id,
                        "label": relation.get("predicate") or predicate_key,
                        "predicate": relation.get("predicate") or predicate_key,
                        "confidence": float(relation.get("confidence") or 0.0),
                        "evidence": relation.get("evidence") or "",
                        "chunk_ids": chunk_ids,
                        "cross_chunk": bool(relation.get("cross_chunk")),
                    }
                )

            nodes = list(nodes_by_id.values())[:node_limit]
            node_ids = {node["id"] for node in nodes}
            edges = [edge for edge in edges if edge["source"] in node_ids and edge["target"] in node_ids]
            return {
                "graph": {
                    "nodes": nodes,
                    "edges": edges,
                    "stats": {"node_count": len(nodes), "edge_count": len(edges)},
                },
                "nodes": nodes,
                "relations": edges,
                "raw": raw,
                "truncated": truncated,
            }
        except Exception as e:
            raise RuntimeError("Neo4j preview failed") from e


es_store = ElasticsearchStore()
milvus_store = MilvusStore()
neo4j_store = Neo4jStore()
