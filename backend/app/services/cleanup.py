# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import DeleteAudit, DocumentChunk, FileAsset, KnowledgeBase
from app.services.external_stores import es_store, milvus_store, neo4j_store
from app.services.storage import storage


def delete_file_resources(db: Session, file_asset: FileAsset) -> None:
    es_store.delete_by_scope(file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)
    milvus_store.delete_by_scope(file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)
    neo4j_store.delete_by_scope(file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)
    storage.delete_object(file_asset.object_key, file_asset.user_id, file_asset.knowledge_base_id, file_asset.id)
    db.execute(delete(DocumentChunk).where(DocumentChunk.file_id == file_asset.id, DocumentChunk.user_id == file_asset.user_id))
    db.delete(file_asset)
    db.add(
        DeleteAudit(
            user_id=file_asset.user_id,
            resource_type="file",
            resource_id=file_asset.id,
            scope_json={"knowledge_base_id": file_asset.knowledge_base_id, "file_id": file_asset.id},
            status="success",
        )
    )
    db.commit()
    

def delete_knowledge_base_resources(db: Session, kb: KnowledgeBase) -> None:
    files = db.scalars(select(FileAsset).where(FileAsset.knowledge_base_id == kb.id, FileAsset.user_id == kb.user_id)).all()
    es_store.delete_by_scope(kb.user_id, kb.id)
    milvus_store.delete_by_scope(kb.user_id, kb.id)
    neo4j_store.delete_by_scope(kb.user_id, kb.id)
    storage.delete_prefix(f"{kb.user_id}/{kb.id}/", kb.user_id, kb.id)
    for file_asset in files:
        db.execute(delete(DocumentChunk).where(DocumentChunk.file_id == file_asset.id, DocumentChunk.user_id == kb.user_id))
        db.delete(file_asset)
    db.delete(kb)
    db.add(
        DeleteAudit(
            user_id=kb.user_id,
            resource_type="knowledge_base",
            resource_id=kb.id,
            scope_json={"knowledge_base_id": kb.id},
            status="success",
        )
    )
