# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from datetime import UTC, datetime

from sqlalchemy import select, update

from app.core.config import get_settings
from app.core.errors import ApiError
from app.db.session import SessionLocal
from app.models import IngestionTask, KnowledgeBase
from app.services.cleanup import delete_knowledge_base_resources
from app.services.ingestion import ingest_file, write_task_target


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class DatabaseTaskConsumer:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="db-task-consumer", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _run(self) -> None:
        settings = get_settings()
        # 循环持续运行，**直到 `_stop_event.set()` 被调用（收到停止信号）才跳出循环**
        while not self._stop_event.is_set():
            try:
                consumed = self.consume_once()
            except Exception as exc:
                consumed = False
                self._last_error = str(exc)
            if not consumed:
                self._stop_event.wait(settings.TASK_CONSUMER_POLL_INTERVAL_SECONDS)

    def consume_once(self) -> bool:
        with SessionLocal() as db:
            task = self._claim_next_task(db)
            if not task:
                return False

            try:
                self._execute_task(db, task)
                self._last_error = None
            except ApiError as exc:
                self._last_error = exc.message
                self._mark_failed(db, task, exc.message)
            except Exception as exc:
                self._last_error = str(exc)
                self._mark_failed(db, task, str(exc))
            return True

    def _claim_next_task(self, db) -> IngestionTask | None:
        task_id = db.scalar(
            select(IngestionTask.id)
            .where(IngestionTask.status == "pending")
            .order_by(IngestionTask.created_at.asc())
            .limit(1)
        )
        if not task_id:
            return None

        result = db.execute(
            update(IngestionTask)
            .where(IngestionTask.id == task_id, IngestionTask.status == "pending")
            .values(
                status="running",
                message="任务已被数据库后台消费者接收",
                error=None,
                started_at=_utcnow(),
                finished_at=None,
            )
        )
        db.commit()
        if result.rowcount != 1:
            return None
        return db.get(IngestionTask, task_id)

    def _execute_task(self, db, task: IngestionTask) -> None:
        payload = task.payload_json or {}
        if task.task_type == "ingest_file":
            ingest_file(
                db,
                task,
                payload.get("strategy_id"),
                payload.get("model_id"),
                payload.get("graph_model_id"),
                bool(payload.get("overwrite")),
                payload.get("load_options"),
                payload.get("strategy_snapshot"),
                bool(payload.get("enable_neo4j")),
            )
            return

        if task.task_type.startswith("write_"):
            target = payload.get("target") or task.task_type.removeprefix("write_")
            write_task_target(db, task, target, payload.get("embedding_model_id"), payload.get("graph_model_id"))
            return

        # if task.task_type == "delete_file":
        #     delete_file_by_id(db, task)
        #     return

        # if task.task_type == "delete_knowledge_base":
        #     self._delete_knowledge_base(db, task)
        #     return

        raise ValueError(f"unsupported task type: {task.task_type}")

    def _delete_knowledge_base(self, db, task: IngestionTask) -> None:
        kb = db.scalar(
            select(KnowledgeBase).where(KnowledgeBase.id == task.knowledge_base_id,
                                        KnowledgeBase.user_id == task.user_id)
        )
        if kb:
            delete_knowledge_base_resources(db, kb)
        task.status = "success"
        task.progress = 100
        task.stage = "complete"
        task.message = "知识库删除完成"
        task.finished_at = _utcnow()
        db.commit()

    def _mark_failed(self, db, task: IngestionTask, message: str) -> None:
        task.status = "failed"
        task.error = message
        task.finished_at = _utcnow()
        db.commit()


task_consumer = DatabaseTaskConsumer()
