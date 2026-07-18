# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.core.deps import db_session, get_current_user
from app.core.response import ok
from app.models import User
from app.schemas import RetrievalIn
from app.services.retrieval_engine import retrieve

router = APIRouter()


@router.post("/debug")
def debug_retrieval(
    payload: RetrievalIn,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(db_session),
):
    results, record = retrieve(db, current_user, payload)
    return ok(request, {"record_id": record.id, "results": results})
