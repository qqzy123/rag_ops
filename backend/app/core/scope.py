# SPDX-FileCopyrightText: 2026 zhoujian
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy import Select

from app.models import User


def scoped_select(stmt: Select, model: type, current_user: User) -> Select:
    if current_user.role == "admin":
        return stmt
    return stmt.where(model.user_id == current_user.id)


def scoped_get_filter(model: type, resource_id: str, current_user: User) -> list:
    conditions = [model.id == resource_id]
    if current_user.role != "admin":
        conditions.append(model.user_id == current_user.id)
    return conditions


def owner_id_for_create(current_user: User, requested_user_id: str | None = None) -> str:
    if current_user.role == "admin" and requested_user_id:
        return requested_user_id
    return current_user.id
