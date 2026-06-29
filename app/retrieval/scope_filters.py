"""Translate a validated scope into lexical-backend filters.

The single place that turns a ``RetrievalScope`` into the filter dict the
lexical backend understands. Both the retrieval gateway and the delete use case
go through it, so scope-enforcement keys are applied consistently and a
caller-supplied filter can never shadow them.
"""
from __future__ import annotations

from typing import Any

from app.retrieval.types import RetrievalScope

# Keys that carry scope enforcement; a caller filter may not override them.
_RESERVED_KEYS = {"service_name", "tenant_id", "collections", "collection"}


def lexical_filters_for(scope: RetrievalScope) -> dict[str, Any]:
    """Return the lexical filter dict for a scope.

    Always includes ``service_name``, ``tenant_id``, and ``collections``; merges
    any additional ``scope.filters`` except the reserved scope-enforcement keys.
    """
    filters: dict[str, Any] = {
        "service_name": scope.service_name,
        "tenant_id": scope.tenant_id,
        "collections": scope.collections,
    }
    for key, value in (scope.filters or {}).items():
        if key not in _RESERVED_KEYS:
            filters[key] = value
    return filters
