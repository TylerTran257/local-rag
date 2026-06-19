from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.schemas import ChunkResult, RetrieveRequest as RetrieveRequestSchema, RetrieveResponse
from app.auth import Principal, enforce_scope, require_principal
from app.retrieval import RetrievalScope, RetrieveRequest

logger = logging.getLogger(__name__)

router = APIRouter()

_CORE_CHUNK_METADATA_KEYS = frozenset(
    {
        "service_name",
        "tenant_id",
        "collection",
        "source_type",
        "source_label",
        "document_id",
        "original_filename",
        "chunk_index",
    }
)


def _extract_domain_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return non-core metadata fields for API responses."""
    return {
        key: value
        for key, value in metadata.items()
        if key not in _CORE_CHUNK_METADATA_KEYS
    }


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    request: Request,
    body: RetrieveRequestSchema,
    principal: Principal = Depends(require_principal),
):
    """
    Retrieve scoped chunks for a query.

    Returns retrieved chunks with scores, metadata, and trace information.
    All results are scoped by service_name, tenant_id, collections, and filters,
    and the caller's API key must be granted that scope.
    """
    enforce_scope(
        principal,
        service_name=body.service_name,
        tenant_id=body.tenant_id,
        collections=body.collections,
    )

    scope = RetrievalScope(
        service_name=body.service_name,
        tenant_id=body.tenant_id,
        collections=body.collections,
        filters=body.filters,
    )

    retrieve_request = RetrieveRequest(
        query=body.query,
        retrieval_mode=body.mode,
        limit=body.limit,
        scope=scope,
    )

    # Domain errors propagate to the registered exception handlers, which
    # render the uniform error envelope.
    result = request.app.state.retrieve_use_case.execute(retrieve_request)

    chunks = [
        ChunkResult(
            text=chunk.content,
            score=chunk.score,
            source_label=chunk.metadata.get("source_label", "unknown"),
            collection=chunk.metadata.get("collection", "unknown"),
            service_name=chunk.metadata.get("service_name", "unknown"),
            tenant_id=chunk.metadata.get("tenant_id", "unknown"),
            chunk_id=chunk.chunk_id,
            domain_metadata=_extract_domain_metadata(chunk.metadata),
        )
        for chunk in result.chunks
    ]

    return RetrieveResponse(
        chunks=chunks,
        trace_id=result.trace_id or "unknown",
    )
