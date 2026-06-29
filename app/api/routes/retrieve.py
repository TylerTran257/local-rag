from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.api.schemas import ChunkResult, RetrieveRequest as RetrieveRequestSchema, RetrieveResponse
from app.auth import Principal, enforce_scope, require_principal
from app.retrieval import RetrievalScope, RetrieveRequest

logger = logging.getLogger(__name__)

router = APIRouter()


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
            domain_metadata=chunk.domain_metadata(),
        )
        for chunk in result.chunks
    ]

    return RetrieveResponse(
        chunks=chunks,
        trace_id=result.trace_id or "unknown",
    )
