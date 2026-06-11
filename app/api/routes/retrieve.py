import logging

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import ChunkResult, RetrieveRequest as RetrieveRequestSchema, RetrieveResponse
from app.retrieval import (
    InvalidRetrievalRequestError,
    NoIndexedCorpusError,
    RetrievalExecutionError,
    RetrievalScope,
    RetrieveRequest,
    RetrievedChunkValidationError,
    UnsupportedRetrievalModeError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: Request, body: RetrieveRequestSchema):
    """
    Retrieve scoped chunks for a query.

    Returns retrieved chunks with scores, metadata, and trace information.
    All results are scoped by service_name, tenant_id, collections, and filters.
    """
    # Build scope from request
    scope = RetrievalScope(
        service_name=body.service_name,
        tenant_id=body.tenant_id,
        collections=body.collections,
        filters=body.filters,
    )

    # Build retrieval request
    retrieve_request = RetrieveRequest(
        query=body.query,
        retrieval_mode=body.mode,
        limit=body.limit,
        scope=scope,
    )

    # Execute retrieval
    try:
        result = request.app.state.retrieve_use_case.execute(retrieve_request)
    except InvalidRetrievalRequestError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (
        UnsupportedRetrievalModeError,
        NoIndexedCorpusError,
        RetrievalExecutionError,
        RetrievedChunkValidationError,
    ) as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Map chunks to response format
    chunks = [
        ChunkResult(
            text=chunk.content,
            score=chunk.score,
            source_label=chunk.metadata.get("source_label", "unknown"),
            collection=chunk.metadata.get("collection", "unknown"),
            service_name=chunk.metadata.get("service_name", "unknown"),
            tenant_id=chunk.metadata.get("tenant_id", "unknown"),
            chunk_id=chunk.chunk_id,
        )
        for chunk in result.chunks
    ]

    return RetrieveResponse(
        chunks=chunks,
        trace_id=result.trace_id or "unknown",
    )
