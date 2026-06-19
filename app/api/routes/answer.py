from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import AnswerRequest, AnswerResponse, ChunkResult, StreamEvent
from app.auth import Principal, enforce_scope, require_principal
from app.retrieval import RetrievalScope, RetrieveRequest
from app.services.generation_service import GenerationServiceError

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


def _build_retrieval_scope(body: AnswerRequest) -> RetrievalScope:
    """Build RetrievalScope from request body."""
    return RetrievalScope(
        service_name=body.service_name,
        tenant_id=body.tenant_id,
        collections=body.collections,
        filters=body.filters,
    )


def _execute_retrieval(request: Request, body: AnswerRequest):
    """Execute retrieval; domain errors propagate to exception handlers."""
    scope = _build_retrieval_scope(body)
    retrieve_request = RetrieveRequest(
        query=body.query,
        retrieval_mode=body.mode,
        limit=body.limit,
        scope=scope,
    )
    return request.app.state.retrieve_use_case.execute(retrieve_request)


def _map_chunks_to_sources(chunks):
    """Map RetrievedChunk to source format expected by GenerationService."""
    return [
        {
            "document_id": chunk.metadata.get("document_id", "unknown"),
            "original_filename": chunk.metadata.get("source_label", "unknown"),
            "chunk_index": chunk.metadata.get("chunk_index", 0),
            "score": chunk.score,
            "text": chunk.content,
        }
        for chunk in chunks
    ]


def _map_chunks_to_chunk_results(chunks):
    """Map RetrievedChunk to ChunkResult schema."""
    return [
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
        for chunk in chunks
    ]


@router.post("/answer", response_model=AnswerResponse)
def answer(
    request: Request,
    body: AnswerRequest,
    principal: Principal = Depends(require_principal),
):
    """
    Retrieve scoped chunks and generate a complete answer.

    Returns a generated answer with source references and trace information.
    """
    enforce_scope(
        principal,
        service_name=body.service_name,
        tenant_id=body.tenant_id,
        collections=body.collections,
    )

    # Retrieve scoped chunks
    result = _execute_retrieval(request, body)

    # Handle empty retrieval
    if len(result.chunks) == 0:
        return AnswerResponse(
            answer="I couldn't find any relevant information to answer your question.",
            sources=[],
            trace_id=result.trace_id or "unknown",
        )

    # Map chunks to generation service format
    sources = _map_chunks_to_sources(result.chunks)

    # Generate answer (GenerationServiceError propagates to the handler).
    answer_text = request.app.state.generation_service.answer_question(
        body.query, sources
    )

    # Map chunks to response format
    source_chunks = _map_chunks_to_chunk_results(result.chunks)

    return AnswerResponse(
        answer=answer_text,
        sources=source_chunks,
        trace_id=result.trace_id or "unknown",
    )


@router.post("/answer/stream")
async def answer_stream(
    request: Request,
    body: AnswerRequest,
    principal: Principal = Depends(require_principal),
):
    """
    Retrieve scoped chunks and stream the generated answer via SSE.

    Returns Server-Sent Events with content and completion signal.
    """
    enforce_scope(
        principal,
        service_name=body.service_name,
        tenant_id=body.tenant_id,
        collections=body.collections,
    )

    # Retrieve scoped chunks
    result = _execute_retrieval(request, body)

    # Handle empty retrieval
    if len(result.chunks) == 0:
        async def empty_stream():
            event = StreamEvent(
                event="content",
                data="I couldn't find any relevant information to answer your question.",
                done=False,
            )
            yield f"data: {event.model_dump_json()}\n\n"

            done_event = StreamEvent(
                event="done",
                data="",
                done=True,
            )
            yield f"data: {done_event.model_dump_json()}\n\n"

        return StreamingResponse(
            empty_stream(),
            media_type="text/event-stream",
        )

    # Map chunks to generation service format
    sources = _map_chunks_to_sources(result.chunks)

    # Stream answer generation
    async def event_generator() -> AsyncIterator[str]:
        try:
            async for token in request.app.state.generation_service.stream_answer_question(
                body.query, sources
            ):
                event = StreamEvent(
                    event="content",
                    data=token,
                    done=False,
                )
                yield f"data: {event.model_dump_json()}\n\n"

            # Emit completion signal
            done_event = StreamEvent(
                event="done",
                data="",
                done=True,
            )
            yield f"data: {done_event.model_dump_json()}\n\n"

        except GenerationServiceError as exc:
            error_event = StreamEvent(
                event="error",
                data=f"Answer generation failed: {exc}",
                done=True,
            )
            yield f"data: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
