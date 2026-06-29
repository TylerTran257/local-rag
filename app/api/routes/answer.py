from __future__ import annotations

import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.answer.contracts import AnswerRequest as AnswerUseCaseRequest
from app.api.schemas import AnswerRequest, AnswerResponse, ChunkResult, StreamEvent
from app.auth import Principal, enforce_scope, require_principal
from app.retrieval import RetrievalScope
from app.services.generation_service import GenerationServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_use_case_request(body: AnswerRequest) -> AnswerUseCaseRequest:
    return AnswerUseCaseRequest(
        query=body.query,
        retrieval_mode=body.mode,
        limit=body.limit,
        scope=RetrievalScope(
            service_name=body.service_name,
            tenant_id=body.tenant_id,
            collections=body.collections,
            filters=body.filters,
        ),
    )


def _to_chunk_results(chunks) -> list[ChunkResult]:
    return [
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

    # Domain/generation errors propagate to the registered exception handlers.
    result = request.app.state.answer_use_case.execute(_to_use_case_request(body))

    return AnswerResponse(
        answer=result.answer,
        sources=_to_chunk_results(result.sources),
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

    # Retrieval runs eagerly inside stream(); a RetrievalError here propagates
    # to the exception handlers before any SSE bytes are sent.
    answer_stream = await request.app.state.answer_use_case.stream(
        _to_use_case_request(body)
    )

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for token in answer_stream.tokens:
                event = StreamEvent(event="content", data=token, done=False)
                yield f"data: {event.model_dump_json()}\n\n"

            done_event = StreamEvent(event="done", data="", done=True)
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
