"""Uniform JSON error envelope for all API endpoints.

Every handled error renders as::

    {"error": {"code": "...", "message": "...", "trace_id": "...", "details": {...}}}

so callers (including other agents) get a stable, machine-readable shape instead
of bare ``detail`` strings.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.auth.errors import AuthenticationError, AuthError, AuthorizationError
from app.ingest.contracts import EmptyDocumentError, MetadataValidationError
from app.profiles.store import ProfileEmbeddingModelImmutableError
from app.retrieval.errors import RetrievalError
from app.services.generation_service import GenerationServiceError
from app.services.text_extractor import TextExtractionError

logger = logging.getLogger(__name__)

# Retrieval domain error code -> HTTP status. Anything unlisted is a 500.
_RETRIEVAL_STATUS = {
    "INVALID_RETRIEVAL_REQUEST": 422,
    "NO_INDEXED_CORPUS": 404,
}


def _trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", None) or "unknown"


def _envelope(
    *, code: str, message: str, trace_id: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message, "trace_id": trace_id}
    if details:
        body["details"] = details
    return {"error": body}


def _json(status_code: int, body: dict[str, Any], trace_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={"X-Trace-Id": trace_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AuthError)
    async def _handle_auth(request: Request, exc: AuthError) -> JSONResponse:
        status = 401 if isinstance(exc, AuthenticationError) else 403
        if not isinstance(exc, (AuthenticationError, AuthorizationError)):
            status = 401
        trace_id = _trace_id(request)
        return _json(
            status,
            _envelope(code=exc.code, message=exc.message, trace_id=trace_id, details=exc.details),
            trace_id,
        )

    @app.exception_handler(RetrievalError)
    async def _handle_retrieval(request: Request, exc: RetrievalError) -> JSONResponse:
        status = _RETRIEVAL_STATUS.get(exc.code, 500)
        trace_id = exc.trace_id if exc.trace_id and exc.trace_id != "unknown" else _trace_id(request)
        # Internal messages may carry implementation detail; only surface them
        # for client errors (4xx), not 5xx.
        message = exc.internal_message if status < 500 else "Retrieval failed"
        return _json(
            status,
            _envelope(code=exc.code, message=message, trace_id=trace_id, details=exc.details),
            trace_id,
        )

    @app.exception_handler(ProfileEmbeddingModelImmutableError)
    async def _handle_profile_immutable(
        request: Request, exc: ProfileEmbeddingModelImmutableError
    ) -> JSONResponse:
        trace_id = _trace_id(request)
        return _json(
            409,
            _envelope(
                code="PROFILE_EMBEDDING_MODEL_IMMUTABLE",
                message=str(exc),
                trace_id=trace_id,
                details={
                    "service_name": exc.service_name,
                    "current": exc.current,
                    "requested": exc.requested,
                },
            ),
            trace_id,
        )

    @app.exception_handler(MetadataValidationError)
    async def _handle_metadata(
        request: Request, exc: MetadataValidationError
    ) -> JSONResponse:
        trace_id = _trace_id(request)
        return _json(
            422,
            _envelope(
                code="METADATA_VALIDATION_FAILED",
                message="Metadata validation failed",
                trace_id=trace_id,
                details={"invalid_fields": exc.invalid_fields},
            ),
            trace_id,
        )

    @app.exception_handler(TextExtractionError)
    async def _handle_extraction(
        request: Request, exc: TextExtractionError
    ) -> JSONResponse:
        trace_id = _trace_id(request)
        return _json(
            422,
            _envelope(code="TEXT_EXTRACTION_FAILED", message=str(exc), trace_id=trace_id),
            trace_id,
        )

    @app.exception_handler(EmptyDocumentError)
    async def _handle_empty(request: Request, exc: EmptyDocumentError) -> JSONResponse:
        trace_id = _trace_id(request)
        return _json(
            422,
            _envelope(code="EMPTY_DOCUMENT", message=str(exc), trace_id=trace_id),
            trace_id,
        )

    @app.exception_handler(GenerationServiceError)
    async def _handle_generation(
        request: Request, exc: GenerationServiceError
    ) -> JSONResponse:
        trace_id = _trace_id(request)
        return _json(
            502,
            _envelope(
                code="GENERATION_FAILED",
                message=f"Answer generation failed: {exc}",
                trace_id=trace_id,
            ),
            trace_id,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        trace_id = _trace_id(request)
        return _json(
            422,
            _envelope(
                code="REQUEST_VALIDATION_FAILED",
                message="Request validation failed",
                trace_id=trace_id,
                details={"errors": jsonable_encoder(exc.errors())},
            ),
            trace_id,
        )
