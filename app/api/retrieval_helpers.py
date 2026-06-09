from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app.retrieval import (
    RetrievalScope,
    RetrievalError,
    InvalidRetrievalRequestError,
    UnsupportedRetrievalModeError,
    NoIndexedCorpusError,
    RetrievalExecutionError,
    RetrievedChunkValidationError,
    RetrievalMode,
)


SAFE_RETRIEVAL_ERROR_MESSAGES = {
    "INVALID_RETRIEVAL_REQUEST": "Invalid retrieval request",
    "UNSUPPORTED_RETRIEVAL_MODE": "Unsupported retrieval mode",
    "NO_INDEXED_CORPUS": "No indexed corpus available",
    "RETRIEVAL_EXECUTION_ERROR": "Retrieval failed",
    "RETRIEVED_CHUNK_VALIDATION_ERROR": "Retrieval failed",
}


def build_default_scope() -> RetrievalScope:
    return RetrievalScope(
        service_name="local-rag",
        tenant_id="default",
        collections=["documents"],
        filters={},
    )


def _normalize_mode_value(mode: object) -> str:
    if isinstance(mode, RetrievalMode):
        return mode.value
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized.startswith("retrievalmode."):
            normalized = normalized.split(".", 1)[1]
        return normalized
    return ""


def _retrieval_error_status_code(error: RetrievalError) -> int:
    if isinstance(error, InvalidRetrievalRequestError):
        return 422
    if isinstance(error, UnsupportedRetrievalModeError):
        mode = _normalize_mode_value(error.details.get("mode"))
        if mode in {RetrievalMode.DENSE.value, RetrievalMode.LEXICAL.value, RetrievalMode.HYBRID.value}:
            return 501
        return 400
    if isinstance(error, NoIndexedCorpusError):
        return 409
    if isinstance(error, (RetrievalExecutionError, RetrievedChunkValidationError)):
        return 500
    return 500


def _retrieval_error_payload(error: RetrievalError) -> dict[str, str | None]:
    return {
        "code": error.code,
        "message": SAFE_RETRIEVAL_ERROR_MESSAGES.get(error.code, "Internal server error"),
        "trace_id": error.trace_id or None,
    }


def map_retrieval_error_to_http(error) -> HTTPException:
    if not isinstance(error, RetrievalError):
        return HTTPException(status_code=500, detail="Internal server error")

    return HTTPException(
        status_code=_retrieval_error_status_code(error),
        detail=_retrieval_error_payload(error),
    )


def map_retrieval_error_to_response(error) -> JSONResponse:
    if not isinstance(error, RetrievalError):
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Internal server error",
                "trace_id": None,
            },
        )

    return JSONResponse(
        status_code=_retrieval_error_status_code(error),
        content=_retrieval_error_payload(error),
    )
