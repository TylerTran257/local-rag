from fastapi import HTTPException

from app.retrieval import (
    RetrievalScope,
    InvalidRetrievalRequestError,
    UnsupportedRetrievalModeError,
    NoIndexedCorpusError,
    RetrievalExecutionError,
    RetrievedChunkValidationError,
)


def build_default_scope() -> RetrievalScope:
    return RetrievalScope(
        service_name="local-rag",
        tenant_id="default",
        collections=["documents"],
        filters={},
    )


def map_retrieval_error_to_http(error) -> HTTPException:
    if isinstance(error, (InvalidRetrievalRequestError, UnsupportedRetrievalModeError)):
        return HTTPException(status_code=400, detail=str(error))
    elif isinstance(error, NoIndexedCorpusError):
        return HTTPException(status_code=409, detail=str(error))
    elif isinstance(error, (RetrievalExecutionError, RetrievedChunkValidationError)):
        return HTTPException(status_code=500, detail=str(error))
    else:
        return HTTPException(status_code=500, detail="Internal server error")
