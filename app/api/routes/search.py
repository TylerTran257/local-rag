from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import SearchRequest
from app.retrieval import (
    RetrieveRequest,
    RetrievalScope,
    RetrievalMode,
    InvalidRetrievalRequestError,
    UnsupportedRetrievalModeError,
    NoIndexedCorpusError,
    RetrievalExecutionError,
    RetrievedChunkValidationError,
)

router = APIRouter()


def _map_retrieval_error_to_http(error) -> HTTPException:
    """Map domain errors to HTTP status codes."""
    if isinstance(error, (InvalidRetrievalRequestError, UnsupportedRetrievalModeError)):
        return HTTPException(status_code=400, detail=str(error))
    elif isinstance(error, NoIndexedCorpusError):
        return HTTPException(status_code=409, detail=str(error))
    elif isinstance(error, (RetrievalExecutionError, RetrievedChunkValidationError)):
        return HTTPException(status_code=500, detail=str(error))
    else:
        return HTTPException(status_code=500, detail="Internal server error")


@router.post("/semantic-search")
def semantic_search_document(request: Request, searchRequest: SearchRequest) -> dict:
    # Construct RetrieveRequest with sentinel scope
    scope = RetrievalScope(
        service_name="local-rag",
        tenant_id="default",
        collections=["documents"],
        filters={}
    )
    retrieve_request = RetrieveRequest(
        query=searchRequest.query,
        retrieval_mode=RetrievalMode.DENSE,
        limit=searchRequest.limit,
        scope=scope
    )

    try:
        result = request.app.state.retrieve_use_case.execute(retrieve_request)
    except (InvalidRetrievalRequestError, UnsupportedRetrievalModeError,
            NoIndexedCorpusError, RetrievalExecutionError, RetrievedChunkValidationError) as e:
        raise _map_retrieval_error_to_http(e)

    # Map RetrievedChunk back to legacy response format
    results = [
        {
            "document_id": chunk.metadata["document_id"],
            "original_filename": chunk.metadata["source_label"],
            "chunk_index": chunk.metadata["chunk_index"],
            "score": chunk.score,
            "text": chunk.content
        }
        for chunk in result.chunks
    ]

    return {
        "query": searchRequest.query,
        "match_count": len(results),
        "results": results
    }


@router.post("/hybrid-search")
def hybrid_search_document(request: Request, searchRequest: SearchRequest) -> dict:
    # Construct RetrieveRequest with sentinel scope
    scope = RetrievalScope(
        service_name="local-rag",
        tenant_id="default",
        collections=["documents"],
        filters={}
    )
    retrieve_request = RetrieveRequest(
        query=searchRequest.query,
        retrieval_mode=RetrievalMode.HYBRID,
        limit=searchRequest.limit,
        scope=scope
    )

    try:
        result = request.app.state.retrieve_use_case.execute(retrieve_request)
    except (InvalidRetrievalRequestError, UnsupportedRetrievalModeError,
            NoIndexedCorpusError, RetrievalExecutionError, RetrievedChunkValidationError) as e:
        raise _map_retrieval_error_to_http(e)

    # Map RetrievedChunk back to legacy response format
    results = [
        {
            "document_id": chunk.metadata["document_id"],
            "original_filename": chunk.metadata["source_label"],
            "chunk_index": chunk.metadata["chunk_index"],
            "score": chunk.score,
            "text": chunk.content
        }
        for chunk in result.chunks
    ]

    return {
        "query": searchRequest.query,
        "match_count": len(results),
        "results": results
    }
