from fastapi import APIRouter, Request

from app.api.retrieval_helpers import build_default_scope, map_retrieval_error_to_http
from app.api.schemas import SearchRequest
from app.retrieval import (
    RetrieveRequest,
    RetrievalMode,
    InvalidRetrievalRequestError,
    UnsupportedRetrievalModeError,
    NoIndexedCorpusError,
    RetrievalExecutionError,
    RetrievedChunkValidationError,
)

router = APIRouter()


@router.post("/semantic-search")
def semantic_search_document(request: Request, searchRequest: SearchRequest) -> dict:
    retrieve_request = RetrieveRequest(
        query=searchRequest.query,
        retrieval_mode=RetrievalMode.DENSE,
        limit=searchRequest.limit,
        scope=build_default_scope()
    )

    try:
        result = request.app.state.retrieve_use_case.execute(retrieve_request)
    except (InvalidRetrievalRequestError, UnsupportedRetrievalModeError,
            NoIndexedCorpusError, RetrievalExecutionError, RetrievedChunkValidationError) as e:
        raise map_retrieval_error_to_http(e)

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
    retrieve_request = RetrieveRequest(
        query=searchRequest.query,
        retrieval_mode=RetrievalMode.HYBRID,
        limit=searchRequest.limit,
        scope=build_default_scope()
    )

    try:
        result = request.app.state.retrieve_use_case.execute(retrieve_request)
    except (InvalidRetrievalRequestError, UnsupportedRetrievalModeError,
            NoIndexedCorpusError, RetrievalExecutionError, RetrievedChunkValidationError) as e:
        raise map_retrieval_error_to_http(e)

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
