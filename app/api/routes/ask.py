from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import AskRequest
from app.services.generation_service import GenerationServiceError
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


@router.post("/ask")
def ask(request: Request, askRequest: AskRequest) -> dict:
    # Construct RetrieveRequest with sentinel scope
    scope = RetrievalScope(
        service_name="local-rag",
        tenant_id="default",
        collections=["documents"],
        filters={}
    )
    retrieve_request = RetrieveRequest(
        query=askRequest.query,
        retrieval_mode=RetrievalMode.DENSE,
        limit=askRequest.limit,
        scope=scope
    )

    try:
        result = request.app.state.retrieve_use_case.execute(retrieve_request)
    except (InvalidRetrievalRequestError, UnsupportedRetrievalModeError,
            NoIndexedCorpusError, RetrievalExecutionError, RetrievedChunkValidationError) as e:
        raise _map_retrieval_error_to_http(e)

    if len(result.chunks) == 0:
        return {
            "query": askRequest.query,
            "answer": "",
            "match_count": 0,
            "sources": [],
            "citations": [],
        }

    # Map RetrievedChunk to context format expected by GenerationService
    contexts = [
        {
            "document_id": chunk.metadata["document_id"],
            "original_filename": chunk.metadata["source_label"],
            "chunk_index": chunk.metadata["chunk_index"],
            "score": chunk.score,
            "text": chunk.content
        }
        for chunk in result.chunks
    ]

    try:
        answer = request.app.state.generation_service.answer_question(
            askRequest.query, contexts
        )
    except GenerationServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Build citations from contexts
    citations = [
        {
            "id": i,
            "document_id": ctx["document_id"],
            "original_filename": ctx["original_filename"],
            "chunk_index": ctx["chunk_index"],
            "score": ctx["score"],
            "text": ctx["text"],
        }
        for i, ctx in enumerate(contexts, start=1)
    ]

    return {
        "query": askRequest.query,
        "answer": answer if len(answer) != 0 else "",
        "match_count": len(contexts),
        "sources": contexts,
        "citations": citations,
    }
