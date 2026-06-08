from fastapi import APIRouter, HTTPException, Request

from app.api.retrieval_helpers import build_default_scope, map_retrieval_error_to_http
from app.api.schemas import AskRequest
from app.services.generation_service import GenerationServiceError
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


@router.post("/ask")
def ask(request: Request, askRequest: AskRequest) -> dict:
    retrieve_request = RetrieveRequest(
        query=askRequest.query,
        retrieval_mode=RetrievalMode.DENSE,
        limit=askRequest.limit,
        scope=build_default_scope()
    )

    try:
        result = request.app.state.retrieve_use_case.execute(retrieve_request)
    except (InvalidRetrievalRequestError, UnsupportedRetrievalModeError,
            NoIndexedCorpusError, RetrievalExecutionError, RetrievedChunkValidationError) as e:
        raise map_retrieval_error_to_http(e)

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
