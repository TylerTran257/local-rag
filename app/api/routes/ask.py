from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import AskRequest
from app.services.generation_service import GenerationServiceError

router = APIRouter()


@router.post("/ask")
def ask(request: Request, askRequest: AskRequest) -> dict:
    contexts = request.app.state.document_service.retrieve_context(
        askRequest.query, askRequest.limit
    )
    if len(contexts) == 0:
        return {
            "query": askRequest.query,
            "answer": "",
            "match_count": 0,
            "sources": [],
            "citations": [],
        }
    try:
        answer = request.app.state.generation_service.answer_question(
            askRequest.query, contexts
        )
    except GenerationServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    citations = request.app.state.document_service.serialize_citations(contexts)

    return {
        "query": askRequest.query,
        "answer": answer if len(answer) != 0 else "",
        "match_count": len(contexts),
        "sources": contexts,
        "citations": citations,
    }
