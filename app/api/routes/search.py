from fastapi import APIRouter, Request

from app.api.schemas import SearchRequest

router = APIRouter()


@router.post("/semantic-search")
def semantic_search_document(request: Request, searchRequest: SearchRequest) -> dict:
    return request.app.state.document_service.semantic_search(
        searchRequest.query, searchRequest.limit
    )


@router.post("/hybrid-search")
def hybrid_search_document(request: Request, searchRequest: SearchRequest) -> dict:
    return request.app.state.document_service.hybrid_search(
        searchRequest.query, searchRequest.limit
    )
