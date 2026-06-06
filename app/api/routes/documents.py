from fastapi import APIRouter, Request

from app.api.schemas import SearchRequest
from app.services.document_service import DocumentData

router = APIRouter()


@router.get("/{document_id}")
def get_document(request: Request, document_id: str) -> DocumentData:
    return request.app.state.document_service.get_document(document_id)


@router.post("/{document_id}/extract")
def extract_document(request: Request, document_id: str) -> DocumentData:
    return request.app.state.document_service.extract_text(document_id)


@router.post("/{document_id}/chunk")
def chunk_document(request: Request, document_id: str) -> dict[str, str | int]:
    return request.app.state.document_service.chunk_document(document_id)


@router.post("/{document_id}/search")
def search_document(
    request: Request, document_id: str, searchRequest: SearchRequest
) -> dict:
    return request.app.state.document_service.search_document(
        document_id, searchRequest.query, searchRequest.limit
    )


@router.post("/{document_id}/embed")
def embed_document(request: Request, document_id: str) -> dict:
    return request.app.state.document_service.embed_document(document_id)
