from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.services.document_service import DocumentData

router = APIRouter()


@router.post("/upload_v1")
async def upload_document_v1(
    request: Request, file: UploadFile = File(...)
) -> DocumentData:
    return await request.app.state.document_service.create_document(file)


@router.post("/upload_v2")
async def upload_document_v2(
    request: Request, file: UploadFile = File(...)
) -> DocumentData:
    uploaded_file = await request.app.state.document_service.create_document(file)
    document_id = uploaded_file.get("document_id")
    if document_id is None:
        raise HTTPException(status_code=500, detail="Document Id Not Found")

    request.app.state.document_service.extract_text(document_id)
    request.app.state.document_service.chunk_document(document_id)
    request.app.state.document_service.embed_document(document_id)

    return request.app.state.document_service.get_document(document_id)
