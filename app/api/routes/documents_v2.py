import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.api.schemas import DocumentIngestRequest, DocumentIngestResponse, DocumentUploadResponse
from app.auth import Principal, enforce_scope, require_principal
from app.ingest.contracts import EmptyDocumentError, IngestDocument
from app.services.text_extractor import TextExtractor

logger = logging.getLogger(__name__)

router = APIRouter()

# Fixed scope applied to manual uploads.
_MANUAL_SERVICE = "manual"
_MANUAL_TENANT = "local"
_MANUAL_COLLECTION = "general"


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(require_principal),
):
    """
    Manual document upload with default metadata.

    Accepts a file upload, extracts text, applies default manual metadata, and
    ingests through the metadata-aware IngestUseCase. The caller's key must be
    granted the fixed manual scope.
    """
    enforce_scope(
        principal,
        service_name=_MANUAL_SERVICE,
        tenant_id=_MANUAL_TENANT,
        collections=[_MANUAL_COLLECTION],
    )

    if not file.filename:
        raise EmptyDocumentError(source_label="<uploaded file>")

    # Save uploaded file to temp location for text extraction
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)

    try:
        # Extract text
        text_extractor = TextExtractor()
        extracted_text = text_extractor.extract(tmp_path)

        # Apply default manual metadata
        document = IngestDocument(
            text=extracted_text,
            service_name=_MANUAL_SERVICE,
            tenant_id=_MANUAL_TENANT,
            collection=_MANUAL_COLLECTION,
            source_type="uploaded_file",
            source_label=file.filename,
            domain_metadata={},
        )

        # Ingest through metadata-aware pipeline (MetadataValidationError /
        # EmptyDocumentError propagate to the exception handlers).
        result = request.app.state.ingest_use_case.ingest_document(document)

        return DocumentUploadResponse(
            success=True,
            chunk_count=result.chunk_count,
            source_label=file.filename,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
def ingest_document(
    request: Request,
    body: DocumentIngestRequest,
    principal: Principal = Depends(require_principal),
):
    """
    Service document ingestion with explicit metadata.

    Accepts whole document text with required metadata and ingests through the
    metadata-aware IngestUseCase. The caller's key must be granted the request
    scope.
    """
    enforce_scope(
        principal,
        service_name=body.service_name,
        tenant_id=body.tenant_id,
        collections=[body.collection],
    )

    document = IngestDocument(
        text=body.text,
        service_name=body.service_name,
        tenant_id=body.tenant_id,
        collection=body.collection,
        source_type=body.source_type,
        source_label=body.source_label,
        domain_metadata=body.domain_metadata,
    )

    # MetadataValidationError / EmptyDocumentError propagate to the handlers.
    result = request.app.state.ingest_use_case.ingest_document(document)

    return DocumentIngestResponse(chunk_count=result.chunk_count)
