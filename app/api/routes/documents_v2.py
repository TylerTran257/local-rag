import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import DocumentIngestRequest, DocumentIngestResponse, DocumentUploadResponse
from app.ingest.contracts import EmptyDocumentError, IngestDocument, MetadataValidationError
from app.services.text_extractor import TextExtractor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Manual document upload with default metadata.

    Accepts a file upload, extracts text, applies default metadata
    (service_name=manual, tenant_id=local, collection=general), and
    ingests through the metadata-aware IngestUseCase.
    """
    if not file.filename:
        return JSONResponse(
            status_code=422,
            content={"detail": "Uploaded file must have a filename"},
        )

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
            service_name="manual",
            tenant_id="local",
            collection="general",
            source_type="uploaded_file",
            source_label=file.filename,
            domain_metadata={},
        )

        # Ingest through metadata-aware pipeline
        result = request.app.state.ingest_use_case.ingest_document(document)

        return DocumentUploadResponse(
            success=True,
            chunk_count=result.chunk_count,
            source_label=file.filename,
        )

    except MetadataValidationError as e:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Metadata validation failed",
                "invalid_fields": e.invalid_fields,
            },
        )
    except ValueError as e:
        # Text extraction failures and empty documents (EmptyDocumentError)
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(e),
            },
        )
    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
def ingest_document(request: Request, body: DocumentIngestRequest):
    """
    Service document ingestion with explicit metadata.

    Accepts whole document text with required metadata and ingests
    through the metadata-aware IngestUseCase.
    """
    document = IngestDocument(
        text=body.text,
        service_name=body.service_name,
        tenant_id=body.tenant_id,
        collection=body.collection,
        source_type=body.source_type,
        source_label=body.source_label,
        domain_metadata=body.domain_metadata,
    )

    try:
        result = request.app.state.ingest_use_case.ingest_document(document)
    except MetadataValidationError as e:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Metadata validation failed",
                "invalid_fields": e.invalid_fields,
            },
        )
    except EmptyDocumentError as e:
        return JSONResponse(
            status_code=422,
            content={"detail": str(e)},
        )

    return DocumentIngestResponse(chunk_count=result.chunk_count)
