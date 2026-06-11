import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.schemas import IngestChunksRequest, IngestDocumentRequest, IngestResponse
from app.ingest.contracts import IngestChunk, IngestDocument, MetadataValidationError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest/document", response_model=IngestResponse)
def ingest_document(request: Request, body: IngestDocumentRequest):
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

    return IngestResponse(chunk_count=result.chunk_count)


@router.post("/ingest/chunks", response_model=IngestResponse)
def ingest_chunks(request: Request, body: IngestChunksRequest):
    chunks = [
        IngestChunk(
            chunk_id=item.chunk_id,
            text=item.text,
            service_name=item.service_name,
            tenant_id=item.tenant_id,
            collection=item.collection,
            source_type=item.source_type,
            source_label=item.source_label,
            domain_metadata=item.domain_metadata,
        )
        for item in body.chunks
    ]

    try:
        result = request.app.state.ingest_use_case.ingest_chunks(chunks)
    except MetadataValidationError as e:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Metadata validation failed",
                "invalid_fields": e.invalid_fields,
            },
        )

    return IngestResponse(chunk_count=result.chunk_count)
