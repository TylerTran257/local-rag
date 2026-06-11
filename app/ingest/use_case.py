import logging
from dataclasses import dataclass, field
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.db.models import DocumentChunk
from app.ingest.contracts import (
    EmptyDocumentError,
    IngestChunk,
    IngestDocument,
    MetadataValidator,
    ValidatedMetadata,
)
from app.services.embedding_service import EmbeddingService
from app.services.lexical_search_service import LexicalSearchService
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

# Internal chunking defaults (not exposed through the public API)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)


@dataclass
class IngestResult:
    """Result of an ingest operation."""

    chunk_count: int
    warnings: list[str] = field(default_factory=list)


class IngestUseCase:
    """
    Metadata-aware ingestion pipeline.

    Accepts IngestDocument or IngestChunk inputs, validates metadata,
    chunks (if needed), embeds, and stores in both vector and lexical
    backends with metadata.

    This is the single ingestion path for the service: both manual uploads
    and service document ingestion flow through it.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        lexical_search_service: LexicalSearchService,
    ):
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.lexical_search_service = lexical_search_service

    def ingest_document(self, document: IngestDocument) -> IngestResult:
        """
        Ingest a document with metadata.

        Validates metadata, chunks text, embeds chunks, and stores in both
        vector and lexical backends with metadata.

        Args:
            document: Document with text and required metadata

        Returns:
            IngestResult with chunk count

        Raises:
            MetadataValidationError: If metadata validation fails
            EmptyDocumentError: If the document text is empty or whitespace-only
        """
        # Validate metadata first (fail fast)
        validated_metadata = self._validate_and_prepare_metadata(document)

        # Reject empty documents before any expensive processing
        if not document.text or not document.text.strip():
            logger.info(
                "event=ingest_document_rejected service_name=%s tenant_id=%s collection=%s reason=empty_text",
                document.service_name,
                document.tenant_id,
                document.collection,
            )
            raise EmptyDocumentError(source_label=document.source_label)

        chunk_texts = text_splitter.split_text(document.text)

        if not chunk_texts:
            logger.info(
                "event=ingest_document_rejected service_name=%s tenant_id=%s collection=%s reason=no_chunks_produced",
                document.service_name,
                document.tenant_id,
                document.collection,
            )
            raise EmptyDocumentError(source_label=document.source_label)

        # Generate document ID for this ingest operation
        document_id = str(uuid4())

        # Create DocumentChunk objects
        chunks = [
            DocumentChunk(
                id=f"{document_id}_{idx}",
                chunk_index=idx,
                text=text,
            )
            for idx, text in enumerate(chunk_texts)
        ]

        # Embed, store, and return result
        return self._embed_and_store(
            document_id=document_id,
            original_filename=document.source_label,
            chunks=chunks,
            metadata_dict=self._metadata_to_dict(validated_metadata),
        )

    def ingest_chunks(self, chunks: list[IngestChunk]) -> IngestResult:
        """
        Ingest pre-chunked content with metadata.

        Validates metadata, embeds chunks, and stores in both backends.

        Args:
            chunks: List of pre-chunked content with metadata

        Returns:
            IngestResult with chunk count

        Raises:
            MetadataValidationError: If metadata validation fails
        """
        if not chunks:
            logger.info("event=ingest_chunks_completed chunk_count=0 reason=empty_input")
            return IngestResult(chunk_count=0)

        # Validate every chunk's metadata before any indexing (fail fast),
        # and keep metadata per chunk - chunks in a batch may differ.
        metadata_dicts = [
            self._metadata_to_dict(self._validate_and_prepare_metadata(chunk))
            for chunk in chunks
        ]

        # Generate document ID for this batch
        document_id = str(uuid4())

        # Convert to DocumentChunk objects
        document_chunks = [
            DocumentChunk(
                id=chunk.chunk_id,
                chunk_index=idx,
                text=chunk.text,
            )
            for idx, chunk in enumerate(chunks)
        ]

        # Embed, store, and return result
        return self._embed_and_store(
            document_id=document_id,
            original_filename=chunks[0].source_label,
            chunks=document_chunks,
            metadata_dict=metadata_dicts,
        )

    def _validate_and_prepare_metadata(
        self, item: IngestDocument | IngestChunk
    ) -> ValidatedMetadata:
        """Validate metadata and return validated result."""
        metadata_dict = {
            "service_name": item.service_name,
            "tenant_id": item.tenant_id,
            "collection": item.collection,
            "source_type": item.source_type,
            "source_label": item.source_label,
        }

        # Add domain metadata if present
        if item.domain_metadata:
            metadata_dict.update(item.domain_metadata)

        return MetadataValidator.validate(metadata_dict)

    def _metadata_to_dict(self, validated: ValidatedMetadata) -> dict:
        """Convert ValidatedMetadata to dict for storage."""
        metadata_dict = {
            "service_name": validated.service_name,
            "tenant_id": validated.tenant_id,
            "collection": validated.collection,
            "source_type": validated.source_type,
            "source_label": validated.source_label,
        }

        # Include domain metadata if present
        if validated.domain_metadata:
            metadata_dict.update(validated.domain_metadata)

        return metadata_dict

    def _embed_and_store(
        self,
        document_id: str,
        original_filename: str,
        chunks: list[DocumentChunk],
        metadata_dict: dict | list[dict],
    ) -> IngestResult:
        """Embed chunks and store in both backends with metadata.

        metadata_dict may be a single dict (applies to all chunks) or a list
        of per-chunk dicts matching the chunk count.
        """
        # Embed all chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        # Store in vector backend with metadata
        self.vector_store_service.upsert_document_chunks(
            document_id=document_id,
            original_filename=original_filename,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata_dict,
        )

        # Store in lexical backend with metadata
        self.lexical_search_service.index_document_chunks(
            document_id=document_id,
            original_filename=original_filename,
            chunks=chunks,
            metadata=metadata_dict,
        )

        first_metadata = metadata_dict[0] if isinstance(metadata_dict, list) else metadata_dict
        logger.info(
            "event=ingest_completed service_name=%s tenant_id=%s collection=%s chunk_count=%s",
            first_metadata["service_name"],
            first_metadata["tenant_id"],
            first_metadata["collection"],
            len(chunks),
        )

        return IngestResult(chunk_count=len(chunks))
