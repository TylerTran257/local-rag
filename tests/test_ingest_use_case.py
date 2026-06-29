import pytest
from unittest.mock import Mock, call

from app.db.models import DocumentChunk
from app.ingest.contracts import EmptyDocumentError, IngestDocument, IngestChunk, MetadataValidationError
from app.ingest.use_case import IngestUseCase, IngestResult


@pytest.fixture
def mock_embedding_service():
    service = Mock()
    service.embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return service


@pytest.fixture
def mock_vector_store():
    return Mock()


@pytest.fixture
def mock_lexical_search():
    return Mock()


@pytest.fixture
def ingest_use_case(mock_embedding_service, mock_vector_store, mock_lexical_search):
    return IngestUseCase(
        embedding_service=mock_embedding_service,
        vector_store_service=mock_vector_store,
        lexical_search_service=mock_lexical_search,
    )


@pytest.fixture
def valid_metadata():
    return {
        "service_name": "test-service",
        "tenant_id": "tenant-123",
        "collection": "documents",
        "source_type": "pdf",
        "source_label": "test.pdf",
    }


class TestIngestDocument:
    """Tests for document ingestion (text → chunks → embed → store)."""

    def test_ingest_document_end_to_end(
        self, ingest_use_case, mock_embedding_service, mock_vector_store, mock_lexical_search, valid_metadata
    ):
        """Tracer bullet: ingest document validates, chunks, embeds, and stores."""
        document = IngestDocument(
            text="This is a test document. " * 100,  # Long enough to create multiple chunks
            **valid_metadata,
        )

        result = ingest_use_case.ingest_document(document)

        # Should return chunk count
        assert isinstance(result, IngestResult)
        assert result.chunk_count > 0

        # Should have embedded the chunks
        assert mock_embedding_service.embed_texts.called
        embedded_texts = mock_embedding_service.embed_texts.call_args[0][0]
        assert len(embedded_texts) > 0

        # Should have stored in vector store with metadata
        assert mock_vector_store.upsert_document_chunks.called
        vector_call = mock_vector_store.upsert_document_chunks.call_args
        assert vector_call.kwargs["metadata"] == valid_metadata

        # Should have stored in lexical search with metadata
        assert mock_lexical_search.index_document_chunks.called
        lexical_call = mock_lexical_search.index_document_chunks.call_args
        assert lexical_call.kwargs["metadata"] == valid_metadata

    def test_ingest_document_with_short_text(
        self, ingest_use_case, mock_embedding_service, mock_vector_store, mock_lexical_search, valid_metadata
    ):
        """Short document creates single chunk."""
        document = IngestDocument(
            text="Short text.",
            **valid_metadata,
        )

        result = ingest_use_case.ingest_document(document)

        assert result.chunk_count == 1
        assert mock_embedding_service.embed_texts.called
        assert mock_vector_store.upsert_document_chunks.called
        assert mock_lexical_search.index_document_chunks.called

    def test_ingest_document_with_empty_text_is_rejected(
        self, ingest_use_case, mock_embedding_service, mock_vector_store, mock_lexical_search, valid_metadata
    ):
        """Empty document raises EmptyDocumentError before any indexing."""
        document = IngestDocument(
            text="",
            **valid_metadata,
        )

        with pytest.raises(EmptyDocumentError):
            ingest_use_case.ingest_document(document)

        assert not mock_embedding_service.embed_texts.called
        assert not mock_vector_store.upsert_document_chunks.called
        assert not mock_lexical_search.index_document_chunks.called

    def test_ingest_document_with_whitespace_only_text_is_rejected(
        self, ingest_use_case, mock_embedding_service, valid_metadata
    ):
        """Whitespace-only document raises EmptyDocumentError."""
        document = IngestDocument(
            text="   \n\t  ",
            **valid_metadata,
        )

        with pytest.raises(EmptyDocumentError):
            ingest_use_case.ingest_document(document)

        assert not mock_embedding_service.embed_texts.called

    def test_ingest_document_validates_metadata(self, ingest_use_case):
        """Invalid metadata raises MetadataValidationError before indexing."""
        document = IngestDocument(
            text="Test content",
            service_name="",  # Invalid: empty string
            tenant_id="tenant-1",
            collection="docs",
            source_type="pdf",
            source_label="test.pdf",
        )

        with pytest.raises(MetadataValidationError) as exc_info:
            ingest_use_case.ingest_document(document)

        assert "service_name" in exc_info.value.invalid_fields

    def test_ingest_document_with_domain_metadata(
        self, ingest_use_case, mock_vector_store, mock_lexical_search, valid_metadata
    ):
        """Domain metadata is passed through to storage."""
        document = IngestDocument(
            text="Test content",
            domain_metadata={"author": "John Doe", "category": "research"},
            **valid_metadata,
        )

        result = ingest_use_case.ingest_document(document)

        # Metadata should include domain_metadata
        vector_call = mock_vector_store.upsert_document_chunks.call_args
        stored_metadata = vector_call.kwargs["metadata"]
        assert stored_metadata["service_name"] == "test-service"
        assert stored_metadata["author"] == "John Doe"
        assert stored_metadata["category"] == "research"
        assert "domain_metadata" not in stored_metadata


class TestIngestChunks:
    """Tests for pre-chunked ingestion (chunks → embed → store)."""

    def test_ingest_chunks_end_to_end(
        self, ingest_use_case, mock_embedding_service, mock_vector_store, mock_lexical_search, valid_metadata
    ):
        """Pre-chunked ingestion validates, embeds, and stores."""
        chunks = [
            IngestChunk(
                chunk_id="chunk-1",
                text="First chunk text",
                **valid_metadata,
            ),
            IngestChunk(
                chunk_id="chunk-2",
                text="Second chunk text",
                **valid_metadata,
            ),
        ]

        result = ingest_use_case.ingest_chunks(chunks)

        assert result.chunk_count == 2

        # Should embed both chunks
        assert mock_embedding_service.embed_texts.called
        embedded_texts = mock_embedding_service.embed_texts.call_args[0][0]
        assert len(embedded_texts) == 2

        # Should store in both backends
        assert mock_vector_store.upsert_document_chunks.called
        assert mock_lexical_search.index_document_chunks.called

    def test_ingest_chunks_validates_metadata(self, ingest_use_case):
        """Invalid metadata in chunks raises validation error."""
        chunks = [
            IngestChunk(
                chunk_id="chunk-1",
                text="Test",
                service_name="service-1",
                tenant_id="",  # Invalid
                collection="docs",
                source_type="pdf",
                source_label="test.pdf",
            )
        ]

        with pytest.raises(MetadataValidationError):
            ingest_use_case.ingest_chunks(chunks)

    def test_ingest_chunks_validates_every_chunk(
        self, ingest_use_case, mock_vector_store, mock_lexical_search, valid_metadata
    ):
        """An invalid chunk anywhere in the batch fails before any indexing."""
        chunks = [
            IngestChunk(chunk_id="chunk-1", text="Valid", **valid_metadata),
            IngestChunk(
                chunk_id="chunk-2",
                text="Invalid",
                service_name="service-1",
                tenant_id="",  # Invalid, but not the first chunk
                collection="docs",
                source_type="pdf",
                source_label="test.pdf",
            ),
        ]

        with pytest.raises(MetadataValidationError):
            ingest_use_case.ingest_chunks(chunks)

        assert not mock_vector_store.upsert_document_chunks.called
        assert not mock_lexical_search.index_document_chunks.called

    def test_ingest_chunks_preserves_per_chunk_metadata(
        self, ingest_use_case, mock_vector_store, mock_lexical_search, valid_metadata
    ):
        """Chunks with differing metadata keep their own metadata (no broadcast)."""
        chunks = [
            IngestChunk(
                chunk_id="chunk-a",
                text="Tenant A content",
                service_name="svc",
                tenant_id="tenant-a",
                collection="docs",
                source_type="pdf",
                source_label="a.pdf",
            ),
            IngestChunk(
                chunk_id="chunk-b",
                text="Tenant B content",
                service_name="svc",
                tenant_id="tenant-b",
                collection="docs",
                source_type="pdf",
                source_label="b.pdf",
            ),
        ]

        result = ingest_use_case.ingest_chunks(chunks)

        assert result.chunk_count == 2

        stored_vector_metadata = mock_vector_store.upsert_document_chunks.call_args.kwargs["metadata"]
        stored_lexical_metadata = mock_lexical_search.index_document_chunks.call_args.kwargs["metadata"]

        for stored in (stored_vector_metadata, stored_lexical_metadata):
            assert isinstance(stored, list)
            assert stored[0]["tenant_id"] == "tenant-a"
            assert stored[1]["tenant_id"] == "tenant-b"
            assert stored[0]["source_label"] == "a.pdf"
            assert stored[1]["source_label"] == "b.pdf"

    def test_ingest_empty_chunk_list(
        self, ingest_use_case, mock_embedding_service, mock_vector_store, mock_lexical_search
    ):
        """Empty chunk list produces zero chunks."""
        result = ingest_use_case.ingest_chunks([])

        assert result.chunk_count == 0
        assert not mock_embedding_service.embed_texts.called
        assert not mock_vector_store.upsert_document_chunks.called
        assert not mock_lexical_search.index_document_chunks.called


class TestMetadataPropagation:
    """Tests verifying metadata flows correctly to both backends."""

    def test_metadata_passed_to_vector_store(
        self, ingest_use_case, mock_vector_store, valid_metadata
    ):
        """Verify vector store receives correct metadata."""
        document = IngestDocument(
            text="Test content",
            **valid_metadata,
        )

        ingest_use_case.ingest_document(document)

        call_kwargs = mock_vector_store.upsert_document_chunks.call_args.kwargs
        metadata = call_kwargs["metadata"]

        assert metadata["service_name"] == "test-service"
        assert metadata["tenant_id"] == "tenant-123"
        assert metadata["collection"] == "documents"
        assert metadata["source_type"] == "pdf"
        assert metadata["source_label"] == "test.pdf"

    def test_metadata_passed_to_lexical_search(
        self, ingest_use_case, mock_lexical_search, valid_metadata
    ):
        """Verify lexical search receives correct metadata."""
        document = IngestDocument(
            text="Test content",
            **valid_metadata,
        )

        ingest_use_case.ingest_document(document)

        call_kwargs = mock_lexical_search.index_document_chunks.call_args.kwargs
        metadata = call_kwargs["metadata"]

        assert metadata["service_name"] == "test-service"
        assert metadata["tenant_id"] == "tenant-123"
        assert metadata["collection"] == "documents"
        assert metadata["source_type"] == "pdf"
        assert metadata["source_label"] == "test.pdf"

    def test_consistent_metadata_across_backends(
        self, ingest_use_case, mock_vector_store, mock_lexical_search, valid_metadata
    ):
        """Both backends receive identical metadata."""
        document = IngestDocument(
            text="Test content",
            **valid_metadata,
        )

        ingest_use_case.ingest_document(document)

        vector_metadata = mock_vector_store.upsert_document_chunks.call_args.kwargs["metadata"]
        lexical_metadata = mock_lexical_search.index_document_chunks.call_args.kwargs["metadata"]

        assert vector_metadata == lexical_metadata
