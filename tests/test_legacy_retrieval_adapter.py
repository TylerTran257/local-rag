"""Tests for LegacyDocumentRetrievalAdapter - bridges DocumentService to Retrieval Core."""
import pytest
from fastapi import HTTPException

from app.retrieval import (
    EffectiveRetrieveRequest,
    RetrievalScope,
    RetrievalMode,
    WarningCode,
    WarningSeverity,
    NoIndexedCorpusError,
    RetrievalExecutionError,
)
from app.retrieval.legacy_adapter import LegacyDocumentRetrievalAdapter


# Stub DocumentService for testing
class StubDocumentService:
    """Stub DocumentService for testing the adapter."""

    def __init__(self):
        self.dense_results = []
        self.lexical_results = []
        self.hybrid_results = []
        self.exception_to_raise = None
        self.calls = []

    def retrieve_context_dense(self, query: str, limit: int) -> list[dict]:
        self.calls.append(("dense", query, limit))
        if self.exception_to_raise:
            raise self.exception_to_raise
        return self.dense_results

    def retrieve_context_lexical(self, query: str, limit: int) -> list[dict]:
        self.calls.append(("lexical", query, limit))
        if self.exception_to_raise:
            raise self.exception_to_raise
        return self.lexical_results

    def retrieve_context_hybrid(self, query: str, limit: int) -> list[dict]:
        self.calls.append(("hybrid", query, limit))
        if self.exception_to_raise:
            raise self.exception_to_raise
        return self.hybrid_results


@pytest.fixture
def stub_document_service():
    return StubDocumentService()


@pytest.fixture
def adapter(stub_document_service):
    return LegacyDocumentRetrievalAdapter(document_service=stub_document_service)


@pytest.fixture
def effective_request():
    scope = RetrievalScope(
        service_name="local-rag",
        tenant_id="default",
        collections=["documents"],
        filters={}
    )
    return EffectiveRetrieveRequest(
        normalized_query="test query",
        original_query="test query",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        validated_scope=scope,
        correlation_id=None
    )


# Test: Constructor
def test_adapter_accepts_document_service(stub_document_service):
    adapter = LegacyDocumentRetrievalAdapter(document_service=stub_document_service)
    assert adapter is not None


# Test: Mode dispatching
def test_dense_mode_calls_retrieve_context_dense(adapter, stub_document_service, effective_request):
    stub_document_service.dense_results = [
        {
            "document_id": "doc-1",
            "original_filename": "test.pdf",
            "chunk_index": 0,
            "score": 0.95,
            "text": "test content"
        }
    ]

    result = adapter.retrieve(effective_request)

    assert len(stub_document_service.calls) == 1
    assert stub_document_service.calls[0][0] == "dense"
    assert stub_document_service.calls[0][1] == "test query"
    assert stub_document_service.calls[0][2] == 5


def test_lexical_mode_calls_retrieve_context_lexical(adapter, stub_document_service, effective_request):
    effective_request.retrieval_mode = RetrievalMode.LEXICAL
    stub_document_service.lexical_results = [
        {
            "document_id": "doc-1",
            "original_filename": "test.pdf",
            "chunk_index": 0,
            "score": 0.95,
            "text": "test content"
        }
    ]

    result = adapter.retrieve(effective_request)

    assert len(stub_document_service.calls) == 1
    assert stub_document_service.calls[0][0] == "lexical"
    assert result.chunks[0].retrieval_mode == RetrievalMode.LEXICAL
    assert result.chunks[0].rank == 1


def test_hybrid_mode_calls_retrieve_context_hybrid(adapter, stub_document_service, effective_request):
    effective_request.retrieval_mode = RetrievalMode.HYBRID
    stub_document_service.hybrid_results = [
        {
            "document_id": "doc-1",
            "original_filename": "test.pdf",
            "chunk_index": 0,
            "score": 0.95,
            "text": "test content"
        }
    ]

    result = adapter.retrieve(effective_request)

    assert len(stub_document_service.calls) == 1
    assert stub_document_service.calls[0][0] == "hybrid"
    assert result.chunks[0].retrieval_mode == RetrievalMode.HYBRID
    assert result.chunks[0].rank == 1


# Test: Result normalization to RetrievedChunk with sentinel defaults
def test_maps_document_service_result_to_retrieved_chunk_with_sentinel_defaults(adapter, stub_document_service, effective_request):
    stub_document_service.dense_results = [
        {
            "document_id": "doc-123",
            "original_filename": "example.pdf",
            "chunk_index": 5,
            "score": 0.87,
            "text": "chunk content here"
        }
    ]

    result = adapter.retrieve(effective_request)

    assert len(result.chunks) == 1
    chunk = result.chunks[0]

    # Check content and score mapping
    assert chunk.chunk_id == "doc-123:5"
    assert chunk.document_id == "doc-123"
    assert chunk.content == "chunk content here"
    assert chunk.score == 0.87
    assert chunk.rank == 1
    assert chunk.retrieval_mode == RetrievalMode.DENSE

    # Check metadata fields from DocumentService
    assert chunk.metadata["document_id"] == "doc-123"
    assert chunk.metadata["chunk_index"] == 5

    # Check sentinel defaults (from ADR 003)
    assert chunk.metadata["service_name"] == "local-rag"
    assert chunk.metadata["tenant_id"] == "default"
    assert chunk.metadata["collection"] == "documents"
    assert chunk.metadata["source_type"] == "document"
    assert chunk.metadata["source_label"] == "example.pdf"  # original_filename


def test_returns_multiple_chunks_with_sentinel_defaults(adapter, stub_document_service, effective_request):
    stub_document_service.dense_results = [
        {
            "document_id": "doc-1",
            "original_filename": "file1.pdf",
            "chunk_index": 0,
            "score": 0.95,
            "text": "first chunk"
        },
        {
            "document_id": "doc-2",
            "original_filename": "file2.pdf",
            "chunk_index": 1,
            "score": 0.85,
            "text": "second chunk"
        }
    ]

    result = adapter.retrieve(effective_request)

    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_id == "doc-1:0"
    assert result.chunks[0].rank == 1
    assert result.chunks[1].chunk_id == "doc-2:1"
    assert result.chunks[1].rank == 2
    assert result.chunks[0].metadata["service_name"] == "local-rag"
    assert result.chunks[1].metadata["service_name"] == "local-rag"


# Test: LEGACY_METADATA_DEFAULTED warning emission
def test_emits_single_legacy_metadata_defaulted_warning_with_chunk_count(adapter, stub_document_service, effective_request):
    stub_document_service.dense_results = [
        {
            "document_id": "doc-1",
            "original_filename": "test.pdf",
            "chunk_index": 0,
            "score": 0.95,
            "text": "chunk 1"
        },
        {
            "document_id": "doc-2",
            "original_filename": "test2.pdf",
            "chunk_index": 0,
            "score": 0.85,
            "text": "chunk 2"
        }
    ]

    result = adapter.retrieve(effective_request)

    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert warning.code == WarningCode.LEGACY_METADATA_DEFAULTED
    assert warning.severity == WarningSeverity.MEDIUM
    assert warning.source == "LegacyDocumentRetrievalAdapter"
    assert "2" in warning.message
    assert warning.details["chunk_count"] == 2
    assert warning.details["service_name"] == "local-rag"
    assert warning.details["tenant_id"] == "default"
    assert warning.details["collection"] == "documents"
    assert warning.details["source_type"] == "document"


# Test: Exception translation
def test_http_exception_409_translates_to_no_indexed_corpus_error(adapter, stub_document_service, effective_request):
    stub_document_service.exception_to_raise = HTTPException(
        status_code=409,
        detail="At least one document must be embedded before semantic searching"
    )

    with pytest.raises(NoIndexedCorpusError) as exc_info:
        adapter.retrieve(effective_request)

    error = exc_info.value
    assert error.code == "NO_INDEXED_CORPUS"
    assert "409" in str(error.details) or "embedded" in error.internal_message.lower()


def test_http_exception_other_status_translates_to_retrieval_execution_error(adapter, stub_document_service, effective_request):
    stub_document_service.exception_to_raise = HTTPException(
        status_code=500,
        detail="Internal server error"
    )

    with pytest.raises(RetrievalExecutionError) as exc_info:
        adapter.retrieve(effective_request)

    error = exc_info.value
    assert error.code == "RETRIEVAL_EXECUTION_ERROR"


def test_unexpected_exception_translates_to_retrieval_execution_error(adapter, stub_document_service, effective_request):
    stub_document_service.exception_to_raise = RuntimeError("Unexpected error")

    with pytest.raises(RetrievalExecutionError) as exc_info:
        adapter.retrieve(effective_request)

    error = exc_info.value
    assert error.code == "RETRIEVAL_EXECUTION_ERROR"
    assert "Unexpected error" in error.internal_message or "RuntimeError" in str(error.details)


# Test: Empty diagnostics
def test_returns_empty_diagnostics(adapter, stub_document_service, effective_request):
    stub_document_service.dense_results = [
        {
            "document_id": "doc-1",
            "original_filename": "test.pdf",
            "chunk_index": 0,
            "score": 0.95,
            "text": "test"
        }
    ]

    result = adapter.retrieve(effective_request)

    assert result.diagnostics == {}


# Test: Empty results
def test_handles_empty_results_from_document_service(adapter, stub_document_service, effective_request):
    stub_document_service.dense_results = []

    result = adapter.retrieve(effective_request)

    assert len(result.chunks) == 0
    assert len(result.warnings) == 0  # No warnings if no chunks
    assert result.diagnostics == {}
