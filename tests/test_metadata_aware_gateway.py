import pytest
from unittest.mock import Mock

from app.retrieval.contracts import RetrievalGateway
from app.retrieval.errors import NoIndexedCorpusError, RetrievalExecutionError
from app.retrieval.metadata_gateway import MetadataAwareRetrievalGateway
from app.retrieval.types import (
    EffectiveRetrieveRequest,
    RetrievalMode,
    RetrievalScope,
    RetrievalGatewayResult,
    RetrievedChunk,
)


@pytest.fixture
def mock_vector_store():
    service = Mock()
    service.search.return_value = [
        {
            "document_id": "doc-1",
            "original_filename": "test.pdf",
            "chunk_index": 0,
            "service_name": "test-service",
            "tenant_id": "tenant-1",
            "collection": "docs",
            "source_type": "pdf",
            "source_label": "test.pdf",
            "style_category": "voice_rules",
            "platform": "twitter",
            "text": "Dense result 1",
            "score": 0.95,
        },
        {
            "document_id": "doc-1",
            "original_filename": "test.pdf",
            "chunk_index": 1,
            "service_name": "test-service",
            "tenant_id": "tenant-1",
            "collection": "docs",
            "source_type": "pdf",
            "source_label": "test.pdf",
            "style_category": "voice_rules",
            "platform": "twitter",
            "text": "Dense result 2",
            "score": 0.85,
        },
    ]
    return service


@pytest.fixture
def mock_lexical_search():
    service = Mock()
    service.search.return_value = [
        {
            "document_id": "doc-1",
            "original_filename": "test.pdf",
            "chunk_index": 0,
            "service_name": "test-service",
            "tenant_id": "tenant-1",
            "collection": "docs",
            "source_type": "pdf",
            "source_label": "test.pdf",
            "style_category": "voice_rules",
            "platform": "twitter",
            "text": "Lexical result 1",
            "score": -0.5,  # BM25 score
        },
        {
            "document_id": "doc-2",
            "original_filename": "other.pdf",
            "chunk_index": 0,
            "service_name": "test-service",
            "tenant_id": "tenant-1",
            "collection": "docs",
            "source_type": "pdf",
            "source_label": "other.pdf",
            "style_category": "voice_rules",
            "platform": "twitter",
            "text": "Lexical result 2",
            "score": -1.2,
        },
    ]
    return service


@pytest.fixture
def mock_embedding_service():
    service = Mock()
    service.embed_text.return_value = [0.1, 0.2, 0.3]
    return service


@pytest.fixture
def gateway(mock_vector_store, mock_lexical_search, mock_embedding_service):
    return MetadataAwareRetrievalGateway(
        vector_store_service=mock_vector_store,
        lexical_search_service=mock_lexical_search,
        embedding_service=mock_embedding_service,
    )


@pytest.fixture
def effective_request():
    return EffectiveRetrieveRequest(
        normalized_query="test query",
        original_query="test query",
        retrieval_mode=RetrievalMode.DENSE,
        limit=10,
        validated_scope=RetrievalScope(
            service_name="test-service",
            tenant_id="tenant-1",
            collections=["docs"],
            filters={"source_type": "pdf"},
        ),
    )


class TestProtocolCompliance:
    """Test that MetadataAwareRetrievalGateway implements RetrievalGateway protocol."""

    def test_has_retrieve_method(self, gateway):
        """Gateway has retrieve method with correct signature."""
        assert hasattr(gateway, "retrieve")
        assert callable(gateway.retrieve)


class TestDenseRetrieval:
    """Tests for dense retrieval mode."""

    def test_dense_retrieval_embeds_query(self, gateway, effective_request, mock_embedding_service):
        """Dense retrieval embeds the query."""
        result = gateway.retrieve(effective_request)

        mock_embedding_service.embed_text.assert_called_once_with("test query")

    def test_dense_retrieval_applies_scope_filters(
        self, gateway, effective_request, mock_vector_store, mock_embedding_service
    ):
        """Dense retrieval passes scope filters to vector store."""
        result = gateway.retrieve(effective_request)

        # Verify build_query_filter was called with the scope
        mock_vector_store.build_query_filter.assert_called_once()
        call_args = mock_vector_store.build_query_filter.call_args[0]
        scope = call_args[0]

        assert scope.service_name == "test-service"
        assert scope.tenant_id == "tenant-1"
        assert scope.collections == ["docs"]

    def test_dense_retrieval_normalizes_results(self, gateway, effective_request):
        """Dense retrieval normalizes results to RetrievedChunk objects."""
        result = gateway.retrieve(effective_request)

        assert isinstance(result, RetrievalGatewayResult)
        assert len(result.chunks) == 2
        assert all(isinstance(chunk, RetrievedChunk) for chunk in result.chunks)

    def test_dense_retrieval_sets_chunk_id(self, gateway, effective_request):
        """Dense retrieval derives chunk_id from document_id and chunk_index."""
        result = gateway.retrieve(effective_request)

        assert result.chunks[0].chunk_id == "doc-1:0"
        assert result.chunks[1].chunk_id == "doc-1:1"

    def test_dense_retrieval_sets_rank(self, gateway, effective_request):
        """Dense retrieval sets rank reflecting result order."""
        result = gateway.retrieve(effective_request)

        assert result.chunks[0].rank == 0
        assert result.chunks[1].rank == 1

    def test_dense_retrieval_sets_retrieval_mode(self, gateway, effective_request):
        """Dense retrieval sets retrieval_mode to DENSE."""
        result = gateway.retrieve(effective_request)

        assert all(chunk.retrieval_mode == RetrievalMode.DENSE for chunk in result.chunks)

    def test_dense_retrieval_includes_diagnostics(self, gateway, effective_request):
        """Dense retrieval reports filter diagnostics."""
        result = gateway.retrieve(effective_request)

        assert result.diagnostics["retrieval_mode"] == "dense"
        assert result.diagnostics["backend_names"] == ["vector"]
        assert result.diagnostics["requested_collections"] == ["docs"]
        assert result.diagnostics["filter_keys"] == ["service_name", "tenant_id", "collections", "source_type"]
        assert result.diagnostics["dense_candidate_count"] == 2
        assert result.diagnostics["filters_applied_by_dense"] is True

    def test_dense_retrieval_preserves_domain_metadata(self, gateway, effective_request):
        """Dense retrieval keeps backend domain metadata on the chunk."""
        result = gateway.retrieve(effective_request)

        assert result.chunks[0].metadata["style_category"] == "voice_rules"
        assert result.chunks[0].metadata["platform"] == "twitter"


class TestLexicalRetrieval:
    """Tests for lexical retrieval mode."""

    def test_lexical_retrieval_no_embedding(
        self, gateway, effective_request, mock_embedding_service
    ):
        """Lexical retrieval does not embed the query."""
        effective_request.retrieval_mode = RetrievalMode.LEXICAL
        result = gateway.retrieve(effective_request)

        mock_embedding_service.embed_text.assert_not_called()

    def test_lexical_retrieval_applies_scope_filters(
        self, gateway, effective_request, mock_lexical_search
    ):
        """Lexical retrieval passes scope filters to lexical search."""
        effective_request.retrieval_mode = RetrievalMode.LEXICAL
        result = gateway.retrieve(effective_request)

        call_kwargs = mock_lexical_search.search.call_args.kwargs
        filters = call_kwargs["filters"]

        assert filters["service_name"] == "test-service"
        assert filters["tenant_id"] == "tenant-1"
        assert filters["collections"] == ["docs"]
        assert filters["source_type"] == "pdf"

    def test_lexical_retrieval_normalizes_results(self, gateway, effective_request):
        """Lexical retrieval normalizes results to RetrievedChunk objects."""
        effective_request.retrieval_mode = RetrievalMode.LEXICAL
        result = gateway.retrieve(effective_request)

        assert isinstance(result, RetrievalGatewayResult)
        assert len(result.chunks) == 2
        assert all(isinstance(chunk, RetrievedChunk) for chunk in result.chunks)

    def test_lexical_retrieval_sets_retrieval_mode(self, gateway, effective_request):
        """Lexical retrieval sets retrieval_mode to LEXICAL."""
        effective_request.retrieval_mode = RetrievalMode.LEXICAL
        result = gateway.retrieve(effective_request)

        assert all(chunk.retrieval_mode == RetrievalMode.LEXICAL for chunk in result.chunks)

    def test_lexical_retrieval_includes_diagnostics(self, gateway, effective_request):
        """Lexical retrieval reports filter diagnostics."""
        effective_request.retrieval_mode = RetrievalMode.LEXICAL
        result = gateway.retrieve(effective_request)

        assert result.diagnostics["retrieval_mode"] == "lexical"
        assert result.diagnostics["backend_names"] == ["lexical"]
        assert result.diagnostics["lexical_candidate_count"] == 2
        assert result.diagnostics["filters_applied_by_lexical"] is True

    def test_lexical_retrieval_preserves_domain_metadata(self, gateway, effective_request):
        """Lexical retrieval keeps backend domain metadata on the chunk."""
        effective_request.retrieval_mode = RetrievalMode.LEXICAL
        result = gateway.retrieve(effective_request)

        assert result.chunks[0].metadata["style_category"] == "voice_rules"
        assert result.chunks[0].metadata["platform"] == "twitter"


class TestHybridRetrieval:
    """Tests for hybrid retrieval mode with RRF fusion."""

    def test_hybrid_retrieval_runs_both_backends(
        self, gateway, effective_request, mock_vector_store, mock_lexical_search
    ):
        """Hybrid retrieval queries both vector and lexical backends."""
        effective_request.retrieval_mode = RetrievalMode.HYBRID
        result = gateway.retrieve(effective_request)

        assert mock_vector_store.search.called
        assert mock_lexical_search.search.called

    def test_hybrid_retrieval_preserves_domain_metadata(self, gateway, effective_request):
        effective_request.retrieval_mode = RetrievalMode.HYBRID
        result = gateway.retrieve(effective_request)

        assert result.chunks[0].metadata["style_category"] == "voice_rules"
        assert result.chunks[0].metadata["platform"] == "twitter"

    def test_hybrid_retrieval_applies_same_filters(
        self, gateway, effective_request, mock_vector_store, mock_lexical_search
    ):
        """Hybrid retrieval applies same filters to both backends."""
        effective_request.retrieval_mode = RetrievalMode.HYBRID
        result = gateway.retrieve(effective_request)

        # Verify both backends were called with filters
        assert mock_vector_store.build_query_filter.called
        assert mock_lexical_search.search.called

        lexical_filters = mock_lexical_search.search.call_args.kwargs["filters"]
        assert lexical_filters["service_name"] == "test-service"
        assert lexical_filters["tenant_id"] == "tenant-1"

    def test_hybrid_retrieval_fuses_results_with_rrf(self, gateway, effective_request):
        """Hybrid retrieval fuses results using reciprocal rank fusion."""
        effective_request.retrieval_mode = RetrievalMode.HYBRID
        result = gateway.retrieve(effective_request)

        # Should have results from both backends fused
        assert len(result.chunks) > 0
        # Results should be sorted by fused RRF score
        # (exact ordering depends on RRF calculation)

    def test_hybrid_retrieval_includes_both_diagnostics(self, gateway, effective_request):
        """Hybrid retrieval reports diagnostics for both backends."""
        effective_request.retrieval_mode = RetrievalMode.HYBRID
        result = gateway.retrieve(effective_request)

        assert result.diagnostics["retrieval_mode"] == "hybrid"
        assert "vector" in result.diagnostics["backend_names"]
        assert "lexical" in result.diagnostics["backend_names"]
        assert "dense_candidate_count" in result.diagnostics
        assert "lexical_candidate_count" in result.diagnostics
        assert result.diagnostics["filters_applied_by_dense"] is True
        assert result.diagnostics["filters_applied_by_lexical"] is True


class TestResultNormalization:
    """Tests for result normalization to RetrievedChunk."""

    def test_chunk_metadata_includes_all_backend_fields(self, gateway, effective_request):
        """Normalized chunks include all metadata from backend."""
        result = gateway.retrieve(effective_request)

        chunk = result.chunks[0]
        assert chunk.metadata["service_name"] == "test-service"
        assert chunk.metadata["tenant_id"] == "tenant-1"
        assert chunk.metadata["collection"] == "docs"
        assert chunk.metadata["source_type"] == "pdf"
        assert chunk.metadata["source_label"] == "test.pdf"

    def test_chunk_document_id_preserved(self, gateway, effective_request):
        """Normalized chunks preserve document_id."""
        result = gateway.retrieve(effective_request)

        assert result.chunks[0].document_id == "doc-1"
        assert result.chunks[1].document_id == "doc-1"

    def test_chunk_content_from_text_field(self, gateway, effective_request):
        """Normalized chunks use text field as content."""
        result = gateway.retrieve(effective_request)

        assert result.chunks[0].content == "Dense result 1"
        assert result.chunks[1].content == "Dense result 2"


class TestFilterTranslation:
    """Tests for filter translation from scope to backend filters."""

    def test_scope_filters_translated_correctly(
        self, gateway, effective_request, mock_vector_store
    ):
        """Scope filters are translated to backend filter format."""
        effective_request.validated_scope.filters = {
            "author": "John Doe",
            "tags": ["important", "urgent"],
        }

        result = gateway.retrieve(effective_request)

        # Verify the scope with custom filters was passed to build_query_filter
        call_args = mock_vector_store.build_query_filter.call_args[0]
        scope = call_args[0]
        assert scope.filters["author"] == "John Doe"
        assert scope.filters["tags"] == ["important", "urgent"]


class TestErrorTranslation:
    """Tests for gateway error translation to domain errors."""

    def test_empty_vector_corpus_raises_no_indexed_corpus_for_dense(
        self, gateway, effective_request, mock_vector_store
    ):
        mock_vector_store.has_indexed_chunks.return_value = False

        with pytest.raises(NoIndexedCorpusError):
            gateway.retrieve(effective_request)

    def test_empty_lexical_corpus_raises_no_indexed_corpus_for_lexical(
        self, gateway, effective_request, mock_lexical_search
    ):
        effective_request.retrieval_mode = RetrievalMode.LEXICAL
        mock_lexical_search.has_indexed_chunks.return_value = False

        with pytest.raises(NoIndexedCorpusError):
            gateway.retrieve(effective_request)

    def test_hybrid_succeeds_when_one_backend_has_content(
        self, gateway, effective_request, mock_vector_store, mock_lexical_search
    ):
        """Hybrid only fails when both backends are empty."""
        effective_request.retrieval_mode = RetrievalMode.HYBRID
        mock_vector_store.has_indexed_chunks.return_value = False
        mock_lexical_search.has_indexed_chunks.return_value = True
        mock_vector_store.search.return_value = []

        result = gateway.retrieve(effective_request)

        assert len(result.chunks) > 0

    def test_hybrid_raises_when_both_backends_empty(
        self, gateway, effective_request, mock_vector_store, mock_lexical_search
    ):
        effective_request.retrieval_mode = RetrievalMode.HYBRID
        mock_vector_store.has_indexed_chunks.return_value = False
        mock_lexical_search.has_indexed_chunks.return_value = False

        with pytest.raises(NoIndexedCorpusError):
            gateway.retrieve(effective_request)

    def test_backend_exception_wrapped_in_retrieval_execution_error(
        self, gateway, effective_request, mock_vector_store
    ):
        mock_vector_store.search.side_effect = RuntimeError("qdrant down")

        with pytest.raises(RetrievalExecutionError) as exc_info:
            gateway.retrieve(effective_request)

        assert "qdrant down" in exc_info.value.internal_message
        assert exc_info.value.details["exception_type"] == "RuntimeError"

    def test_lexical_filter_error_wrapped_in_retrieval_execution_error(
        self, gateway, effective_request, mock_lexical_search
    ):
        effective_request.retrieval_mode = RetrievalMode.LEXICAL
        mock_lexical_search.search.side_effect = ValueError("Invalid filter key: 'x y'")

        with pytest.raises(RetrievalExecutionError):
            gateway.retrieve(effective_request)
