"""Tests for the metadata-aware runtime composition factory and create_app() modes."""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.composition import MetadataAwareRuntime, build_metadata_aware_runtime
from app.ingest.use_case import IngestUseCase
from app.main import create_app
from app.retrieval import (
    InMemoryTraceSink,
    NoOpTraceSink,
    PassthroughScopePolicy,
)
from app.retrieval.legacy_adapter import LegacyDocumentRetrievalAdapter
from app.retrieval.metadata_gateway import MetadataAwareRetrievalGateway
from app.retrieval.use_case import RetrieveUseCase
from app.social.service import SocialStyleRetrievalService


# ---------------------------------------------------------------------------
# Fake infrastructure services for unit-level composition tests
# ---------------------------------------------------------------------------

class FakeEmbeddingService:
    """Minimal stand-in for EmbeddingService."""

    def embed_text(self, text: str) -> list[float]:
        return [0.0] * 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]


class FakeVectorStoreService:
    """Minimal stand-in for VectorStoreService."""

    def has_indexed_chunks(self) -> bool:
        return False

    def search(self, *, query_embedding, limit, query_filter=None):
        return []

    def build_query_filter(self, scope):
        return None

    def upsert_document_chunks(self, **kwargs):
        pass


class FakeLexicalSearchService:
    """Minimal stand-in for LexicalSearchService."""

    def has_indexed_chunks(self) -> bool:
        return False

    def search(self, *, query, limit, filters=None):
        return []

    def index_document_chunks(self, **kwargs):
        pass


class FakeClock:
    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeTraceIdGenerator:
    def __init__(self, trace_id: str = "test-trace-id"):
        self._trace_id = trace_id

    def generate(self) -> str:
        return self._trace_id


# ---------------------------------------------------------------------------
# Composition factory tests
# ---------------------------------------------------------------------------

class TestBuildMetadataAwareRuntime:
    """Tests for build_metadata_aware_runtime()."""

    def test_returns_metadata_aware_runtime_dataclass(self):
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
        )
        assert isinstance(runtime, MetadataAwareRuntime)

    def test_retrieve_use_case_uses_metadata_gateway(self):
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
        )
        assert isinstance(runtime.retrieve_use_case, RetrieveUseCase)
        assert isinstance(runtime.retrieve_use_case.gateway, MetadataAwareRetrievalGateway)

    def test_ingest_use_case_is_constructed(self):
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
        )
        assert isinstance(runtime.ingest_use_case, IngestUseCase)

    def test_social_style_service_is_constructed(self):
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
        )
        assert isinstance(runtime.social_style_service, SocialStyleRetrievalService)

    def test_gateway_exposed_on_runtime(self):
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
        )
        assert isinstance(runtime.gateway, MetadataAwareRetrievalGateway)

    def test_accepts_custom_scope_policy(self):
        from app.retrieval.policy import NamespacePolicy

        custom_policy = NamespacePolicy(allowed_collections={"test-collection"})
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
            scope_policy=custom_policy,
        )
        assert runtime.retrieve_use_case.scope_policy is custom_policy

    def test_accepts_custom_trace_sink(self):
        sink = InMemoryTraceSink()
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
            trace_sink=sink,
        )
        assert runtime.retrieve_use_case.trace_sink is sink

    def test_accepts_custom_clock_and_trace_id_generator(self):
        clock = FakeClock()
        gen = FakeTraceIdGenerator("custom-id")
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
            clock=clock,
            trace_id_generator=gen,
        )
        assert runtime.retrieve_use_case.clock is clock
        assert runtime.retrieve_use_case.trace_id_generator is gen

    def test_default_scope_policy_is_passthrough(self):
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
        )
        assert isinstance(runtime.retrieve_use_case.scope_policy, PassthroughScopePolicy)

    def test_custom_social_style_service_name(self):
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
            social_style_service_name="custom-service",
        )
        assert runtime.social_style_service.service_name == "custom-service"

    def test_shared_infrastructure_services(self):
        """All components should share the same infrastructure service instances."""
        embedding = FakeEmbeddingService()
        vector = FakeVectorStoreService()
        lexical = FakeLexicalSearchService()

        runtime = build_metadata_aware_runtime(
            embedding_service=embedding,
            vector_store_service=vector,
            lexical_search_service=lexical,
        )

        # Gateway uses same services
        assert runtime.gateway.embedding_service is embedding
        assert runtime.gateway.vector_store_service is vector
        assert runtime.gateway.lexical_search_service is lexical

        # IngestUseCase uses same services
        assert runtime.ingest_use_case.embedding_service is embedding
        assert runtime.ingest_use_case.vector_store_service is vector
        assert runtime.ingest_use_case.lexical_search_service is lexical

    def test_social_style_service_uses_retrieve_use_case(self):
        runtime = build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
        )
        assert runtime.social_style_service.retrieve_use_case is runtime.retrieve_use_case


# ---------------------------------------------------------------------------
# create_app() mode tests
# ---------------------------------------------------------------------------

class FakeDocumentService:
    """Reusable fake for legacy mode tests."""
    pass


class FakeGenerationService:
    """Reusable fake for generation service."""
    pass


class TestCreateAppLegacyMode:
    """Tests for create_app() in default legacy mode."""

    def test_legacy_mode_by_default(self):
        doc_svc = FakeDocumentService()
        gen_svc = FakeGenerationService()
        app = create_app(document_service=doc_svc, generation_service=gen_svc)
        # Legacy mode should use LegacyDocumentRetrievalAdapter as gateway
        assert isinstance(
            app.state.retrieve_use_case.gateway, LegacyDocumentRetrievalAdapter
        )

    def test_legacy_mode_explicit_false(self):
        doc_svc = FakeDocumentService()
        gen_svc = FakeGenerationService()
        app = create_app(
            document_service=doc_svc,
            generation_service=gen_svc,
            metadata_aware=False,
        )
        assert isinstance(
            app.state.retrieve_use_case.gateway, LegacyDocumentRetrievalAdapter
        )

    def test_legacy_mode_preserves_document_service(self):
        doc_svc = FakeDocumentService()
        gen_svc = FakeGenerationService()
        app = create_app(document_service=doc_svc, generation_service=gen_svc)
        assert app.state.document_service is doc_svc

    def test_legacy_mode_no_ingest_or_social_on_state(self):
        doc_svc = FakeDocumentService()
        gen_svc = FakeGenerationService()
        app = create_app(document_service=doc_svc, generation_service=gen_svc)
        assert not hasattr(app.state, "ingest_use_case")
        assert not hasattr(app.state, "social_style_service")


class TestCreateAppMetadataAwareMode:
    """Tests for create_app() with metadata_aware=True."""

    def _make_runtime(self) -> MetadataAwareRuntime:
        return build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
            trace_sink=NoOpTraceSink(),
        )

    def test_metadata_aware_mode_uses_metadata_gateway(self):
        runtime = self._make_runtime()
        app = create_app(
            document_service=FakeDocumentService(),
            generation_service=FakeGenerationService(),
            metadata_aware=True,
            metadata_aware_runtime=runtime,
        )
        assert isinstance(
            app.state.retrieve_use_case.gateway, MetadataAwareRetrievalGateway
        )

    def test_metadata_aware_mode_wires_ingest_use_case(self):
        runtime = self._make_runtime()
        app = create_app(
            document_service=FakeDocumentService(),
            generation_service=FakeGenerationService(),
            metadata_aware=True,
            metadata_aware_runtime=runtime,
        )
        assert app.state.ingest_use_case is runtime.ingest_use_case

    def test_metadata_aware_mode_wires_social_style_service(self):
        runtime = self._make_runtime()
        app = create_app(
            document_service=FakeDocumentService(),
            generation_service=FakeGenerationService(),
            metadata_aware=True,
            metadata_aware_runtime=runtime,
        )
        assert app.state.social_style_service is runtime.social_style_service

    def test_metadata_aware_mode_wires_retrieve_use_case(self):
        runtime = self._make_runtime()
        app = create_app(
            document_service=FakeDocumentService(),
            generation_service=FakeGenerationService(),
            metadata_aware=True,
            metadata_aware_runtime=runtime,
        )
        assert app.state.retrieve_use_case is runtime.retrieve_use_case

    def test_metadata_aware_mode_preserves_generation_service(self):
        runtime = self._make_runtime()
        gen_svc = FakeGenerationService()
        app = create_app(
            document_service=FakeDocumentService(),
            generation_service=gen_svc,
            metadata_aware=True,
            metadata_aware_runtime=runtime,
        )
        assert app.state.generation_service is gen_svc

    def test_metadata_aware_mode_preserves_document_service(self):
        runtime = self._make_runtime()
        doc_svc = FakeDocumentService()
        app = create_app(
            document_service=doc_svc,
            generation_service=FakeGenerationService(),
            metadata_aware=True,
            metadata_aware_runtime=runtime,
        )
        assert app.state.document_service is doc_svc
