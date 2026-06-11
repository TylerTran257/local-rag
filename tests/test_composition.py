"""Tests for the metadata-aware runtime composition factory and create_app()."""
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
from app.retrieval.metadata_gateway import MetadataAwareRetrievalGateway
from app.retrieval.use_case import RetrieveUseCase


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


# ---------------------------------------------------------------------------
# create_app() tests
# ---------------------------------------------------------------------------

class FakeGenerationService:
    """Reusable fake for generation service."""
    pass


class TestCreateApp:
    """Tests for create_app()."""

    def _make_runtime(self) -> MetadataAwareRuntime:
        return build_metadata_aware_runtime(
            embedding_service=FakeEmbeddingService(),
            vector_store_service=FakeVectorStoreService(),
            lexical_search_service=FakeLexicalSearchService(),
            trace_sink=NoOpTraceSink(),
        )

    def test_uses_metadata_gateway(self):
        runtime = self._make_runtime()
        app = create_app(
            generation_service=FakeGenerationService(),
            metadata_aware_runtime=runtime,
        )
        assert isinstance(
            app.state.retrieve_use_case.gateway, MetadataAwareRetrievalGateway
        )

    def test_wires_ingest_use_case(self):
        runtime = self._make_runtime()
        app = create_app(
            generation_service=FakeGenerationService(),
            metadata_aware_runtime=runtime,
        )
        assert app.state.ingest_use_case is runtime.ingest_use_case

    def test_wires_retrieve_use_case(self):
        runtime = self._make_runtime()
        app = create_app(
            generation_service=FakeGenerationService(),
            metadata_aware_runtime=runtime,
        )
        assert app.state.retrieve_use_case is runtime.retrieve_use_case

    def test_preserves_generation_service(self):
        runtime = self._make_runtime()
        gen_svc = FakeGenerationService()
        app = create_app(
            generation_service=gen_svc,
            metadata_aware_runtime=runtime,
        )
        assert app.state.generation_service is gen_svc
