"""Integration tests wiring IngestUseCase and the metadata gateway to real backends.

These tests use a real SQLite FTS5 index and a real local Qdrant store in
temporary directories, with a deterministic fake embedder. They exist because
mock-based unit tests cannot catch contract mismatches between the pipeline
and the storage services.
"""
import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.ingest.contracts import IngestChunk, IngestDocument
from app.ingest.use_case import IngestUseCase
from app.retrieval import NoOpTraceSink, PassthroughScopePolicy, SystemClock, UuidTraceIdGenerator
from app.retrieval.errors import NoIndexedCorpusError, RetrievalExecutionError
from app.retrieval.metadata_gateway import MetadataAwareRetrievalGateway
from app.retrieval.types import RetrievalMode, RetrievalScope, RetrieveRequest
from app.retrieval.use_case import RetrieveUseCase
from app.services.lexical_search_service import LexicalSearchService
from app.services.vector_store_service import VectorStoreService


class FakeEmbeddingService:
    """Deterministic 384-dim embedder so tests run without a model download."""

    def embed_text(self, text: str, model_name: str | None = None) -> list[float]:
        return self._vector(text)

    def embed_texts(
        self, texts: list[str], model_name: str | None = None
    ) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        values: list[float] = []
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        while len(values) < 384:
            seed = hashlib.sha256(seed).digest()
            values.extend(byte / 255.0 for byte in seed)
        return values[:384]


@pytest.fixture
def runtime(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/integration.db",
        connect_args={"check_same_thread": False},
    )
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    embedding_service = FakeEmbeddingService()
    vector_store = VectorStoreService(qdrant_path=tmp_path / "qdrant")
    lexical_search = LexicalSearchService(session_factory=session_factory)

    ingest_use_case = IngestUseCase(
        embedding_service=embedding_service,
        vector_store_service=vector_store,
        lexical_search_service=lexical_search,
    )
    retrieve_use_case = RetrieveUseCase(
        gateway=MetadataAwareRetrievalGateway(
            vector_store_service=vector_store,
            lexical_search_service=lexical_search,
            embedding_service=embedding_service,
        ),
        scope_policy=PassthroughScopePolicy(),
        clock=SystemClock(),
        trace_id_generator=UuidTraceIdGenerator(),
        trace_sink=NoOpTraceSink(),
    )

    yield ingest_use_case, retrieve_use_case

    engine.dispose()


def _scope(tenant_id: str, filters: dict | None = None) -> RetrievalScope:
    return RetrievalScope(
        service_name="integration-svc",
        tenant_id=tenant_id,
        collections=["notes"],
        filters=filters or {},
    )


def _chunk(chunk_id: str, text: str, tenant_id: str, **domain) -> IngestChunk:
    return IngestChunk(
        chunk_id=chunk_id,
        text=text,
        service_name="integration-svc",
        tenant_id=tenant_id,
        collection="notes",
        source_type="note",
        source_label=f"{chunk_id}.txt",
        domain_metadata=domain,
    )


class TestIngestRetrievalRoundTrip:
    """End-to-end: ingest with real backends, retrieve through the use case."""

    @pytest.mark.parametrize(
        "mode", [RetrievalMode.DENSE, RetrievalMode.LEXICAL, RetrievalMode.HYBRID]
    )
    def test_round_trip_all_modes(self, runtime, mode):
        ingest, retrieve = runtime
        result = ingest.ingest_chunks(
            [_chunk("c1", "postgres index tuning tips", "tenant-a", topic="databases")]
        )
        assert result.chunk_count == 1

        retrieved = retrieve.execute(
            RetrieveRequest(
                query="postgres tuning",
                retrieval_mode=mode,
                limit=5,
                scope=_scope("tenant-a"),
            )
        )

        assert len(retrieved.chunks) == 1
        chunk = retrieved.chunks[0]
        assert chunk.content == "postgres index tuning tips"
        assert chunk.metadata["service_name"] == "integration-svc"
        assert chunk.metadata["tenant_id"] == "tenant-a"
        assert chunk.metadata["collection"] == "notes"
        assert chunk.metadata["source_label"] == "c1.txt"
        # Domain metadata flows through both backends
        assert chunk.metadata["topic"] == "databases"
        assert retrieved.trace_id is not None

    def test_tenant_scoping_enforced(self, runtime):
        ingest, retrieve = runtime
        ingest.ingest_chunks(
            [
                _chunk("a1", "shared keyword alpha report", "tenant-a"),
                _chunk("b1", "shared keyword alpha report", "tenant-b"),
            ]
        )

        retrieved = retrieve.execute(
            RetrieveRequest(
                query="alpha report",
                retrieval_mode=RetrievalMode.HYBRID,
                limit=10,
                scope=_scope("tenant-a"),
            )
        )

        assert len(retrieved.chunks) == 1
        assert retrieved.chunks[0].metadata["tenant_id"] == "tenant-a"

    def test_domain_metadata_filter_in_hybrid_mode(self, runtime):
        """Domain filters work across both backends in hybrid mode."""
        ingest, retrieve = runtime
        ingest.ingest_chunks(
            [
                _chunk("p1", "blood pressure summary", "tenant-a", patient_id="p-1"),
                _chunk("p2", "blood pressure summary", "tenant-a", patient_id="p-2"),
            ]
        )

        retrieved = retrieve.execute(
            RetrieveRequest(
                query="blood pressure",
                retrieval_mode=RetrievalMode.HYBRID,
                limit=10,
                scope=_scope("tenant-a", filters={"patient_id": "p-1"}),
            )
        )

        assert len(retrieved.chunks) == 1
        assert retrieved.chunks[0].metadata["patient_id"] == "p-1"

    def test_mixed_metadata_batch_keeps_per_chunk_metadata(self, runtime):
        """The bug fixed in review: batches must not broadcast chunk 0's metadata."""
        ingest, retrieve = runtime
        ingest.ingest_chunks(
            [
                _chunk("m1", "quarterly revenue numbers", "tenant-a"),
                _chunk("m2", "quarterly revenue numbers", "tenant-b"),
            ]
        )

        for tenant in ("tenant-a", "tenant-b"):
            retrieved = retrieve.execute(
                RetrieveRequest(
                    query="quarterly revenue",
                    retrieval_mode=RetrievalMode.HYBRID,
                    limit=10,
                    scope=_scope(tenant),
                )
            )
            assert len(retrieved.chunks) == 1
            assert retrieved.chunks[0].metadata["tenant_id"] == tenant

    def test_document_ingest_round_trip(self, runtime):
        ingest, retrieve = runtime
        result = ingest.ingest_document(
            IngestDocument(
                text="kubernetes cluster autoscaling guide",
                service_name="integration-svc",
                tenant_id="tenant-a",
                collection="notes",
                source_type="doc",
                source_label="k8s.md",
                domain_metadata={"category": "infra"},
            )
        )
        assert result.chunk_count == 1

        retrieved = retrieve.execute(
            RetrieveRequest(
                query="kubernetes autoscaling",
                retrieval_mode=RetrievalMode.HYBRID,
                limit=5,
                scope=_scope("tenant-a"),
            )
        )

        assert len(retrieved.chunks) == 1
        assert retrieved.chunks[0].metadata["category"] == "infra"


class TestEmptyCorpusErrors:
    """Gateway raises NoIndexedCorpusError before any backend query."""

    def test_dense_empty_corpus(self, runtime):
        _, retrieve = runtime
        with pytest.raises(NoIndexedCorpusError):
            retrieve.execute(
                RetrieveRequest(
                    query="anything",
                    retrieval_mode=RetrievalMode.DENSE,
                    limit=5,
                    scope=_scope("tenant-a"),
                )
            )

    def test_lexical_empty_corpus(self, runtime):
        _, retrieve = runtime
        with pytest.raises(NoIndexedCorpusError):
            retrieve.execute(
                RetrieveRequest(
                    query="anything",
                    retrieval_mode=RetrievalMode.LEXICAL,
                    limit=5,
                    scope=_scope("tenant-a"),
                )
            )


class TestFilterErrorTranslation:
    """Backend filter failures surface as domain errors, not raw exceptions."""

    def test_invalid_filter_key_becomes_retrieval_execution_error(self, runtime):
        ingest, retrieve = runtime
        ingest.ingest_chunks([_chunk("c1", "some indexed text", "tenant-a")])

        with pytest.raises(RetrievalExecutionError):
            retrieve.execute(
                RetrieveRequest(
                    query="indexed text",
                    retrieval_mode=RetrievalMode.LEXICAL,
                    limit=5,
                    scope=_scope("tenant-a", filters={"bad key!": "x"}),
                )
            )
