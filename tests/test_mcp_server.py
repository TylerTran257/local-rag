"""MCP tool logic (McpService) reuses the same use cases, profiles, and auth."""
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth import ApiKeyRegistry
from app.auth.errors import AuthenticationError, AuthorizationError
from app.composition import MetadataAwareRuntime
from app.db.database import Base
from app.ingest.use_case import IngestResult
from app.mcp import McpService
from app.profiles.store import ProfileStore
from app.retrieval.types import RetrievedChunk, RetrievalMode
from app.retrieval.use_case import RetrieveResult


def _registry():
    return ApiKeyRegistry.from_entries(
        [
            {"key": "svc-a-key", "key_id": "svc-a", "services": ["service-a"],
             "tenants": ["tenant-1"], "collections": ["docs"]},
        ]
    )


@pytest.fixture
def profile_store():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    return ProfileStore(session_factory=sessionmaker(bind=engine))


@pytest.fixture
def service(profile_store):
    retrieve_use_case = Mock()
    retrieve_use_case.execute.return_value = RetrieveResult(
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                content="hello",
                score=0.9,
                rank=1,
                retrieval_mode=RetrievalMode.HYBRID,
                metadata={
                    "service_name": "service-a",
                    "tenant_id": "tenant-1",
                    "collection": "docs",
                    "source_label": "f.txt",
                    "chunk_index": 0,
                },
            )
        ],
        warnings=[],
        trace_id="trace-1",
    )
    ingest_use_case = Mock()
    ingest_use_case.ingest_document.return_value = IngestResult(chunk_count=4)
    generation = Mock()
    generation.answer_question.return_value = "an answer"

    runtime = MetadataAwareRuntime(
        retrieve_use_case=retrieve_use_case,
        ingest_use_case=ingest_use_case,
        gateway=Mock(),
        profile_store=profile_store,
    )
    return McpService(runtime=runtime, generation_service=generation, registry=_registry())


def _principal(service):
    return service.resolve_principal("svc-a-key")


class TestAuth:
    def test_resolve_valid_key(self, service):
        assert service.resolve_principal("svc-a-key").key_id == "svc-a"

    def test_resolve_invalid_key_raises(self, service):
        with pytest.raises(AuthenticationError):
            service.resolve_principal("bad")


class TestRetrieve:
    def test_retrieve_in_scope(self, service):
        result = service.retrieve(
            _principal(service),
            query="q",
            service_name="service-a",
            tenant_id="tenant-1",
            collections=["docs"],
        )
        assert result["chunks"][0]["text"] == "hello"
        assert result["trace_id"] == "trace-1"

    def test_retrieve_out_of_scope_raises(self, service):
        with pytest.raises(AuthorizationError):
            service.retrieve(
                _principal(service),
                query="q",
                service_name="service-b",
                tenant_id="tenant-1",
                collections=["docs"],
            )


class TestAnswer:
    def test_answer_generates(self, service):
        result = service.answer(
            _principal(service),
            query="q",
            service_name="service-a",
            tenant_id="tenant-1",
            collections=["docs"],
        )
        assert result["answer"] == "an answer"
        assert len(result["sources"]) == 1


class TestIngest:
    def test_ingest_in_scope(self, service):
        result = service.ingest_document(
            _principal(service),
            text="body",
            service_name="service-a",
            tenant_id="tenant-1",
            collection="docs",
            source_type="text",
            source_label="f.txt",
        )
        assert result["chunk_count"] == 4

    def test_ingest_out_of_scope_raises(self, service):
        with pytest.raises(AuthorizationError):
            service.ingest_document(
                _principal(service),
                text="body",
                service_name="service-a",
                tenant_id="tenant-1",
                collection="secret",
                source_type="text",
                source_label="f.txt",
            )


class TestProfiles:
    def test_configure_and_get_profile(self, service):
        saved = service.configure_profile(
            _principal(service), service_name="service-a", chunk_size=321
        )
        assert saved["chunk_size"] == 321
        got = service.get_profile(_principal(service), service_name="service-a")
        assert got["chunk_size"] == 321

    def test_configure_other_service_raises(self, service):
        with pytest.raises(AuthorizationError):
            service.configure_profile(
                _principal(service), service_name="service-b", chunk_size=10
            )


def test_health(service):
    assert service.health() == {"status": "ok"}
