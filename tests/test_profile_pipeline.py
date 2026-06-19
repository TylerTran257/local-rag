"""Profiles change ingestion behavior: chunking size and target collection."""
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.ingest.contracts import IngestDocument
from app.ingest.use_case import IngestUseCase
from app.profiles import ServiceProfile, collection_for, default_profile
from app.profiles.store import ProfileStore
from app.settings import settings


class FakeEmbeddingService:
    """Records the model used; returns small deterministic vectors."""

    def __init__(self):
        self.models_used = []

    def embed_texts(self, texts, model_name=None):
        self.models_used.append(model_name)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest.fixture
def profile_store():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return ProfileStore(session_factory=sessionmaker(bind=engine))


def _use_case(profile_store, embedding):
    return IngestUseCase(
        embedding_service=embedding,
        vector_store_service=Mock(),
        lexical_search_service=Mock(),
        profile_store=profile_store,
    )


LONG_TEXT = " ".join(f"word{i}" for i in range(400))


def _doc(service_name, text=LONG_TEXT):
    return IngestDocument(
        text=text,
        service_name=service_name,
        tenant_id="t1",
        collection="docs",
        source_type="text",
        source_label="f.txt",
    )


def test_smaller_chunk_size_produces_more_chunks(profile_store):
    embedding = FakeEmbeddingService()
    use_case = _use_case(profile_store, embedding)

    # Default profile (chunk_size 800) for service-default.
    default_count = use_case.ingest_document(_doc("service-default")).chunk_count

    # Small chunk size for service-small.
    profile_store.upsert(
        ServiceProfile(
            service_name="service-small",
            embedding_model=settings.embedding_model_name,
            chunk_size=120,
            chunk_overlap=0,
        )
    )
    small_count = use_case.ingest_document(_doc("service-small")).chunk_count

    assert small_count > default_count


def test_embedding_model_selects_collection_and_model(profile_store):
    embedding = FakeEmbeddingService()
    vector_store = Mock()
    use_case = IngestUseCase(
        embedding_service=embedding,
        vector_store_service=vector_store,
        lexical_search_service=Mock(),
        profile_store=profile_store,
    )

    profile_store.upsert(
        ServiceProfile(service_name="svc-custom", embedding_model="custom/model-x")
    )
    use_case.ingest_document(_doc("svc-custom"))

    # Embedding used the profile's model.
    assert embedding.models_used[-1] == "custom/model-x"
    # Vectors were stored in the model's dedicated collection.
    expected_collection = collection_for("custom/model-x")
    assert expected_collection != settings.qdrant_collection_name
    upsert_kwargs = vector_store.upsert_document_chunks.call_args.kwargs
    assert upsert_kwargs["collection_name"] == expected_collection
    ensure_args = vector_store.ensure_collection.call_args
    assert ensure_args.args[0] == expected_collection


def test_default_model_uses_default_collection():
    assert collection_for(settings.embedding_model_name) == settings.qdrant_collection_name
    assert default_profile("svc").embedding_model == settings.embedding_model_name
