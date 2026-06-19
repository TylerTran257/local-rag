"""ProfileStore persistence/resolution and the /profiles API."""
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth import ApiKeyRegistry
from app.composition import MetadataAwareRuntime
from app.db.database import Base
from app.main import create_app
from app.profiles import ServiceProfile, default_profile
from app.profiles.store import ProfileEmbeddingModelImmutableError, ProfileStore
from app.retrieval.types import RetrievalMode
from app.settings import settings


@pytest.fixture
def profile_store():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return ProfileStore(session_factory=factory)


# --- store tests ----------------------------------------------------------

class TestProfileStore:
    def test_get_missing_returns_default(self, profile_store):
        profile = profile_store.get("svc")
        assert profile.service_name == "svc"
        assert profile.embedding_model == settings.embedding_model_name
        assert profile.chunk_size == 800
        assert profile.chunk_overlap == 120

    def test_upsert_then_get_roundtrip(self, profile_store):
        profile_store.upsert(
            ServiceProfile(
                service_name="svc",
                embedding_model=settings.embedding_model_name,
                chunk_size=300,
                chunk_overlap=30,
                dense_limit=7,
                lexical_limit=9,
                fusion_rrf_k=42,
                default_mode=RetrievalMode.DENSE,
                generation_overrides={"temperature": 0.1},
            )
        )
        loaded = profile_store.get("svc")
        assert loaded.chunk_size == 300
        assert loaded.chunk_overlap == 30
        assert loaded.dense_limit == 7
        assert loaded.default_mode == RetrievalMode.DENSE
        assert loaded.generation_overrides == {"temperature": 0.1}

    def test_update_preserves_immutable_embedding_model(self, profile_store):
        profile_store.upsert(default_profile("svc"))
        changed = ServiceProfile(service_name="svc", embedding_model="other-model")
        with pytest.raises(ProfileEmbeddingModelImmutableError):
            profile_store.upsert(changed)

    def test_update_same_embedding_model_allowed(self, profile_store):
        profile_store.upsert(default_profile("svc"))
        updated = ServiceProfile(
            service_name="svc",
            embedding_model=settings.embedding_model_name,
            chunk_size=512,
        )
        result = profile_store.upsert(updated)
        assert result.chunk_size == 512


# --- API tests ------------------------------------------------------------

def _registry():
    return ApiKeyRegistry.from_entries(
        [
            {"key": "admin-key", "key_id": "admin", "services": ["*"], "tenants": ["*"],
             "collections": ["*"], "admin": True},
            {"key": "svc-a-key", "key_id": "svc-a", "services": ["service-a"],
             "tenants": ["*"], "collections": ["*"]},
        ]
    )


@pytest.fixture
def client(profile_store):
    runtime = MetadataAwareRuntime(
        retrieve_use_case=Mock(),
        ingest_use_case=Mock(),
        gateway=Mock(),
        profile_store=profile_store,
    )
    app = create_app(
        generation_service=Mock(),
        metadata_aware_runtime=runtime,
        api_key_registry=_registry(),
    )
    return TestClient(app)


class TestProfilesApi:
    def test_owner_can_upsert_and_get(self, client):
        resp = client.post(
            "/profiles",
            json={"service_name": "service-a", "chunk_size": 256},
            headers={"X-API-Key": "svc-a-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["chunk_size"] == 256

        got = client.get("/profiles/service-a", headers={"X-API-Key": "svc-a-key"})
        assert got.status_code == 200
        assert got.json()["chunk_size"] == 256

    def test_non_owner_cannot_manage_other_service(self, client):
        resp = client.post(
            "/profiles",
            json={"service_name": "service-b", "chunk_size": 256},
            headers={"X-API-Key": "svc-a-key"},
        )
        assert resp.status_code == 403

    def test_admin_can_manage_any_service(self, client):
        resp = client.post(
            "/profiles",
            json={"service_name": "service-b", "chunk_size": 999},
            headers={"X-API-Key": "admin-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["chunk_size"] == 999

    def test_changing_embedding_model_returns_409(self, client):
        client.post(
            "/profiles",
            json={"service_name": "service-a", "embedding_model": settings.embedding_model_name},
            headers={"X-API-Key": "svc-a-key"},
        )
        resp = client.post(
            "/profiles",
            json={"service_name": "service-a", "embedding_model": "different-model"},
            headers={"X-API-Key": "svc-a-key"},
        )
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "PROFILE_EMBEDDING_MODEL_IMMUTABLE"

    def test_missing_auth_returns_401(self, client):
        resp = client.get("/profiles/service-a")
        assert resp.status_code == 401
