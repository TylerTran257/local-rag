"""API-key registry resolution and endpoint scope enforcement."""
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.auth import ApiKeyRegistry, enforce_scope
from app.auth.errors import AuthorizationError
from app.composition import MetadataAwareRuntime
from app.main import create_app
from app.retrieval.types import RetrievedChunk, RetrievalMode
from app.retrieval.use_case import RetrieveResult


def _registry():
    return ApiKeyRegistry.from_entries(
        [
            {
                "key": "admin-key",
                "key_id": "admin",
                "services": ["*"],
                "tenants": ["*"],
                "collections": ["*"],
                "admin": True,
            },
            {
                "key": "service-a-key",
                "key_id": "service-a",
                "services": ["service-a"],
                "tenants": ["tenant-1"],
                "collections": ["docs"],
            },
        ]
    )


# --- registry unit tests --------------------------------------------------

class TestApiKeyRegistry:
    def test_resolves_known_key(self):
        principal = _registry().resolve("service-a-key")
        assert principal is not None
        assert principal.key_id == "service-a"
        assert principal.allows_service("service-a")
        assert not principal.allows_service("service-b")

    def test_unknown_key_resolves_to_none(self):
        assert _registry().resolve("nope") is None

    def test_missing_key_resolves_to_none(self):
        assert _registry().resolve(None) is None

    def test_raw_key_is_not_stored(self):
        registry = _registry()
        # The registry stores hashes; the raw secret must not be a dict key.
        assert "service-a-key" not in registry._by_hash


# --- enforce_scope unit tests ---------------------------------------------

class TestEnforceScope:
    def test_admin_bypasses_all(self):
        principal = _registry().resolve("admin-key")
        enforce_scope(principal, service_name="any", tenant_id="any", collections=["x"])

    def test_in_scope_passes(self):
        principal = _registry().resolve("service-a-key")
        enforce_scope(
            principal, service_name="service-a", tenant_id="tenant-1", collections=["docs"]
        )

    def test_service_out_of_scope_raises(self):
        principal = _registry().resolve("service-a-key")
        with pytest.raises(AuthorizationError):
            enforce_scope(
                principal, service_name="service-b", tenant_id="tenant-1", collections=["docs"]
            )

    def test_collection_out_of_scope_raises(self):
        principal = _registry().resolve("service-a-key")
        with pytest.raises(AuthorizationError):
            enforce_scope(
                principal, service_name="service-a", tenant_id="tenant-1", collections=["secret"]
            )


# --- endpoint-level tests -------------------------------------------------

@pytest.fixture
def client():
    use_case = Mock()
    use_case.execute.return_value = RetrieveResult(
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                content="content",
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
    runtime = MetadataAwareRuntime(
        retrieve_use_case=use_case, ingest_use_case=Mock(), gateway=Mock()
    )
    app = create_app(
        generation_service=Mock(),
        metadata_aware_runtime=runtime,
        api_key_registry=_registry(),
    )
    return TestClient(app)


def _retrieve_payload(service_name="service-a"):
    return {
        "query": "q",
        "service_name": service_name,
        "tenant_id": "tenant-1",
        "collections": ["docs"],
    }


class TestEndpointAuth:
    def test_missing_key_returns_401(self, client):
        resp = client.post("/retrieve", json=_retrieve_payload())
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "AUTHENTICATION_FAILED"
        assert resp.headers.get("X-Trace-Id")

    def test_invalid_key_returns_401(self, client):
        resp = client.post(
            "/retrieve", json=_retrieve_payload(), headers={"X-API-Key": "bad"}
        )
        assert resp.status_code == 401

    def test_in_scope_key_succeeds(self, client):
        resp = client.post(
            "/retrieve", json=_retrieve_payload(), headers={"X-API-Key": "service-a-key"}
        )
        assert resp.status_code == 200
        assert resp.json()["chunks"][0]["service_name"] == "service-a"

    def test_out_of_scope_service_returns_403(self, client):
        resp = client.post(
            "/retrieve",
            json=_retrieve_payload(service_name="service-b"),
            headers={"X-API-Key": "service-a-key"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "AUTHORIZATION_FAILED"

    def test_bearer_token_is_accepted(self, client):
        resp = client.post(
            "/retrieve",
            json=_retrieve_payload(),
            headers={"Authorization": "Bearer service-a-key"},
        )
        assert resp.status_code == 200
