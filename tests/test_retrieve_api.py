import pytest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.composition import MetadataAwareRuntime
from app.main import create_app
from app.retrieval.types import RetrievedChunk, RetrievalMode, RetrievalWarning
from app.retrieval.use_case import RetrieveResult


@pytest.fixture
def mock_retrieve_use_case():
    use_case = Mock()
    # Return a successful retrieval result with one chunk
    use_case.execute.return_value = RetrieveResult(
        chunks=[
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="This is retrieved content.",
                score=0.95,
                rank=1,
                retrieval_mode=RetrievalMode.HYBRID,
                metadata={
                    "service_name": "test-service",
                    "tenant_id": "tenant-123",
                    "collection": "documents",
                    "source_label": "test.pdf",
                    "chunk_index": 0,
                    "is_external": False,
                    "topic": "platform",
                },
            )
        ],
        warnings=[],
        trace_id="trace-123",
    )
    return use_case


@pytest.fixture
def client(mock_retrieve_use_case, api_key_registry, auth_headers):
    mock_runtime = MetadataAwareRuntime(
        retrieve_use_case=mock_retrieve_use_case,
        ingest_use_case=Mock(),
        gateway=Mock(),
    )

    app = create_app(
        generation_service=Mock(),
        metadata_aware_runtime=mock_runtime,
        api_key_registry=api_key_registry,
    )
    return TestClient(app, headers=auth_headers)


class TestRetrieveEndpoint:
    """Tests for POST /retrieve."""

    def test_successful_retrieval_returns_chunks(self, client, mock_retrieve_use_case):
        payload = {
            "query": "test query",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collections": ["documents"],
            "filters": {},
            "limit": 5,
            "mode": "hybrid",
        }

        response = client.post("/retrieve", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["trace_id"] == "trace-123"
        assert len(body["chunks"]) == 1
        assert body["chunks"][0]["text"] == "This is retrieved content."
        assert body["chunks"][0]["score"] == 0.95
        assert body["chunks"][0]["chunk_id"] == "chunk-1"
        assert body["chunks"][0]["service_name"] == "test-service"
        assert body["chunks"][0]["tenant_id"] == "tenant-123"
        assert body["chunks"][0]["collection"] == "documents"
        assert body["chunks"][0]["domain_metadata"] == {
            "is_external": False,
            "topic": "platform",
        }

    def test_empty_retrieval_returns_empty_chunks(self, client, mock_retrieve_use_case):
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[],
            warnings=[],
            trace_id="trace-456",
        )
        payload = {
            "query": "no results query",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collections": ["documents"],
        }

        response = client.post("/retrieve", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["trace_id"] == "trace-456"
        assert len(body["chunks"]) == 0

    def test_missing_required_fields_returns_422(self, client):
        payload = {
            "query": "test query",
            # Missing service_name, tenant_id, collections
        }

        response = client.post("/retrieve", json=payload)

        assert response.status_code == 422

    def test_empty_query_returns_422(self, client):
        payload = {
            "query": "",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collections": ["documents"],
        }

        response = client.post("/retrieve", json=payload)

        assert response.status_code == 422

    def test_reserved_filter_keys_return_422(self, client, mock_retrieve_use_case):
        """Filters must not be able to override scope-enforcement keys."""
        for reserved_key in ["service_name", "tenant_id", "collection", "collections"]:
            payload = {
                "query": "test query",
                "service_name": "test-service",
                "tenant_id": "tenant-123",
                "collections": ["documents"],
                "filters": {reserved_key: "other-value"},
            }

            response = client.post("/retrieve", json=payload)

            assert response.status_code == 422, f"filter key {reserved_key} was not rejected"

        mock_retrieve_use_case.execute.assert_not_called()
