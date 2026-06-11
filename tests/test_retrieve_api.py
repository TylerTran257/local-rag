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
                },
            )
        ],
        warnings=[],
        trace_id="trace-123",
    )
    return use_case


@pytest.fixture
def client(mock_retrieve_use_case):
    mock_runtime = MetadataAwareRuntime(
        retrieve_use_case=mock_retrieve_use_case,
        ingest_use_case=Mock(),
        gateway=Mock(),
    )

    app = create_app(
        generation_service=Mock(),
        metadata_aware_runtime=mock_runtime,
    )
    return TestClient(app)


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
