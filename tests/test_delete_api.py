from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.composition import MetadataAwareRuntime
from app.delete.contracts import DeleteResult
from app.main import create_app


@pytest.fixture
def mock_delete_use_case():
    use_case = Mock()
    use_case.execute.return_value = DeleteResult(deleted_count=5)
    return use_case


@pytest.fixture
def client(mock_delete_use_case, api_key_registry, auth_headers):
    mock_runtime = MetadataAwareRuntime(
        retrieve_use_case=Mock(),
        ingest_use_case=Mock(),
        gateway=Mock(),
        delete_use_case=mock_delete_use_case,
    )
    app = create_app(
        metadata_aware_runtime=mock_runtime,
        api_key_registry=api_key_registry,
    )
    return TestClient(app, headers=auth_headers)


class TestDeleteEndpoint:
    def test_valid_delete_returns_200(self, client, mock_delete_use_case):
        payload = {
            "service_name": "test-service",
            "tenant_id": "tenant-1",
            "collections": ["docs"],
        }
        response = client.post("/documents/delete", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["deleted_count"] == 5
        mock_delete_use_case.execute.assert_called_once()

    def test_delete_passes_filters(self, client, mock_delete_use_case):
        payload = {
            "service_name": "test-service",
            "tenant_id": "tenant-1",
            "collections": ["docs"],
            "filters": {"document_id": "abc-123"},
        }
        response = client.post("/documents/delete", json=payload)

        assert response.status_code == 200
        call_args = mock_delete_use_case.execute.call_args[0][0]
        assert call_args.filters == {"document_id": "abc-123"}

    def test_missing_service_name_returns_422(self, client):
        payload = {
            "tenant_id": "tenant-1",
            "collections": ["docs"],
        }
        response = client.post("/documents/delete", json=payload)
        assert response.status_code == 422

    def test_empty_collections_returns_422(self, client):
        payload = {
            "service_name": "test-service",
            "tenant_id": "tenant-1",
            "collections": [],
        }
        response = client.post("/documents/delete", json=payload)
        assert response.status_code == 422

    def test_reserved_filter_key_returns_422(self, client):
        payload = {
            "service_name": "test-service",
            "tenant_id": "tenant-1",
            "collections": ["docs"],
            "filters": {"service_name": "override-attempt"},
        }
        response = client.post("/documents/delete", json=payload)
        assert response.status_code == 422
