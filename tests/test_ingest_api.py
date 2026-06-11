import pytest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.ingest.contracts import MetadataValidationError
from app.ingest.use_case import IngestResult
from app.main import create_app


def _valid_document_payload(**overrides):
    payload = {
        "text": "Some document text to ingest.",
        "service_name": "test-service",
        "tenant_id": "tenant-123",
        "collection": "documents",
        "source_type": "pdf",
        "source_label": "test.pdf",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def mock_ingest_use_case():
    use_case = Mock()
    use_case.ingest_document.return_value = IngestResult(chunk_count=3)
    use_case.ingest_chunks.return_value = IngestResult(chunk_count=2)
    return use_case


@pytest.fixture
def client(mock_ingest_use_case, fake_document_service, fake_generation_service):
    from unittest.mock import Mock
    from app.composition import MetadataAwareRuntime

    # Create a minimal runtime with only the ingest use case
    mock_runtime = MetadataAwareRuntime(
        retrieve_use_case=Mock(),
        ingest_use_case=mock_ingest_use_case,
        social_style_service=Mock(),
        gateway=Mock(),
    )

    app = create_app(
        document_service=fake_document_service,
        generation_service=fake_generation_service,
        metadata_aware=True,
        metadata_aware_runtime=mock_runtime,
    )
    return TestClient(app)


class TestIngestDocumentEndpoint:
    """Tests for POST /ingest/document."""

    def test_valid_document_returns_200_with_chunk_count(self, client, mock_ingest_use_case):
        response = client.post("/ingest/document", json=_valid_document_payload())

        assert response.status_code == 200
        assert response.json() == {"chunk_count": 3}
        mock_ingest_use_case.ingest_document.assert_called_once()

    def test_missing_required_field_returns_422(self, client):
        payload = _valid_document_payload()
        del payload["service_name"]

        response = client.post("/ingest/document", json=payload)

        assert response.status_code == 422

    def test_empty_required_field_returns_422(self, client):
        payload = _valid_document_payload(service_name="")

        response = client.post("/ingest/document", json=payload)

        assert response.status_code == 422

    def test_domain_metadata_is_accepted(self, client, mock_ingest_use_case):
        payload = _valid_document_payload(
            domain_metadata={"author": "Jane Doe", "category": "research"}
        )

        response = client.post("/ingest/document", json=payload)

        assert response.status_code == 200
        assert response.json() == {"chunk_count": 3}
        call_args = mock_ingest_use_case.ingest_document.call_args
        document = call_args[0][0]
        assert document.domain_metadata == {"author": "Jane Doe", "category": "research"}

    def test_metadata_validation_error_returns_422(self, client, mock_ingest_use_case):
        mock_ingest_use_case.ingest_document.side_effect = MetadataValidationError(
            invalid_fields=["service_name"],
            metadata={"service_name": ""},
        )

        response = client.post("/ingest/document", json=_valid_document_payload())

        assert response.status_code == 422
        body = response.json()
        assert body["invalid_fields"] == ["service_name"]
        assert body["detail"] == "Metadata validation failed"


# TestIngestChunksEndpoint removed - /ingest/chunks endpoint removed per narrow-service-spec


class TestLegacyRoutesStillWork:
    """Ensure existing upload routes are unaffected by ingest additions."""

    def test_upload_v1_still_works(self, client):
        response = client.post(
            "/upload_v1",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 200

    def test_upload_v2_still_works(self, client):
        response = client.post(
            "/upload_v2",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 200
