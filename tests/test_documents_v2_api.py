import io
import pytest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.composition import MetadataAwareRuntime
from app.ingest.contracts import MetadataValidationError
from app.ingest.use_case import IngestResult
from app.main import create_app


@pytest.fixture
def mock_ingest_use_case():
    use_case = Mock()
    use_case.ingest_document.return_value = IngestResult(chunk_count=3)
    return use_case


@pytest.fixture
def fake_document_service():
    from app.services.document_service import DocumentService
    return Mock(spec=DocumentService)


@pytest.fixture
def fake_generation_service():
    from app.services.generation_service import GenerationService
    return Mock(spec=GenerationService)


@pytest.fixture
def client(mock_ingest_use_case, fake_generation_service):
    mock_runtime = MetadataAwareRuntime(
        retrieve_use_case=Mock(),
        ingest_use_case=mock_ingest_use_case,
        gateway=Mock(),
    )

    app = create_app(
        generation_service=fake_generation_service,
        metadata_aware_runtime=mock_runtime,
    )
    return TestClient(app)


class TestDocumentUploadEndpoint:
    """Tests for POST /documents/upload."""

    def test_valid_txt_upload_returns_200_with_chunk_count(self, client, mock_ingest_use_case):
        file_content = b"This is a test document for upload."
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}

        response = client.post("/documents/upload", files=files)

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["chunk_count"] == 3
        assert body["source_label"] == "test.txt"
        mock_ingest_use_case.ingest_document.assert_called_once()

        # Verify default metadata was applied
        call_args = mock_ingest_use_case.ingest_document.call_args
        document = call_args[0][0]
        assert document.service_name == "manual"
        assert document.tenant_id == "local"
        assert document.collection == "general"
        assert document.source_type == "uploaded_file"
        assert document.source_label == "test.txt"
        assert document.domain_metadata == {}

    def test_multiple_file_types_supported(self, client, mock_ingest_use_case):
        # Test txt file works (already tested above, this just confirms the flow)
        file_content = b"Another test document."
        files = {"file": ("another.txt", io.BytesIO(file_content), "text/plain")}

        response = client.post("/documents/upload", files=files)

        assert response.status_code == 200
        assert response.json()["source_label"] == "another.txt"

    def test_metadata_validation_error_returns_422(self, client, mock_ingest_use_case):
        mock_ingest_use_case.ingest_document.side_effect = MetadataValidationError(
            invalid_fields=["service_name"],
            metadata={"service_name": ""},
        )
        file_content = b"Test content"
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}

        response = client.post("/documents/upload", files=files)

        assert response.status_code == 422
        body = response.json()
        assert body["invalid_fields"] == ["service_name"]

    def test_unsupported_file_type_returns_422(self, client):
        file_content = b"binary data"
        files = {"file": ("test.docx", io.BytesIO(file_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}

        response = client.post("/documents/upload", files=files)

        assert response.status_code == 422
        body = response.json()
        assert "Unsupported file type" in body["detail"]

    def test_empty_file_returns_422(self, client, mock_ingest_use_case):
        from app.ingest.contracts import EmptyDocumentError

        mock_ingest_use_case.ingest_document.side_effect = EmptyDocumentError(
            source_label="empty.txt"
        )
        files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}

        response = client.post("/documents/upload", files=files)

        assert response.status_code == 422
        assert "no text to ingest" in response.json()["detail"]


class TestDocumentIngestEndpoint:
    """Tests for POST /documents/ingest."""

    def test_valid_ingest_returns_200_with_chunk_count(self, client, mock_ingest_use_case):
        payload = {
            "text": "Some document text to ingest.",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collection": "documents",
            "source_type": "pdf",
            "source_label": "test.pdf",
        }

        response = client.post("/documents/ingest", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["chunk_count"] == 3
        mock_ingest_use_case.ingest_document.assert_called_once()

    def test_missing_required_field_returns_422(self, client):
        payload = {
            "text": "Some document text.",
            "tenant_id": "tenant-123",
            "collection": "documents",
            "source_type": "pdf",
            "source_label": "test.pdf",
            # Missing service_name
        }

        response = client.post("/documents/ingest", json=payload)

        assert response.status_code == 422

    def test_empty_text_returns_422(self, client):
        payload = {
            "text": "",  # Empty text
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collection": "documents",
            "source_type": "pdf",
            "source_label": "test.pdf",
        }

        response = client.post("/documents/ingest", json=payload)

        assert response.status_code == 422

    def test_whitespace_only_text_returns_422(self, client, mock_ingest_use_case):
        from app.ingest.contracts import EmptyDocumentError

        mock_ingest_use_case.ingest_document.side_effect = EmptyDocumentError(
            source_label="test.pdf"
        )
        payload = {
            "text": "   \n\t  ",  # Whitespace-only passes schema, rejected by use case
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collection": "documents",
            "source_type": "pdf",
            "source_label": "test.pdf",
        }

        response = client.post("/documents/ingest", json=payload)

        assert response.status_code == 422
        assert "no text to ingest" in response.json()["detail"]

    def test_domain_metadata_is_accepted(self, client, mock_ingest_use_case):
        payload = {
            "text": "Document text.",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collection": "documents",
            "source_type": "pdf",
            "source_label": "test.pdf",
            "domain_metadata": {"author": "Jane Doe", "category": "research"},
        }

        response = client.post("/documents/ingest", json=payload)

        assert response.status_code == 200
        call_args = mock_ingest_use_case.ingest_document.call_args
        document = call_args[0][0]
        assert document.domain_metadata == {"author": "Jane Doe", "category": "research"}

    def test_metadata_validation_error_returns_422(self, client, mock_ingest_use_case):
        mock_ingest_use_case.ingest_document.side_effect = MetadataValidationError(
            invalid_fields=["tenant_id"],
            metadata={"tenant_id": ""},
        )
        payload = {
            "text": "Document text.",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collection": "documents",
            "source_type": "pdf",
            "source_label": "test.pdf",
        }

        response = client.post("/documents/ingest", json=payload)

        assert response.status_code == 422
        body = response.json()
        assert body["invalid_fields"] == ["tenant_id"]
