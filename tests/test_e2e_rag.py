"""End-to-end tests for the narrowed RAG service.

These tests validate the full request → response flow for the narrowed API surface.
"""
import io
import pytest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.composition import MetadataAwareRuntime
from app.ingest.use_case import IngestResult
from app.main import create_app
from app.retrieval.types import RetrievedChunk, RetrievalMode
from app.retrieval.use_case import RetrieveResult


class TestManualUploadRetrieveAnswerFlow:
    """Test the full flow: upload file → retrieve → answer."""

    @pytest.fixture
    def mock_runtime(self):
        """Runtime with mocked services that track the full flow."""
        ingest_use_case = Mock()
        ingest_use_case.ingest_document.return_value = IngestResult(chunk_count=2)

        retrieve_use_case = Mock()
        retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="Python is a programming language.",
                    score=0.95,
                    rank=1,
                    retrieval_mode=RetrievalMode.HYBRID,
                    metadata={
                        "service_name": "manual",
                        "tenant_id": "local",
                        "collection": "general",
                        "source_label": "test.txt",
                        "chunk_index": 0,
                        "document_id": "doc-1",
                    },
                )
            ],
            warnings=[],
            trace_id="trace-123",
        )

        return MetadataAwareRuntime(
            retrieve_use_case=retrieve_use_case,
            ingest_use_case=ingest_use_case,
            gateway=Mock(),
        )

    @pytest.fixture
    def generation_service(self):
        service = Mock()
        service.answer_question.return_value = "Python is a high-level programming language."
        return service

    @pytest.fixture
    def client(self, mock_runtime, generation_service, api_key_registry, auth_headers):
        app = create_app(
            generation_service=generation_service,
            metadata_aware_runtime=mock_runtime,
            api_key_registry=api_key_registry,
        )
        return TestClient(app, headers=auth_headers)

    def test_upload_retrieve_answer_flow(self, client, mock_runtime, generation_service):
        """Test the full flow from upload to answer generation."""
        # Step 1: Upload a file
        file_content = b"Python is a programming language used for data science."
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
        upload_response = client.post("/documents/upload", files=files)

        assert upload_response.status_code == 200
        assert upload_response.json()["chunk_count"] == 2
        mock_runtime.ingest_use_case.ingest_document.assert_called_once()

        # Verify default metadata was applied during upload
        call_args = mock_runtime.ingest_use_case.ingest_document.call_args
        document = call_args[0][0]
        assert document.service_name == "manual"
        assert document.tenant_id == "local"
        assert document.collection == "general"

        # Step 2: Retrieve chunks
        retrieve_payload = {
            "query": "what is python",
            "service_name": "manual",
            "tenant_id": "local",
            "collections": ["general"],
        }
        retrieve_response = client.post("/retrieve", json=retrieve_payload)

        assert retrieve_response.status_code == 200
        assert len(retrieve_response.json()["chunks"]) == 1
        assert "Python" in retrieve_response.json()["chunks"][0]["text"]

        # Step 3: Generate answer
        answer_payload = {
            "query": "what is python",
            "service_name": "manual",
            "tenant_id": "local",
            "collections": ["general"],
        }
        answer_response = client.post("/answer", json=answer_payload)

        assert answer_response.status_code == 200
        body = answer_response.json()
        assert "Python" in body["answer"]
        assert len(body["sources"]) == 1
        generation_service.answer_question.assert_called_once()


class TestServiceIngestScopedRetrieveAnswerFlow:
    """Test service document ingestion with metadata-scoped retrieval."""

    @pytest.fixture
    def mock_runtime(self):
        """Runtime with scoped retrieval behavior."""
        ingest_use_case = Mock()
        ingest_use_case.ingest_document.return_value = IngestResult(chunk_count=3)

        retrieve_use_case = Mock()
        retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-service-1",
                    document_id="doc-service-1",
                    content="Service-specific content about APIs.",
                    score=0.92,
                    rank=1,
                    retrieval_mode=RetrievalMode.HYBRID,
                    metadata={
                        "service_name": "api-service",
                        "tenant_id": "tenant-456",
                        "collection": "documentation",
                        "source_label": "api-docs.pdf",
                        "chunk_index": 0,
                        "document_id": "doc-service-1",
                    },
                )
            ],
            warnings=[],
            trace_id="trace-456",
        )

        return MetadataAwareRuntime(
            retrieve_use_case=retrieve_use_case,
            ingest_use_case=ingest_use_case,
            gateway=Mock(),
        )

    @pytest.fixture
    def generation_service(self):
        service = Mock()
        service.answer_question.return_value = "The API service provides REST endpoints."
        return service

    @pytest.fixture
    def client(self, mock_runtime, generation_service, api_key_registry, auth_headers):
        app = create_app(
            generation_service=generation_service,
            metadata_aware_runtime=mock_runtime,
            api_key_registry=api_key_registry,
        )
        return TestClient(app, headers=auth_headers)

    def test_service_ingest_scoped_retrieve_answer(self, client, mock_runtime, generation_service):
        """Test service ingestion with metadata-scoped retrieval and answer."""
        # Step 1: Service ingest with explicit metadata
        ingest_payload = {
            "text": "The API service provides REST endpoints for data access.",
            "service_name": "api-service",
            "tenant_id": "tenant-456",
            "collection": "documentation",
            "source_type": "pdf",
            "source_label": "api-docs.pdf",
        }
        ingest_response = client.post("/documents/ingest", json=ingest_payload)

        assert ingest_response.status_code == 200
        assert ingest_response.json()["chunk_count"] == 3

        # Step 2: Scoped retrieval (same service, tenant, collection)
        retrieve_payload = {
            "query": "how do I use the API",
            "service_name": "api-service",
            "tenant_id": "tenant-456",
            "collections": ["documentation"],
        }
        retrieve_response = client.post("/retrieve", json=retrieve_payload)

        assert retrieve_response.status_code == 200
        chunks = retrieve_response.json()["chunks"]
        assert len(chunks) == 1
        assert chunks[0]["service_name"] == "api-service"
        assert chunks[0]["tenant_id"] == "tenant-456"
        assert chunks[0]["collection"] == "documentation"

        # Step 3: Generate scoped answer
        answer_response = client.post("/answer", json=retrieve_payload)

        assert answer_response.status_code == 200
        body = answer_response.json()
        assert "API service" in body["answer"]
        assert len(body["sources"]) == 1


class TestMetadataIsolation:
    """Test that metadata scoping prevents cross-service data leakage."""

    @pytest.fixture
    def mock_runtime(self):
        """Runtime that enforces metadata isolation."""
        ingest_use_case = Mock()
        ingest_use_case.ingest_document.return_value = IngestResult(chunk_count=1)

        retrieve_use_case = Mock()

        def scoped_retrieve(request):
            """Return chunks only matching the requested scope."""
            # Simulate service A data
            if request.scope.service_name == "service-a":
                return RetrieveResult(
                    chunks=[
                        RetrievedChunk(
                            chunk_id="chunk-a",
                            document_id="doc-a",
                            content="Service A confidential data.",
                            score=0.9,
                            rank=1,
                            retrieval_mode=RetrievalMode.HYBRID,
                            metadata={
                                "service_name": "service-a",
                                "tenant_id": "tenant-1",
                                "collection": "data",
                                "source_label": "a.txt",
                                "chunk_index": 0,
                                "document_id": "doc-a",
                            },
                        )
                    ],
                    warnings=[],
                    trace_id="trace-a",
                )
            # Service B should get empty results when querying
            elif request.scope.service_name == "service-b":
                return RetrieveResult(chunks=[], warnings=[], trace_id="trace-b")
            else:
                return RetrieveResult(chunks=[], warnings=[], trace_id="trace-other")

        retrieve_use_case.execute.side_effect = scoped_retrieve

        return MetadataAwareRuntime(
            retrieve_use_case=retrieve_use_case,
            ingest_use_case=ingest_use_case,
            gateway=Mock(),
        )

    @pytest.fixture
    def client(self, mock_runtime, api_key_registry, auth_headers):
        app = create_app(
            generation_service=Mock(),
            metadata_aware_runtime=mock_runtime,
            api_key_registry=api_key_registry,
        )
        return TestClient(app, headers=auth_headers)

    def test_service_a_cannot_access_service_b_data(self, client):
        """Test that service B cannot retrieve service A's data."""
        # Ingest document for service A
        ingest_a = {
            "text": "Service A confidential data.",
            "service_name": "service-a",
            "tenant_id": "tenant-1",
            "collection": "data",
            "source_type": "text",
            "source_label": "a.txt",
        }
        client.post("/documents/ingest", json=ingest_a)

        # Service A can retrieve its own data
        retrieve_a = {
            "query": "confidential",
            "service_name": "service-a",
            "tenant_id": "tenant-1",
            "collections": ["data"],
        }
        response_a = client.post("/retrieve", json=retrieve_a)
        assert response_a.status_code == 200
        assert len(response_a.json()["chunks"]) == 1

        # Service B cannot retrieve service A's data (metadata isolation)
        retrieve_b = {
            "query": "confidential",
            "service_name": "service-b",
            "tenant_id": "tenant-2",
            "collections": ["data"],
        }
        response_b = client.post("/retrieve", json=retrieve_b)
        assert response_b.status_code == 200
        assert len(response_b.json()["chunks"]) == 0  # Metadata isolation enforced


class TestStreamingAnswer:
    """Test streaming answer generation via SSE."""

    @pytest.fixture
    def mock_runtime(self):
        retrieve_use_case = Mock()
        retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="Streaming content.",
                    score=0.9,
                    rank=1,
                    retrieval_mode=RetrievalMode.HYBRID,
                    metadata={
                        "service_name": "test",
                        "tenant_id": "test",
                        "collection": "test",
                        "source_label": "test.txt",
                        "chunk_index": 0,
                        "document_id": "doc-1",
                    },
                )
            ],
            warnings=[],
            trace_id="trace-stream",
        )

        return MetadataAwareRuntime(
            retrieve_use_case=retrieve_use_case,
            ingest_use_case=Mock(),
            gateway=Mock(),
        )

    @pytest.fixture
    def generation_service(self):
        service = Mock()

        async def stream_tokens(q, s):
            for token in ["Streaming ", "answer ", "content."]:
                yield token

        service.stream_answer_question = stream_tokens
        return service

    @pytest.fixture
    def client(self, mock_runtime, generation_service, api_key_registry, auth_headers):
        app = create_app(
            generation_service=generation_service,
            metadata_aware_runtime=mock_runtime,
            api_key_registry=api_key_registry,
        )
        return TestClient(app, headers=auth_headers)

    def test_streaming_answer_emits_content_and_done(self, client):
        """Test that streaming answer emits SSE events with content and completion signal."""
        payload = {
            "query": "test query",
            "service_name": "test",
            "tenant_id": "test",
            "collections": ["test"],
        }

        response = client.post("/answer/stream", json=payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE events
        events = []
        for line in response.text.split("\n\n"):
            if line.startswith("data: "):
                import json

                event_data = json.loads(line.removeprefix("data: "))
                events.append(event_data)

        # Should have content events + done event
        assert len(events) > 1
        assert events[-1]["done"] is True
        assert events[-1]["event"] == "done"

        # Content events should have done=False
        for event in events[:-1]:
            assert event["done"] is False
            assert event["event"] == "content"
            assert len(event["data"]) > 0  # Has content
