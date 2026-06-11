import pytest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.composition import MetadataAwareRuntime
from app.main import create_app
from app.retrieval.types import RetrievedChunk, RetrievalMode
from app.retrieval.use_case import RetrieveResult
from app.services.generation_service import GenerationServiceError


@pytest.fixture
def mock_retrieve_use_case():
    use_case = Mock()
    use_case.execute.return_value = RetrieveResult(
        chunks=[
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="This is retrieved content for answer generation.",
                score=0.95,
                rank=1,
                retrieval_mode=RetrievalMode.HYBRID,
                metadata={
                    "service_name": "test-service",
                    "tenant_id": "tenant-123",
                    "collection": "documents",
                    "source_label": "test.pdf",
                    "chunk_index": 0,
                    "document_id": "doc-1",
                },
            )
        ],
        warnings=[],
        trace_id="trace-123",
    )
    return use_case


@pytest.fixture
def mock_generation_service():
    service = Mock()
    service.answer_question.return_value = "This is a generated answer based on the context."

    # Mock async streaming - return an async generator directly
    async def mock_stream(q, s):
        for token in ["This ", "is ", "a ", "streamed ", "answer."]:
            yield token

    service.stream_answer_question = mock_stream
    return service


@pytest.fixture
def client(mock_retrieve_use_case, mock_generation_service):
    mock_runtime = MetadataAwareRuntime(
        retrieve_use_case=mock_retrieve_use_case,
        ingest_use_case=Mock(),
        gateway=Mock(),
    )

    app = create_app(
        generation_service=mock_generation_service,
        metadata_aware_runtime=mock_runtime,
    )
    return TestClient(app)


class TestAnswerEndpoint:
    """Tests for POST /answer."""

    def test_successful_answer_returns_answer_and_sources(
        self, client, mock_retrieve_use_case, mock_generation_service
    ):
        payload = {
            "query": "test question",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collections": ["documents"],
        }

        response = client.post("/answer", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "This is a generated answer based on the context."
        assert body["trace_id"] == "trace-123"
        assert len(body["sources"]) == 1
        assert body["sources"][0]["text"] == "This is retrieved content for answer generation."
        assert body["sources"][0]["chunk_id"] == "chunk-1"
        mock_generation_service.answer_question.assert_called_once()

    def test_empty_retrieval_returns_fallback_answer(
        self, client, mock_retrieve_use_case, mock_generation_service
    ):
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[],
            warnings=[],
            trace_id="trace-456",
        )
        payload = {
            "query": "no results question",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collections": ["documents"],
        }

        response = client.post("/answer", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert "couldn't find" in body["answer"].lower()
        assert len(body["sources"]) == 0
        mock_generation_service.answer_question.assert_not_called()

    def test_generation_error_returns_500(
        self, client, mock_retrieve_use_case, mock_generation_service
    ):
        mock_generation_service.answer_question.side_effect = GenerationServiceError(
            "Generation failed"
        )
        payload = {
            "query": "test question",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collections": ["documents"],
        }

        response = client.post("/answer", json=payload)

        assert response.status_code == 500
        assert "generation failed" in response.json()["detail"].lower()

    def test_missing_required_fields_returns_422(self, client):
        payload = {
            "query": "test question",
            # Missing service_name, tenant_id, collections
        }

        response = client.post("/answer", json=payload)

        assert response.status_code == 422


class TestAnswerStreamEndpoint:
    """Tests for POST /answer/stream."""

    def test_successful_stream_emits_content_and_done(
        self, client, mock_retrieve_use_case, mock_generation_service
    ):
        payload = {
            "query": "test question",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collections": ["documents"],
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

    def test_empty_retrieval_streams_fallback_message(
        self, client, mock_retrieve_use_case
    ):
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[],
            warnings=[],
            trace_id="trace-456",
        )
        payload = {
            "query": "no results question",
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collections": ["documents"],
        }

        response = client.post("/answer/stream", json=payload)

        assert response.status_code == 200

        # Parse SSE events
        events = []
        for line in response.text.split("\n\n"):
            if line.startswith("data: "):
                import json

                event_data = json.loads(line.removeprefix("data: "))
                events.append(event_data)

        # Should have fallback content + done event
        assert len(events) == 2
        assert events[0]["event"] == "content"
        assert "couldn't find" in events[0]["data"].lower()
        assert events[1]["event"] == "done"
        assert events[1]["done"] is True
