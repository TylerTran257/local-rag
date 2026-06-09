from fastapi.testclient import TestClient

from app.api.retrieval_helpers import map_retrieval_error_to_response
from app.main import create_app
from app.retrieval import (
    RetrievedChunk,
    RetrievalMode,
    InvalidRetrievalRequestError,
    NoIndexedCorpusError,
    RetrievedChunkValidationError,
    UnsupportedRetrievalModeError,
)
from app.retrieval.use_case import RetrieveResult


def make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc-1:0",
        document_id="doc-1",
        content="chunk text",
        score=0.95,
        rank=1,
        retrieval_mode=RetrievalMode.DENSE,
        metadata={
            "service_name": "local-rag",
            "tenant_id": "default",
            "collection": "documents",
            "source_type": "document",
            "source_label": "file.pdf",
            "document_id": "doc-1",
            "chunk_index": 0,
        },
    )


class FakeRetrieveUseCase:
    def __init__(self, *, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def execute(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.result


def make_client(fake_document_service, fake_generation_service, retrieve_use_case):
    app = create_app(
        document_service=fake_document_service,
        generation_service=fake_generation_service,
        retrieve_use_case=retrieve_use_case,
    )
    return TestClient(app)


def test_semantic_search_preserves_success_response_shape(fake_document_service, fake_generation_service):
    retrieve_use_case = FakeRetrieveUseCase(
        result=RetrieveResult(chunks=[make_chunk()], warnings=[])
    )
    client = make_client(fake_document_service, fake_generation_service, retrieve_use_case)

    response = client.post("/semantic-search", json={"query": "test query", "limit": 3})

    assert response.status_code == 200
    assert response.json() == {
        "query": "test query",
        "match_count": 1,
        "results": [
            {
                "document_id": "doc-1",
                "original_filename": "file.pdf",
                "chunk_index": 0,
                "score": 0.95,
                "text": "chunk text",
            }
        ],
    }
    assert retrieve_use_case.calls[0].retrieval_mode == RetrievalMode.DENSE


def test_hybrid_search_preserves_success_response_shape(fake_document_service, fake_generation_service):
    chunk = make_chunk()
    chunk.retrieval_mode = RetrievalMode.HYBRID
    retrieve_use_case = FakeRetrieveUseCase(
        result=RetrieveResult(chunks=[chunk], warnings=[])
    )
    client = make_client(fake_document_service, fake_generation_service, retrieve_use_case)

    response = client.post("/hybrid-search", json={"query": "test query", "limit": 3})

    assert response.status_code == 200
    assert response.json() == {
        "query": "test query",
        "match_count": 1,
        "results": [
            {
                "document_id": "doc-1",
                "original_filename": "file.pdf",
                "chunk_index": 0,
                "score": 0.95,
                "text": "chunk text",
            }
        ],
    }
    assert retrieve_use_case.calls[0].retrieval_mode == RetrievalMode.HYBRID


def test_semantic_search_returns_safe_invalid_request_error(fake_document_service, fake_generation_service):
    retrieve_use_case = FakeRetrieveUseCase(
        error=InvalidRetrievalRequestError(
            trace_id="trace-123",
            internal_message="query had leading spaces and leaked details",
            details={"query": " bad "},
        )
    )
    client = make_client(fake_document_service, fake_generation_service, retrieve_use_case)

    response = client.post("/semantic-search", json={"query": "test query", "limit": 3})

    assert response.status_code == 422
    assert response.json() == {
        "code": "INVALID_RETRIEVAL_REQUEST",
        "message": "Invalid retrieval request",
        "trace_id": "trace-123",
    }
    assert "detail" not in response.json()


def test_hybrid_search_returns_safe_no_indexed_corpus_error(fake_document_service, fake_generation_service):
    retrieve_use_case = FakeRetrieveUseCase(
        error=NoIndexedCorpusError(
            trace_id="trace-409",
            internal_message="backend mentioned qdrant internals",
            details={"backend": "qdrant"},
        )
    )
    client = make_client(fake_document_service, fake_generation_service, retrieve_use_case)

    response = client.post("/hybrid-search", json={"query": "test query", "limit": 3})

    assert response.status_code == 409
    assert response.json() == {
        "code": "NO_INDEXED_CORPUS",
        "message": "No indexed corpus available",
        "trace_id": "trace-409",
    }


def test_search_routes_hide_internal_chunk_validation_error_details(fake_document_service, fake_generation_service):
    retrieve_use_case = FakeRetrieveUseCase(
        error=RetrievedChunkValidationError(
            trace_id="trace-500",
            internal_message="tenant mismatch on chunk metadata",
            details={"tenant_id": "wrong-tenant"},
        )
    )
    client = make_client(fake_document_service, fake_generation_service, retrieve_use_case)

    response = client.post("/semantic-search", json={"query": "test query", "limit": 3})

    assert response.status_code == 500
    assert response.json() == {
        "code": "RETRIEVED_CHUNK_VALIDATION_ERROR",
        "message": "Retrieval failed",
        "trace_id": "trace-500",
    }
    assert "tenant mismatch" not in response.text
    assert "wrong-tenant" not in response.text


def test_response_helper_maps_supported_but_unimplemented_modes_to_501():
    response = map_retrieval_error_to_response(
        UnsupportedRetrievalModeError(
            trace_id="trace-501",
            internal_message="hybrid mode unsupported by adapter",
            details={"mode": "hybrid"},
        )
    )

    assert response.status_code == 501
    assert response.body == (
        b'{"code":"UNSUPPORTED_RETRIEVAL_MODE","message":"Unsupported retrieval mode","trace_id":"trace-501"}'
    )


def test_response_helper_maps_invalid_modes_to_400():
    response = map_retrieval_error_to_response(
        UnsupportedRetrievalModeError(
            trace_id="trace-400",
            internal_message="mode totally-invalid is not allowed",
            details={"mode": "totally-invalid"},
        )
    )

    assert response.status_code == 400
    assert response.body == (
        b'{"code":"UNSUPPORTED_RETRIEVAL_MODE","message":"Unsupported retrieval mode","trace_id":"trace-400"}'
    )
