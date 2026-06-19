# import sys
# from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth import ApiKeyRegistry
from app.main import create_app

# Shared test API key with an admin wildcard grant, so existing tests that send
# valid scopes pass scope enforcement unchanged.
TEST_API_KEY = "test-key"
TEST_AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


def build_test_registry() -> ApiKeyRegistry:
    return ApiKeyRegistry.from_entries(
        [
            {
                "key": TEST_API_KEY,
                "key_id": "test-admin",
                "services": ["*"],
                "tenants": ["*"],
                "collections": ["*"],
                "admin": True,
            }
        ]
    )

# sys.path.append(str(Path(__file__).resolve().parents[1]))


class FakeDocumentService:
    def __init__(self) -> None:
        self.calls = []
        self.contexts = []
        self.jobs = {}

    async def create_document(self, file):
        self.calls.append(("create_document", file.filename))
        return {"document_id": "doc-123"}

    def extract_text(self, document_id):
        self.calls.append(("extract_text", document_id))
        return {"document_id": document_id, "status": "text_extracted"}

    def chunk_document(self, document_id):
        self.calls.append(("chunk_document", document_id))
        return {"document_id": document_id, "status": "chunked", "chunk_count": 2}

    def embed_document(self, document_id):
        self.calls.append(("embed_document", document_id))
        return {"document_id": document_id, "status": "embedded", "embedding_count": 2}

    def retrieve_context_dense(self, query, limit):
        self.calls.append(("retrieve_context_dense", query, limit))
        return self.contexts

    def retrieve_context_lexical(self, query, limit):
        self.calls.append(("retrieve_context_lexical", query, limit))
        return self.contexts

    def retrieve_context_hybrid(self, query, limit):
        self.calls.append(("retrieve_context_hybrid", query, limit))
        return self.contexts

    def get_document(self, document_id):
        self.calls.append(("get_document", document_id))
        return {
            "document_id": document_id,
            "original_filename": "python_rag_intro.txt",
            "stored_filename": "doc-123.txt",
            "status": "embedded",
            "size_bytes": 123,
            "extracted_text": "some extracted text",
            "chunks": ["chunk one", "chunk two"],
        }



    def create_job(self, document_id: str):
        self.calls.append(("create_job", document_id))
        fake_job = {
            "job_id": "job-123",
            "job_type": "document_index",
            "document_id": document_id,
            "status": "queued",
            "error_message": None,
            "created_at": "2026-05-23T12:00:00",
            "started_at": None,
            "finished_at": None,
        }
        self.jobs[fake_job["job_id"]] = fake_job
        return fake_job

    def get_job(self, job_id: str):
        self.calls.append(("get_job", job_id))
        return self.jobs[job_id]

    def mark_job_running(self, job_id: str):
        self.calls.append(("mark_job_running", job_id))
        fake_job = self.jobs[job_id]
        fake_job["status"] = "running"
        return fake_job

    def mark_job_completed(self, job_id: str):
        self.calls.append(("mark_job_completed", job_id))
        fake_job = self.jobs[job_id]
        fake_job["status"] = "completed"
        fake_job["finished_at"] = "2026-05-25T12:00:00"
        fake_job["error_message"] = ""
        return fake_job

    def mark_job_failed(self, job_id: str, message: str):
        self.calls.append(("mark_job_failed", job_id, message))
        fake_job = self.jobs[job_id]
        fake_job["status"] = "failed"
        fake_job["finished_at"] = "2026-05-25T12:00:00"
        fake_job["error_message"] = message
        return fake_job

    def run_indexing_pipeline(self, document_id: str, job_id: str):
        self.calls.append(("run_indexing_pipeline", document_id, job_id))
        self.mark_job_running(job_id)
        self.extract_text(document_id)
        self.chunk_document(document_id)
        self.embed_document(document_id)
        self.mark_job_completed(job_id)

        return self.jobs[job_id]


class FakeGenerationService:
    def __init__(self, answer="", error=None) -> None:
        self.answer = answer
        self.error = error
        self.streamed_tokens = []
        self.stream_error = None
        self.calls = []

    def answer_question(self, question, sources):
        self.calls.append(("answer_question", question, sources))

        if self.error is not None:
            raise self.error

        return self.answer

    async def stream_answer_question(self, question, sources):
        self.calls.append(("stream_answer_question", question, sources))

        if self.stream_error is not None:
            raise self.stream_error

        for token in self.streamed_tokens:
            yield token


@pytest.fixture
def fake_document_service():
    return FakeDocumentService()


@pytest.fixture
def fake_generation_service():
    return FakeGenerationService()


@pytest.fixture
def api_key_registry():
    return build_test_registry()


@pytest.fixture
def auth_headers():
    return dict(TEST_AUTH_HEADERS)


@pytest.fixture
def client(fake_generation_service, api_key_registry):
    app = create_app(
        generation_service=fake_generation_service,
        api_key_registry=api_key_registry,
    )
    return TestClient(app, headers=TEST_AUTH_HEADERS)
