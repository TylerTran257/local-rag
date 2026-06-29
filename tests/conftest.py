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
