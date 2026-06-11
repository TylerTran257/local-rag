"""Tests for the social style retrieval API endpoint."""
from fastapi.testclient import TestClient

from app.main import create_app
from app.retrieval.types import (
    RetrievalMode,
    RetrievedChunk,
    RetrievalWarning,
    WarningCode,
    WarningSeverity,
)
from app.retrieval.use_case import RetrieveResult
from app.social import SocialStyleRetrievalService


class FakeRetrieveUseCase:
    """Fake retrieve use case that returns predictable results."""

    def __init__(self, results_by_call=None):
        self.results_by_call = results_by_call or []
        self.call_index = 0
        self.calls = []

    def execute(self, request):
        self.calls.append(request)
        if self.call_index < len(self.results_by_call):
            result = self.results_by_call[self.call_index]
            self.call_index += 1
            return result
        return RetrieveResult(chunks=[], warnings=[], trace_id=None)


def _make_chunk(
    content: str,
    category: str,
    source_label: str = "guide.pdf",
    document_id: str = "doc-1",
    score: float = 0.95,
    rank: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{document_id}:{rank}",
        document_id=document_id,
        content=content,
        score=score,
        rank=rank,
        retrieval_mode=RetrievalMode.HYBRID,
        metadata={
            "style_category": category,
            "source_label": source_label,
            "service_name": "social-style",
            "tenant_id": "tenant-1",
            "collection": "style_memory",
            "source_type": "style_memory",
        },
    )


def _make_client(fake_use_case) -> TestClient:
    service = SocialStyleRetrievalService(
        retrieve_use_case=fake_use_case,
        service_name="social-style",
    )
    app = create_app(
        document_service=_FakeDocService(),
        generation_service=_FakeGenService(),
    )
    app.state.social_style_service = service
    return TestClient(app)


class _FakeDocService:
    """Minimal fake to satisfy create_app."""
    pass


class _FakeGenService:
    """Minimal fake to satisfy create_app."""
    pass


class TestSocialStyleRetrieveEndpoint:
    """Tests for POST /social-style/retrieve."""

    def test_valid_request_returns_200_with_grouped_entries(self):
        """Valid request returns 200 with entries grouped by category."""
        fake_use_case = FakeRetrieveUseCase(results_by_call=[
            RetrieveResult(
                chunks=[_make_chunk("Use professional tone", "voice_rules")],
                warnings=[],
                trace_id="trace-1",
            ),
            RetrieveResult(
                chunks=[_make_chunk("Start with a question", "hook_patterns", source_label="hooks.pdf", document_id="doc-2")],
                warnings=[],
                trace_id="trace-2",
            ),
        ])
        client = _make_client(fake_use_case)

        response = client.post("/social-style/retrieve", json={
            "tenant_id": "tenant-1",
            "query": "professional tone",
            "style_categories": ["voice_rules", "hook_patterns"],
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data["voice_rules"]) == 1
        assert data["voice_rules"][0]["content"] == "Use professional tone"
        assert data["voice_rules"][0]["score"] == 0.95
        assert len(data["hook_patterns"]) == 1
        assert data["hook_patterns"][0]["content"] == "Start with a question"

    def test_response_includes_missing_categories(self):
        """Categories with no results appear in missing_categories."""
        fake_use_case = FakeRetrieveUseCase(results_by_call=[
            RetrieveResult(
                chunks=[_make_chunk("Professional tone", "voice_rules")],
                warnings=[],
                trace_id="trace-1",
            ),
            RetrieveResult(
                chunks=[],
                warnings=[],
                trace_id="trace-2",
            ),
        ])
        client = _make_client(fake_use_case)

        response = client.post("/social-style/retrieve", json={
            "tenant_id": "tenant-1",
            "query": "test",
            "style_categories": ["voice_rules", "hook_patterns"],
        })

        assert response.status_code == 200
        data = response.json()
        assert "hook_patterns" in data["missing_categories"]
        assert "voice_rules" not in data["missing_categories"]

    def test_response_includes_trace_ids(self):
        """Trace IDs from all category retrievals are included in response."""
        fake_use_case = FakeRetrieveUseCase(results_by_call=[
            RetrieveResult(chunks=[], warnings=[], trace_id="trace-aaa"),
            RetrieveResult(chunks=[], warnings=[], trace_id="trace-bbb"),
            RetrieveResult(chunks=[], warnings=[], trace_id="trace-ccc"),
        ])
        client = _make_client(fake_use_case)

        response = client.post("/social-style/retrieve", json={
            "tenant_id": "tenant-1",
            "query": "test",
            "style_categories": ["voice_rules", "hook_patterns", "cta_patterns"],
        })

        assert response.status_code == 200
        data = response.json()
        assert data["trace_ids"] == ["trace-aaa", "trace-bbb", "trace-ccc"]

    def test_invalid_style_category_returns_422(self):
        """Invalid style category value returns 422."""
        fake_use_case = FakeRetrieveUseCase()
        client = _make_client(fake_use_case)

        response = client.post("/social-style/retrieve", json={
            "tenant_id": "tenant-1",
            "query": "test",
            "style_categories": ["voice_rules", "invalid_category"],
        })

        assert response.status_code == 422
        data = response.json()
        assert "invalid_category" in data["detail"]

    def test_optional_fields_accepted(self):
        """Optional fields (platform, per_category_limit, collection) are accepted."""
        fake_use_case = FakeRetrieveUseCase(results_by_call=[
            RetrieveResult(chunks=[], warnings=[], trace_id="trace-1"),
        ])
        client = _make_client(fake_use_case)

        response = client.post("/social-style/retrieve", json={
            "tenant_id": "tenant-1",
            "query": "test",
            "style_categories": ["voice_rules"],
            "platform": "twitter",
            "per_category_limit": 10,
            "collection": "custom_collection",
        })

        assert response.status_code == 200

        # Verify the use case received the correct parameters
        assert len(fake_use_case.calls) == 1
        call = fake_use_case.calls[0]
        assert call.scope.filters["platform"] == "twitter"
        assert call.limit == 10
        assert call.scope.collections == ["custom_collection"]

    def test_source_references_included(self):
        """Source references are included in the response."""
        fake_use_case = FakeRetrieveUseCase(results_by_call=[
            RetrieveResult(
                chunks=[
                    _make_chunk("Content 1", "voice_rules", source_label="guide1.pdf", document_id="doc-1"),
                    _make_chunk("Content 2", "voice_rules", source_label="guide2.pdf", document_id="doc-2", rank=1),
                ],
                warnings=[],
                trace_id="trace-1",
            ),
        ])
        client = _make_client(fake_use_case)

        response = client.post("/social-style/retrieve", json={
            "tenant_id": "tenant-1",
            "query": "test",
            "style_categories": ["voice_rules"],
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data["source_references"]) == 2
        assert data["source_references"][0]["document_id"] == "doc-1"
        assert data["source_references"][1]["document_id"] == "doc-2"

    def test_warnings_included_in_response(self):
        """Retrieval warnings are included in the response."""
        warning = RetrievalWarning(
            code=WarningCode.EMPTY_RETRIEVAL_RESULT,
            severity=WarningSeverity.MEDIUM,
            source="voice_rules",
            message="No results found",
        )
        fake_use_case = FakeRetrieveUseCase(results_by_call=[
            RetrieveResult(
                chunks=[],
                warnings=[warning],
                trace_id="trace-1",
            ),
        ])
        client = _make_client(fake_use_case)

        response = client.post("/social-style/retrieve", json={
            "tenant_id": "tenant-1",
            "query": "test",
            "style_categories": ["voice_rules"],
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data["warnings"]) == 1
        assert data["warnings"][0]["code"] == "EMPTY_RETRIEVAL_RESULT"
        assert data["warnings"][0]["message"] == "No results found"

    def test_empty_categories_returns_empty_response(self):
        """Empty style_categories list returns empty response with no errors."""
        fake_use_case = FakeRetrieveUseCase()
        client = _make_client(fake_use_case)

        response = client.post("/social-style/retrieve", json={
            "tenant_id": "tenant-1",
            "query": "test",
            "style_categories": [],
        })

        assert response.status_code == 200
        data = response.json()
        assert data["voice_rules"] == []
        assert data["hook_patterns"] == []
        assert data["trace_ids"] == []

    def test_metadata_preserved_in_entry_response(self):
        """Entry metadata from chunks is preserved in response."""
        fake_use_case = FakeRetrieveUseCase(results_by_call=[
            RetrieveResult(
                chunks=[_make_chunk("Test content", "voice_rules")],
                warnings=[],
                trace_id="trace-1",
            ),
        ])
        client = _make_client(fake_use_case)

        response = client.post("/social-style/retrieve", json={
            "tenant_id": "tenant-1",
            "query": "test",
            "style_categories": ["voice_rules"],
        })

        assert response.status_code == 200
        data = response.json()
        entry = data["voice_rules"][0]
        assert "metadata" in entry
        assert entry["metadata"]["style_category"] == "voice_rules"
