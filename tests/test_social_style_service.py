import pytest
from unittest.mock import Mock

from app.retrieval.types import (
    RetrievalMode,
    RetrievalScope,
    RetrievalWarning,
    WarningCode,
    WarningSeverity,
    RetrievedChunk,
)
from app.retrieval.use_case import RetrieveResult
from app.social import (
    StyleCategory,
    StyleRetrievalRequest,
    StyleContext,
    StyleEntry,
    SourceReference,
    SocialStyleRetrievalService,
    StyleResponseMapper,
)


@pytest.fixture
def mock_retrieve_use_case():
    return Mock()


@pytest.fixture
def social_service(mock_retrieve_use_case):
    return SocialStyleRetrievalService(
        retrieve_use_case=mock_retrieve_use_case,
        service_name="social-style",
    )


@pytest.fixture
def style_mapper():
    return StyleResponseMapper()


@pytest.fixture
def sample_request():
    return StyleRetrievalRequest(
        tenant_id="tenant-123",
        query="professional tone",
        style_categories=[StyleCategory.VOICE_RULES, StyleCategory.HOOK_PATTERNS],
        per_category_limit=3,
    )


class TestStyleCategory:
    """Tests for StyleCategory enum."""

    def test_has_all_six_categories(self):
        """StyleCategory enum has all six initial categories."""
        assert StyleCategory.VOICE_RULES.value == "voice_rules"
        assert StyleCategory.HOOK_PATTERNS.value == "hook_patterns"
        assert StyleCategory.CTA_PATTERNS.value == "cta_patterns"
        assert StyleCategory.PAST_POST_PATTERNS.value == "past_post_patterns"
        assert StyleCategory.AVOID_RULES.value == "avoid_rules"
        assert StyleCategory.OFFER_POSITIONING.value == "offer_positioning"


class TestStyleRetrievalRequest:
    """Tests for StyleRetrievalRequest model."""

    def test_request_has_required_fields(self):
        """StyleRetrievalRequest includes all required fields."""
        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test query",
            style_categories=[StyleCategory.VOICE_RULES],
        )

        assert request.tenant_id == "tenant-1"
        assert request.query == "test query"
        assert request.style_categories == [StyleCategory.VOICE_RULES]

    def test_request_defaults(self):
        """StyleRetrievalRequest has correct defaults."""
        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test",
            style_categories=[],
        )

        assert request.platform is None
        assert request.per_category_limit == 5
        assert request.collection == "style_memory"

    def test_request_with_platform(self):
        """StyleRetrievalRequest accepts optional platform."""
        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test",
            style_categories=[],
            platform="twitter",
        )

        assert request.platform == "twitter"


class TestSocialStyleRetrievalService:
    """Tests for SocialStyleRetrievalService."""

    def test_single_category_retrieval(
        self, social_service, mock_retrieve_use_case
    ):
        """Service translates single category into one RetrieveRequest."""
        # Setup mock
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[],
            warnings=[],
        )

        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test",
            style_categories=[StyleCategory.VOICE_RULES],
        )

        result = social_service.retrieve(request)

        # Should have called retrieve use case once
        assert mock_retrieve_use_case.execute.call_count == 1

    def test_multi_category_retrieval(
        self, social_service, mock_retrieve_use_case
    ):
        """Service translates multiple categories into separate RetrieveRequests."""
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[],
            warnings=[],
        )

        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test",
            style_categories=[
                StyleCategory.VOICE_RULES,
                StyleCategory.HOOK_PATTERNS,
                StyleCategory.CTA_PATTERNS,
            ],
        )

        result = social_service.retrieve(request)

        # Should have called retrieve use case three times (one per category)
        assert mock_retrieve_use_case.execute.call_count == 3

    def test_retrieval_scope_includes_style_category_filter(
        self, social_service, mock_retrieve_use_case
    ):
        """Each RetrieveRequest uses correct scope with style_category filter."""
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[],
            warnings=[],
        )

        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test",
            style_categories=[StyleCategory.VOICE_RULES],
        )

        social_service.retrieve(request)

        # Check the scope passed to retrieve use case
        call_args = mock_retrieve_use_case.execute.call_args[0][0]
        scope = call_args.scope

        assert scope.service_name == "social-style"
        assert scope.tenant_id == "tenant-1"
        assert scope.collections == ["style_memory"]
        assert scope.filters["style_category"] == "voice_rules"

    def test_platform_filter_included_when_provided(
        self, social_service, mock_retrieve_use_case
    ):
        """When platform is provided, it is included as a metadata filter."""
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[],
            warnings=[],
        )

        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test",
            style_categories=[StyleCategory.VOICE_RULES],
            platform="twitter",
        )

        social_service.retrieve(request)

        call_args = mock_retrieve_use_case.execute.call_args[0][0]
        scope = call_args.scope

        assert scope.filters["platform"] == "twitter"

    def test_platform_filter_not_included_when_none(
        self, social_service, mock_retrieve_use_case
    ):
        """When platform is not provided, no platform filter is applied."""
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[],
            warnings=[],
        )

        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test",
            style_categories=[StyleCategory.VOICE_RULES],
            platform=None,
        )

        social_service.retrieve(request)

        call_args = mock_retrieve_use_case.execute.call_args[0][0]
        scope = call_args.scope

        assert "platform" not in scope.filters

    def test_per_category_limit_applied(
        self, social_service, mock_retrieve_use_case
    ):
        """Service uses per_category_limit for each retrieval."""
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[],
            warnings=[],
        )

        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test",
            style_categories=[StyleCategory.VOICE_RULES],
            per_category_limit=10,
        )

        social_service.retrieve(request)

        call_args = mock_retrieve_use_case.execute.call_args[0][0]
        assert call_args.limit == 10

    def test_retrieval_mode_is_hybrid(
        self, social_service, mock_retrieve_use_case
    ):
        """Service uses hybrid retrieval mode."""
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[],
            warnings=[],
        )

        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test",
            style_categories=[StyleCategory.VOICE_RULES],
        )

        social_service.retrieve(request)

        call_args = mock_retrieve_use_case.execute.call_args[0][0]
        assert call_args.retrieval_mode == RetrievalMode.HYBRID

    def test_collects_trace_ids_from_each_retrieval(
        self, social_service, mock_retrieve_use_case
    ):
        """Trace IDs from all per-category retrievals are collected."""
        mock_retrieve_use_case.execute.side_effect = [
            RetrieveResult(chunks=[], warnings=[], trace_id="trace-1"),
            RetrieveResult(chunks=[], warnings=[], trace_id="trace-2"),
        ]

        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="test",
            style_categories=[StyleCategory.VOICE_RULES, StyleCategory.HOOK_PATTERNS],
        )

        context = social_service.retrieve(request)

        assert context.trace_ids == ["trace-1", "trace-2"]


class TestStyleResponseMapper:
    """Tests for StyleResponseMapper."""

    def test_groups_chunks_by_category(self, style_mapper):
        """Mapper groups retrieved entries by category."""
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="Use professional tone",
                score=0.95,
                rank=0,
                retrieval_mode=RetrievalMode.HYBRID,
                metadata={
                    "style_category": "voice_rules",
                    "source_label": "style-guide.pdf",
                },
            ),
            RetrievedChunk(
                chunk_id="chunk-2",
                document_id="doc-2",
                content="Start with a question",
                score=0.90,
                rank=1,
                retrieval_mode=RetrievalMode.HYBRID,
                metadata={
                    "style_category": "hook_patterns",
                    "source_label": "hooks.pdf",
                },
            ),
        ]

        context = style_mapper.map_to_context(
            chunks_by_category={
                StyleCategory.VOICE_RULES: chunks[0:1],
                StyleCategory.HOOK_PATTERNS: chunks[1:2],
            },
            warnings=[],
            trace_ids=["trace-1"],
            requested_categories=[StyleCategory.VOICE_RULES, StyleCategory.HOOK_PATTERNS],
        )

        assert len(context.voice_rules) == 1
        assert context.voice_rules[0].content == "Use professional tone"
        assert len(context.hook_patterns) == 1
        assert context.hook_patterns[0].content == "Start with a question"

    def test_converts_chunks_to_style_entries(self, style_mapper):
        """Mapper converts each chunk to a StyleEntry."""
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="Test content",
                score=0.95,
                rank=0,
                retrieval_mode=RetrievalMode.HYBRID,
                metadata={
                    "style_category": "voice_rules",
                    "source_label": "guide.pdf",
                    "platform": "twitter",
                },
            ),
        ]

        context = style_mapper.map_to_context(
            chunks_by_category={StyleCategory.VOICE_RULES: chunks},
            warnings=[],
            trace_ids=[],
            requested_categories=[StyleCategory.VOICE_RULES],
        )

        entry = context.voice_rules[0]
        assert isinstance(entry, StyleEntry)
        assert entry.content == "Test content"
        assert entry.source_label == "guide.pdf"
        assert entry.score == 0.95
        assert entry.metadata["platform"] == "twitter"

    def test_collects_source_references(self, style_mapper):
        """Mapper collects source references with document_id and source_label."""
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="Content 1",
                score=0.95,
                rank=0,
                retrieval_mode=RetrievalMode.HYBRID,
                metadata={"style_category": "voice_rules", "source_label": "guide1.pdf"},
            ),
            RetrievedChunk(
                chunk_id="chunk-2",
                document_id="doc-2",
                content="Content 2",
                score=0.90,
                rank=1,
                retrieval_mode=RetrievalMode.HYBRID,
                metadata={"style_category": "voice_rules", "source_label": "guide2.pdf"},
            ),
        ]

        context = style_mapper.map_to_context(
            chunks_by_category={StyleCategory.VOICE_RULES: chunks},
            warnings=[],
            trace_ids=[],
            requested_categories=[StyleCategory.VOICE_RULES],
        )

        assert len(context.source_references) == 2
        assert context.source_references[0].document_id == "doc-1"
        assert context.source_references[0].source_label == "guide1.pdf"
        assert context.source_references[1].document_id == "doc-2"

    def test_deduplicates_source_references(self, style_mapper):
        """Mapper deduplicates source references from same document."""
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="Content 1",
                score=0.95,
                rank=0,
                retrieval_mode=RetrievalMode.HYBRID,
                metadata={"style_category": "voice_rules", "source_label": "guide.pdf"},
            ),
            RetrievedChunk(
                chunk_id="chunk-2",
                document_id="doc-1",  # Same document
                content="Content 2",
                score=0.90,
                rank=1,
                retrieval_mode=RetrievalMode.HYBRID,
                metadata={"style_category": "voice_rules", "source_label": "guide.pdf"},
            ),
        ]

        context = style_mapper.map_to_context(
            chunks_by_category={StyleCategory.VOICE_RULES: chunks},
            warnings=[],
            trace_ids=[],
            requested_categories=[StyleCategory.VOICE_RULES],
        )

        # Should only have one source reference
        assert len(context.source_references) == 1
        assert context.source_references[0].document_id == "doc-1"

    def test_reports_missing_categories(self, style_mapper):
        """Missing categories (requested but zero results) are reported."""
        context = style_mapper.map_to_context(
            chunks_by_category={
                StyleCategory.VOICE_RULES: [
                    RetrievedChunk(
                        chunk_id="chunk-1",
                        document_id="doc-1",
                        content="Content",
                        score=0.95,
                        rank=0,
                        retrieval_mode=RetrievalMode.HYBRID,
                        metadata={"style_category": "voice_rules", "source_label": "guide.pdf"},
                    ),
                ],
                StyleCategory.HOOK_PATTERNS: [],  # Empty
            },
            warnings=[],
            trace_ids=[],
            requested_categories=[StyleCategory.VOICE_RULES, StyleCategory.HOOK_PATTERNS],
        )

        assert StyleCategory.HOOK_PATTERNS in context.missing_categories
        assert StyleCategory.VOICE_RULES not in context.missing_categories

    def test_merges_warnings_from_all_categories(self, style_mapper):
        """Warnings from all per-category retrievals are merged."""
        warning1 = RetrievalWarning(
            code=WarningCode.EMPTY_RETRIEVAL_RESULT,
            severity=WarningSeverity.MEDIUM,
            source="voice_rules",
            message="No results found",
        )
        warning2 = RetrievalWarning(
            code=WarningCode.EMPTY_RETRIEVAL_RESULT,
            severity=WarningSeverity.MEDIUM,
            source="hook_patterns",
            message="No results found",
        )

        context = style_mapper.map_to_context(
            chunks_by_category={},
            warnings=[warning1, warning2],
            trace_ids=[],
            requested_categories=[],
        )

        assert len(context.warnings) == 2
        assert warning1 in context.warnings
        assert warning2 in context.warnings

    def test_collects_trace_ids(self, style_mapper):
        """Trace IDs from all per-category retrievals are collected."""
        context = style_mapper.map_to_context(
            chunks_by_category={},
            warnings=[],
            trace_ids=["trace-1", "trace-2", "trace-3"],
            requested_categories=[],
        )

        assert len(context.trace_ids) == 3
        assert "trace-1" in context.trace_ids
        assert "trace-2" in context.trace_ids
        assert "trace-3" in context.trace_ids


class TestIntegration:
    """Integration tests combining service and mapper."""

    def test_end_to_end_style_retrieval(
        self, social_service, mock_retrieve_use_case
    ):
        """End-to-end test: request -> retrieve -> map -> response."""
        # Setup mock to return chunks
        mock_retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="Professional tone",
                    score=0.95,
                    rank=0,
                    retrieval_mode=RetrievalMode.HYBRID,
                    metadata={
                        "style_category": "voice_rules",
                        "source_label": "guide.pdf",
                    },
                ),
            ],
            warnings=[],
        )

        request = StyleRetrievalRequest(
            tenant_id="tenant-1",
            query="tone guidelines",
            style_categories=[StyleCategory.VOICE_RULES],
        )

        context = social_service.retrieve(request)

        assert isinstance(context, StyleContext)
        assert len(context.voice_rules) == 1
        assert context.voice_rules[0].content == "Professional tone"
        assert len(context.source_references) == 1
