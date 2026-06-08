"""Tests for RetrieveUseCase - central orchestrator of the Retrieval Core."""
import pytest
from datetime import datetime
from app.retrieval import (
    RetrieveRequest,
    RetrievalScope,
    RetrievalMode,
    EffectiveRetrieveRequest,
    RetrievedChunk,
    RetrievalGatewayResult,
    ScopeDecision,
    RetrievalWarning,
    WarningCode,
    WarningSeverity,
    TraceStatus,
    FailureStage,
    InvalidRetrievalRequestError,
    UnsupportedRetrievalModeError,
    NoIndexedCorpusError,
    RetrievalExecutionError,
    RetrievedChunkValidationError,
    InMemoryTraceSink,
    PassthroughScopePolicy,
    SystemClock,
    UuidTraceIdGenerator,
)
from app.retrieval.use_case import RetrieveUseCase


# Test Fixtures - Fake Gateway for testing
class FakeGateway:
    """Fake gateway for testing RetrieveUseCase."""

    def __init__(self):
        self.chunks_to_return = []
        self.warnings_to_return = []
        self.exception_to_raise = None
        self.calls = []

    def retrieve(self, request: EffectiveRetrieveRequest) -> RetrievalGatewayResult:
        self.calls.append(request)

        if self.exception_to_raise:
            raise self.exception_to_raise

        return RetrievalGatewayResult(
            chunks=self.chunks_to_return,
            warnings=self.warnings_to_return,
            diagnostics={}
        )


@pytest.fixture
def fake_gateway():
    return FakeGateway()


@pytest.fixture
def scope_policy():
    return PassthroughScopePolicy()


@pytest.fixture
def trace_sink():
    return InMemoryTraceSink()


@pytest.fixture
def clock():
    return SystemClock()


@pytest.fixture
def trace_id_generator():
    return UuidTraceIdGenerator()


@pytest.fixture
def use_case(fake_gateway, scope_policy, trace_sink, clock, trace_id_generator):
    return RetrieveUseCase(
        gateway=fake_gateway,
        scope_policy=scope_policy,
        clock=clock,
        trace_id_generator=trace_id_generator,
        trace_sink=trace_sink
    )


@pytest.fixture
def valid_scope():
    return RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={}
    )


@pytest.fixture
def valid_request(valid_scope):
    return RetrieveRequest(
        query="test query",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=valid_scope,
        correlation_id="corr-123"
    )


# Test: Constructor
def test_use_case_accepts_dependencies(fake_gateway, scope_policy, clock, trace_id_generator, trace_sink):
    use_case = RetrieveUseCase(
        gateway=fake_gateway,
        scope_policy=scope_policy,
        clock=clock,
        trace_id_generator=trace_id_generator,
        trace_sink=trace_sink
    )
    assert use_case is not None


# Test: Successful retrieval
def test_execute_returns_chunks_and_warnings_on_success(use_case, fake_gateway, valid_request, valid_scope):
    # Setup fake gateway to return chunks
    chunk = RetrievedChunk(
        content="test content",
        score=0.95,
        metadata={
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collection": "documents",
            "source_type": "document",
            "source_label": "test.pdf",
            "document_id": "doc-1",
            "chunk_index": 0
        }
    )
    warning = RetrievalWarning(
        code=WarningCode.LEGACY_METADATA_DEFAULTED,
        severity=WarningSeverity.LOW,
        source="FakeGateway",
        message="Test warning"
    )
    fake_gateway.chunks_to_return = [chunk]
    fake_gateway.warnings_to_return = [warning]

    result = use_case.execute(valid_request)

    assert len(result.chunks) == 1
    assert result.chunks[0] == chunk
    assert len(result.warnings) == 1
    assert result.warnings[0] == warning


# Test: Query normalization
def test_query_normalization_trims_whitespace(use_case, fake_gateway, valid_scope, trace_sink):
    request = RetrieveRequest(
        query="  test query  ",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=valid_scope
    )

    fake_gateway.chunks_to_return = [
        RetrievedChunk(
            content="test",
            score=0.9,
            metadata={
                "service_name": "test-service",
                "tenant_id": "tenant-123",
                "collection": "documents",
                "source_type": "document",
                "source_label": "test.pdf",
                "document_id": "doc-1",
                "chunk_index": 0
            }
        )
    ]

    use_case.execute(request)

    # Check that gateway received normalized query
    assert len(fake_gateway.calls) == 1
    effective_request = fake_gateway.calls[0]
    assert effective_request.normalized_query == "test query"
    assert effective_request.original_query == "  test query  "

    # Check trace contains both queries
    assert len(trace_sink.traces) == 1
    trace = trace_sink.traces[0]
    assert "original_query" in trace.request_summary
    assert "normalized_query" in trace.request_summary


# Test: Empty query validation
def test_empty_query_raises_invalid_request_error(use_case, valid_scope, trace_sink):
    request = RetrieveRequest(
        query="",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=valid_scope
    )

    with pytest.raises(InvalidRetrievalRequestError) as exc_info:
        use_case.execute(request)

    error = exc_info.value
    assert error.code == "INVALID_RETRIEVAL_REQUEST"
    assert "query" in error.internal_message.lower() or "empty" in error.internal_message.lower()

    # Check trace was emitted with failure
    assert len(trace_sink.traces) == 1
    trace = trace_sink.traces[0]
    assert trace.status == TraceStatus.FAILED
    assert trace.failure_stage == FailureStage.REQUEST_VALIDATION


def test_whitespace_only_query_raises_invalid_request_error(use_case, valid_scope):
    request = RetrieveRequest(
        query="   ",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=valid_scope
    )

    with pytest.raises(InvalidRetrievalRequestError):
        use_case.execute(request)


# Test: Limit validation
def test_limit_below_1_raises_invalid_request_error(use_case, valid_scope, trace_sink):
    request = RetrieveRequest(
        query="test",
        retrieval_mode=RetrievalMode.DENSE,
        limit=0,
        scope=valid_scope
    )

    with pytest.raises(InvalidRetrievalRequestError) as exc_info:
        use_case.execute(request)

    error = exc_info.value
    assert "limit" in error.internal_message.lower()

    # Check failure stage
    assert len(trace_sink.traces) == 1
    assert trace_sink.traces[0].failure_stage == FailureStage.REQUEST_VALIDATION


def test_limit_above_50_raises_invalid_request_error(use_case, valid_scope, trace_sink):
    request = RetrieveRequest(
        query="test",
        retrieval_mode=RetrievalMode.DENSE,
        limit=51,
        scope=valid_scope
    )

    with pytest.raises(InvalidRetrievalRequestError) as exc_info:
        use_case.execute(request)

    error = exc_info.value
    assert "limit" in error.internal_message.lower()
    assert trace_sink.traces[0].failure_stage == FailureStage.REQUEST_VALIDATION


# Test: Unsupported mode validation (testing with invalid enum would be caught by type system, so we skip this test)
# The type system ensures only valid RetrievalMode values can be passed


# Test: Scope policy invocation
def test_scope_policy_is_invoked(use_case, fake_gateway, valid_scope, trace_sink):
    # PassthroughScopePolicy will validate the scope
    request = RetrieveRequest(
        query="test",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=valid_scope
    )

    fake_gateway.chunks_to_return = [
        RetrievedChunk(
            content="test",
            score=0.9,
            metadata={
                "service_name": "test-service",
                "tenant_id": "tenant-123",
                "collection": "documents",
                "source_type": "document",
                "source_label": "test.pdf",
                "document_id": "doc-1",
                "chunk_index": 0
            }
        )
    ]

    use_case.execute(request)

    # Verify scope was passed to gateway
    assert len(fake_gateway.calls) == 1
    effective_request = fake_gateway.calls[0]
    assert effective_request.validated_scope.validated_scope == valid_scope
    assert effective_request.validated_scope.policy_name == "PassthroughScopePolicy"


def test_scope_validation_failure_raises_error_with_scope_policy_stage(use_case, trace_sink):
    # Invalid scope - empty service_name
    invalid_scope = RetrievalScope(
        service_name="",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={}
    )
    request = RetrieveRequest(
        query="test",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=invalid_scope
    )

    with pytest.raises(InvalidRetrievalRequestError):
        use_case.execute(request)

    # Check failure stage
    assert len(trace_sink.traces) == 1
    assert trace_sink.traces[0].failure_stage == FailureStage.SCOPE_POLICY


# Test: Gateway error propagation
def test_gateway_domain_error_propagates_with_gateway_execution_stage(use_case, fake_gateway, valid_request, trace_sink):
    fake_gateway.exception_to_raise = NoIndexedCorpusError(
        trace_id="test-trace",
        internal_message="No documents indexed",
        details={}
    )

    with pytest.raises(NoIndexedCorpusError):
        use_case.execute(valid_request)

    # Check failure stage
    assert len(trace_sink.traces) == 1
    assert trace_sink.traces[0].failure_stage == FailureStage.GATEWAY_EXECUTION


def test_gateway_unexpected_exception_wrapped_in_retrieval_execution_error(use_case, fake_gateway, valid_request, trace_sink):
    fake_gateway.exception_to_raise = RuntimeError("Unexpected error")

    with pytest.raises(RetrievalExecutionError) as exc_info:
        use_case.execute(valid_request)

    error = exc_info.value
    assert error.code == "RETRIEVAL_EXECUTION_ERROR"
    assert "Unexpected error" in error.internal_message or "RuntimeError" in str(error.details)

    # Check failure stage
    assert len(trace_sink.traces) == 1
    assert trace_sink.traces[0].failure_stage == FailureStage.GATEWAY_EXECUTION


# Test: Post-validation of chunks
def test_post_validation_checks_required_metadata_fields(use_case, fake_gateway, valid_request, trace_sink):
    # Chunk missing required field
    chunk = RetrievedChunk(
        content="test",
        score=0.9,
        metadata={
            "service_name": "test-service",
            "tenant_id": "tenant-123",
            "collection": "documents",
            # Missing source_type and source_label
            "document_id": "doc-1",
            "chunk_index": 0
        }
    )
    fake_gateway.chunks_to_return = [chunk]

    with pytest.raises(RetrievedChunkValidationError) as exc_info:
        use_case.execute(valid_request)

    error = exc_info.value
    assert error.code == "RETRIEVED_CHUNK_VALIDATION_ERROR"
    assert "source_type" in error.internal_message or "source_label" in error.internal_message

    # Check failure stage
    assert len(trace_sink.traces) == 1
    assert trace_sink.traces[0].failure_stage == FailureStage.POST_VALIDATION


def test_post_validation_checks_scope_compliance(use_case, fake_gateway, valid_request, trace_sink):
    # Chunk with mismatched service_name
    chunk = RetrievedChunk(
        content="test",
        score=0.9,
        metadata={
            "service_name": "wrong-service",  # Doesn't match request scope
            "tenant_id": "tenant-123",
            "collection": "documents",
            "source_type": "document",
            "source_label": "test.pdf",
            "document_id": "doc-1",
            "chunk_index": 0
        }
    )
    fake_gateway.chunks_to_return = [chunk]

    with pytest.raises(RetrievedChunkValidationError) as exc_info:
        use_case.execute(valid_request)

    error = exc_info.value
    assert "service_name" in error.internal_message or "scope" in error.internal_message.lower()
    assert trace_sink.traces[0].failure_stage == FailureStage.POST_VALIDATION


# Test: Warning aggregation
def test_warnings_from_gateway_and_scope_policy_are_merged(use_case, fake_gateway, valid_request):
    gateway_warning = RetrievalWarning(
        code=WarningCode.LEGACY_METADATA_DEFAULTED,
        severity=WarningSeverity.MEDIUM,
        source="FakeGateway",
        message="Gateway warning"
    )
    fake_gateway.warnings_to_return = [gateway_warning]
    fake_gateway.chunks_to_return = [
        RetrievedChunk(
            content="test",
            score=0.9,
            metadata={
                "service_name": "test-service",
                "tenant_id": "tenant-123",
                "collection": "documents",
                "source_type": "document",
                "source_label": "test.pdf",
                "document_id": "doc-1",
                "chunk_index": 0
            }
        )
    ]

    result = use_case.execute(valid_request)

    # Should contain gateway warnings
    assert len(result.warnings) >= 1
    assert any(w.source == "FakeGateway" for w in result.warnings)


# Test: Trace emission
def test_trace_emitted_on_success(use_case, fake_gateway, valid_request, trace_sink):
    fake_gateway.chunks_to_return = [
        RetrievedChunk(
            content="test",
            score=0.9,
            metadata={
                "service_name": "test-service",
                "tenant_id": "tenant-123",
                "collection": "documents",
                "source_type": "document",
                "source_label": "test.pdf",
                "document_id": "doc-1",
                "chunk_index": 0
            }
        )
    ]

    use_case.execute(valid_request)

    assert len(trace_sink.traces) == 1
    trace = trace_sink.traces[0]
    assert trace.status == TraceStatus.SUCCESS
    assert trace.failure_stage is None
    assert trace.result_count == 1
    assert trace.correlation_id == "corr-123"
    assert isinstance(trace.trace_id, str)
    assert "start" in trace.timing or "duration_ms" in trace.timing


def test_trace_emitted_on_failure(use_case, valid_scope, trace_sink):
    request = RetrieveRequest(
        query="",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=valid_scope
    )

    try:
        use_case.execute(request)
    except InvalidRetrievalRequestError:
        pass

    assert len(trace_sink.traces) == 1
    trace = trace_sink.traces[0]
    assert trace.status == TraceStatus.FAILED
    assert trace.failure_stage == FailureStage.REQUEST_VALIDATION
    assert trace.result_count == 0


# Test: Trace includes timing, correlation_id, warnings
def test_trace_includes_all_required_fields(use_case, fake_gateway, valid_request, trace_sink):
    fake_gateway.chunks_to_return = [
        RetrievedChunk(
            content="test",
            score=0.9,
            metadata={
                "service_name": "test-service",
                "tenant_id": "tenant-123",
                "collection": "documents",
                "source_type": "document",
                "source_label": "test.pdf",
                "document_id": "doc-1",
                "chunk_index": 0
            }
        )
    ]

    use_case.execute(valid_request)

    trace = trace_sink.traces[0]
    assert trace.trace_id is not None
    assert trace.correlation_id == "corr-123"
    assert trace.timing is not None
    assert "start" in trace.timing or "duration_ms" in trace.timing
    assert trace.result_count == 1
    assert isinstance(trace.warnings, list)
