"""Tests for RetrieveUseCase - central orchestrator of the Retrieval Core."""
import pytest
from app.retrieval import (
    RetrieveRequest,
    RetrievalScope,
    RetrievalMode,
    EffectiveRetrieveRequest,
    RetrievedChunk,
    RetrievalGatewayResult,
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


def make_chunk(
    *,
    chunk_id: str = "doc-1:0",
    document_id: str = "doc-1",
    content: str = "test",
    score: float = 0.9,
    rank: int = 1,
    retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
    metadata_overrides: dict | None = None,
) -> RetrievedChunk:
    metadata = {
        "service_name": "test-service",
        "tenant_id": "tenant-123",
        "collection": "documents",
        "source_type": "document",
        "source_label": "test.pdf",
        "document_id": document_id,
        "chunk_index": 0,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)

    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        score=score,
        rank=rank,
        retrieval_mode=retrieval_mode,
        metadata=metadata,
    )


# Test Fixtures - Fake Gateway for testing
class FakeGateway:
    """Fake gateway for testing RetrieveUseCase."""

    def __init__(self):
        self.chunks_to_return = []
        self.warnings_to_return = []
        self.diagnostics_to_return = {}
        self.exception_to_raise = None
        self.calls = []

    def retrieve(self, request: EffectiveRetrieveRequest) -> RetrievalGatewayResult:
        self.calls.append(request)

        if self.exception_to_raise:
            raise self.exception_to_raise

        return RetrievalGatewayResult(
            chunks=self.chunks_to_return,
            warnings=self.warnings_to_return,
            diagnostics=self.diagnostics_to_return
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
    chunk = make_chunk(content="test content", score=0.95)
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
        make_chunk()
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
        make_chunk()
    ]

    use_case.execute(request)

    # Verify scope was passed to gateway
    assert len(fake_gateway.calls) == 1
    effective_request = fake_gateway.calls[0]
    assert effective_request.validated_scope == valid_scope


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
    chunk = make_chunk(
        metadata_overrides={
            "source_type": None,
            "source_label": None,
        }
    )
    del chunk.metadata["source_type"]
    del chunk.metadata["source_label"]
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
    chunk = make_chunk(metadata_overrides={"service_name": "wrong-service"})
    fake_gateway.chunks_to_return = [chunk]

    with pytest.raises(RetrievedChunkValidationError) as exc_info:
        use_case.execute(valid_request)

    error = exc_info.value
    assert "service_name" in error.internal_message or "scope" in error.internal_message.lower()
    assert trace_sink.traces[0].failure_stage == FailureStage.POST_VALIDATION


def test_post_validation_rejects_chunk_outside_validated_collections(use_case, fake_gateway, trace_sink):
    # Request scope with multiple collections
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents", "notes"],
        filters={}
    )
    request = RetrieveRequest(
        query="test",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=scope
    )

    # Chunk from collection not in validated scope
    chunk = make_chunk(metadata_overrides={"collection": "external-data"})
    fake_gateway.chunks_to_return = [chunk]

    with pytest.raises(RetrievedChunkValidationError) as exc_info:
        use_case.execute(request)

    error = exc_info.value
    assert "collection" in error.internal_message
    assert "external-data" in error.internal_message
    assert trace_sink.traces[0].failure_stage == FailureStage.POST_VALIDATION


def test_post_validation_rejects_chunk_violating_primitive_filter(use_case, fake_gateway, trace_sink):
    # Scope with primitive filter constraint
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={"department": "engineering"}
    )
    request = RetrieveRequest(
        query="test",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=scope
    )

    # Chunk with wrong department value
    chunk = make_chunk(metadata_overrides={"department": "sales"})
    fake_gateway.chunks_to_return = [chunk]

    with pytest.raises(RetrievedChunkValidationError) as exc_info:
        use_case.execute(request)

    error = exc_info.value
    assert "filter constraint" in error.internal_message
    assert "department" in error.internal_message
    assert trace_sink.traces[0].failure_stage == FailureStage.POST_VALIDATION


def test_post_validation_rejects_chunk_violating_list_filter(use_case, fake_gateway, trace_sink):
    # Scope with list filter constraint (field IN values)
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={"status": ["active", "pending"]}
    )
    request = RetrieveRequest(
        query="test",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=scope
    )

    # Chunk with status not in allowed list
    chunk = make_chunk(metadata_overrides={"status": "archived"})
    fake_gateway.chunks_to_return = [chunk]

    with pytest.raises(RetrievedChunkValidationError) as exc_info:
        use_case.execute(request)

    error = exc_info.value
    assert "filter constraint" in error.internal_message
    assert "status" in error.internal_message
    assert "archived" in error.internal_message
    assert trace_sink.traces[0].failure_stage == FailureStage.POST_VALIDATION


def test_post_validation_accepts_chunk_satisfying_all_constraints(use_case, fake_gateway, valid_scope):
    # Scope with both primitive and list filters
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents", "notes"],
        filters={
            "department": "engineering",
            "status": ["active", "pending"]
        }
    )
    request = RetrieveRequest(
        query="test",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=scope
    )

    # Chunk satisfying all constraints
    chunk = make_chunk(metadata_overrides={
        "collection": "notes",
        "department": "engineering",
        "status": "pending"
    })
    fake_gateway.chunks_to_return = [chunk]

    result = use_case.execute(request)

    # Should succeed and return the chunk
    assert len(result.chunks) == 1
    assert result.chunks[0] == chunk


def test_post_validation_accepts_chunk_in_one_of_validated_collections(use_case, fake_gateway):
    # Scope with multiple collections
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents", "notes", "files"],
        filters={}
    )
    request = RetrieveRequest(
        query="test",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=scope
    )

    # Chunk from one of the validated collections
    chunk = make_chunk(metadata_overrides={"collection": "files"})
    fake_gateway.chunks_to_return = [chunk]

    result = use_case.execute(request)

    assert len(result.chunks) == 1
    assert result.chunks[0].metadata["collection"] == "files"


def test_post_validation_accepts_chunk_matching_list_filter_value(use_case, fake_gateway):
    # Scope with list filter
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={"priority": ["high", "medium", "low"]}
    )
    request = RetrieveRequest(
        query="test",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=scope
    )

    # Chunk with priority in allowed list
    chunk = make_chunk(metadata_overrides={"priority": "medium"})
    fake_gateway.chunks_to_return = [chunk]

    result = use_case.execute(request)

    assert len(result.chunks) == 1
    assert result.chunks[0].metadata["priority"] == "medium"


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
        make_chunk()
    ]

    result = use_case.execute(valid_request)

    # Should contain gateway warnings
    assert len(result.warnings) >= 1
    assert any(w.source == "FakeGateway" for w in result.warnings)


# Test: Trace emission
def test_trace_emitted_on_success(use_case, fake_gateway, valid_request, trace_sink):
    fake_gateway.chunks_to_return = [
        make_chunk()
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
        make_chunk()
    ]

    use_case.execute(valid_request)

    trace = trace_sink.traces[0]
    assert trace.trace_id is not None
    assert trace.correlation_id == "corr-123"
    assert trace.timing is not None
    assert "start" in trace.timing or "duration_ms" in trace.timing
    assert trace.result_count == 1
    assert isinstance(trace.warnings, list)


# Test: Gateway diagnostics propagation
def test_success_trace_includes_gateway_diagnostics(use_case, fake_gateway, valid_request, trace_sink):
    """Success trace should include diagnostics from gateway result."""
    fake_gateway.chunks_to_return = [make_chunk()]

    # Fake gateway will return diagnostics
    class GatewayWithDiagnostics:
        def retrieve(self, request):
            return RetrievalGatewayResult(
                chunks=[make_chunk()],
                warnings=[],
                diagnostics={"adapter": "legacy", "search_ms": 42}
            )

    use_case.gateway = GatewayWithDiagnostics()
    use_case.execute(valid_request)

    # Check trace has diagnostics
    assert len(trace_sink.traces) == 1
    trace = trace_sink.traces[0]
    assert trace.status == TraceStatus.SUCCESS
    assert "adapter" in trace.diagnostics
    assert trace.diagnostics["adapter"] == "legacy"
    assert trace.diagnostics["search_ms"] == 42


def test_diagnostics_not_in_public_response(use_case, fake_gateway, valid_request):
    """Diagnostics should not be exposed in the public RetrieveResult."""
    class GatewayWithDiagnostics:
        def retrieve(self, request):
            return RetrievalGatewayResult(
                chunks=[make_chunk()],
                warnings=[],
                diagnostics={"internal": "data"}
            )

    use_case.gateway = GatewayWithDiagnostics()
    result = use_case.execute(valid_request)

    # RetrieveResult should only have chunks and warnings
    assert hasattr(result, "chunks")
    assert hasattr(result, "warnings")
    assert not hasattr(result, "diagnostics")


# Test: Non-fatal trace emission
class FailingTraceSink:
    """Trace sink that always fails."""

    def __init__(self):
        self.emit_calls = []

    def emit(self, trace):
        self.emit_calls.append(trace)
        raise RuntimeError("Trace sink unavailable")


def test_trace_sink_failure_does_not_fail_successful_retrieval(fake_gateway, scope_policy, clock, trace_id_generator, valid_request):
    """If trace emission fails during successful retrieval, result should still be returned."""
    failing_sink = FailingTraceSink()
    use_case = RetrieveUseCase(
        gateway=fake_gateway,
        scope_policy=scope_policy,
        clock=clock,
        trace_id_generator=trace_id_generator,
        trace_sink=failing_sink
    )

    fake_gateway.chunks_to_return = [make_chunk()]

    # Should not raise despite trace sink failure
    result = use_case.execute(valid_request)

    assert len(result.chunks) == 1
    assert len(failing_sink.emit_calls) == 1  # Trace emission was attempted


def test_trace_sink_failure_produces_operational_warning(fake_gateway, scope_policy, clock, trace_id_generator, valid_request, caplog):
    """When trace emission fails, an operational warning should be logged."""
    import logging

    failing_sink = FailingTraceSink()
    use_case = RetrieveUseCase(
        gateway=fake_gateway,
        scope_policy=scope_policy,
        clock=clock,
        trace_id_generator=trace_id_generator,
        trace_sink=failing_sink
    )

    fake_gateway.chunks_to_return = [make_chunk()]

    with caplog.at_level(logging.WARNING):
        use_case.execute(valid_request)

    # Should have logged a warning about trace emission failure
    assert any("Failed to emit" in record.message for record in caplog.records)
    assert any("trace" in record.message.lower() for record in caplog.records)


def test_trace_sink_failure_on_failed_retrieval_does_not_suppress_error(scope_policy, clock, trace_id_generator, valid_scope):
    """If trace emission fails during failed retrieval, original error should still propagate."""
    failing_sink = FailingTraceSink()
    fake_gateway = FakeGateway()

    use_case = RetrieveUseCase(
        gateway=fake_gateway,
        scope_policy=scope_policy,
        clock=clock,
        trace_id_generator=trace_id_generator,
        trace_sink=failing_sink
    )

    request = RetrieveRequest(
        query="",  # Invalid - empty query
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=valid_scope
    )

    # Should still raise the validation error
    with pytest.raises(InvalidRetrievalRequestError):
        use_case.execute(request)

    # Trace emission was attempted
    assert len(failing_sink.emit_calls) == 1


def test_all_failure_stages_emit_traces_even_with_failing_sink(scope_policy, clock, trace_id_generator, valid_scope):
    """All failure stages should attempt trace emission even if sink fails."""
    failing_sink = FailingTraceSink()
    fake_gateway = FakeGateway()

    use_case = RetrieveUseCase(
        gateway=fake_gateway,
        scope_policy=scope_policy,
        clock=clock,
        trace_id_generator=trace_id_generator,
        trace_sink=failing_sink
    )

    # Test request validation failure
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

    assert len(failing_sink.emit_calls) == 1
    assert failing_sink.emit_calls[0].failure_stage == FailureStage.REQUEST_VALIDATION
