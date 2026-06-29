"""Tests for Retrieval Core domain types, contracts, and test doubles."""
import pytest
from datetime import datetime
from app.retrieval import (
    # Enums
    RetrievalMode,
    WarningCode,
    WarningSeverity,
    TraceStatus,
    FailureStage,
    # Value objects
    RetrievalScope,
    RetrieveRequest,
    EffectiveRetrieveRequest,
    RetrievedChunk,
    RetrievalGatewayResult,
    ScopeDecision,
    RetrievalWarning,
    RetrievalTrace,
    # Errors
    RetrievalError,
    InvalidRetrievalRequestError,
    UnsupportedRetrievalModeError,
    NoIndexedCorpusError,
    RetrievalExecutionError,
    RetrievedChunkValidationError,
    # Protocols
    RetrievalGateway,
    ScopePolicy,
    RetrievalTraceSink,
    Clock,
    TraceIdGenerator,
    # Implementations
    InMemoryTraceSink,
    NoOpTraceSink,
    StructuredLoggingTraceSink,
    NamespacePolicy,
    PassthroughScopePolicy,
    SystemClock,
    UuidTraceIdGenerator,
)


# Test Enums
def test_retrieval_mode_enum():
    assert RetrievalMode.DENSE
    assert RetrievalMode.LEXICAL
    assert RetrievalMode.HYBRID


def test_warning_code_enum():
    assert WarningCode.LEGACY_METADATA_DEFAULTED
    assert WarningCode.NAMESPACE_DEFAULT_SCOPE
    assert WarningCode.EMPTY_RETRIEVAL_RESULT


def test_warning_severity_enum():
    assert WarningSeverity.LOW
    assert WarningSeverity.MEDIUM
    assert WarningSeverity.HIGH


def test_trace_status_enum():
    assert TraceStatus.SUCCESS
    assert TraceStatus.FAILED


def test_failure_stage_enum():
    assert FailureStage.REQUEST_VALIDATION
    assert FailureStage.SCOPE_POLICY
    assert FailureStage.GATEWAY_EXECUTION
    assert FailureStage.POST_VALIDATION


# Test RetrievalScope
def test_retrieval_scope_construction():
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["col1", "col2"],
        filters={"key": "value"}
    )
    assert scope.service_name == "test-service"
    assert scope.tenant_id == "tenant-123"
    assert scope.collections == ["col1", "col2"]
    assert scope.filters == {"key": "value"}


def test_retrieval_scope_empty_filters():
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["col1"],
        filters={}
    )
    assert scope.filters == {}


# Test RetrieveRequest
def test_retrieve_request_construction():
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={}
    )
    request = RetrieveRequest(
        query="test query",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=scope,
        correlation_id="corr-123"
    )
    assert request.query == "test query"
    assert request.retrieval_mode == RetrievalMode.DENSE
    assert request.limit == 5
    assert request.scope == scope
    assert request.correlation_id == "corr-123"


def test_retrieve_request_optional_correlation_id():
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={}
    )
    request = RetrieveRequest(
        query="test query",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        scope=scope
    )
    assert request.correlation_id is None


# Test EffectiveRetrieveRequest
def test_effective_retrieve_request_construction():
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={}
    )
    effective_request = EffectiveRetrieveRequest(
        normalized_query="test query",
        original_query=" test query ",
        retrieval_mode=RetrievalMode.DENSE,
        limit=5,
        validated_scope=scope,
        correlation_id="corr-123"
    )
    assert effective_request.normalized_query == "test query"
    assert effective_request.original_query == " test query "
    assert effective_request.retrieval_mode == RetrievalMode.DENSE
    assert effective_request.limit == 5
    assert effective_request.validated_scope == scope
    assert effective_request.correlation_id == "corr-123"


# Test RetrievedChunk
def test_retrieved_chunk_construction():
    chunk = RetrievedChunk(
        chunk_id="doc-123:0",
        document_id="doc-123",
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
            "document_id": "doc-123",
            "chunk_index": 0
        }
    )
    assert chunk.chunk_id == "doc-123:0"
    assert chunk.document_id == "doc-123"
    assert chunk.content == "chunk text"
    assert chunk.score == 0.95
    assert chunk.rank == 1
    assert chunk.retrieval_mode == RetrievalMode.DENSE
    assert chunk.metadata["service_name"] == "local-rag"
    assert chunk.metadata["tenant_id"] == "default"
    assert chunk.metadata["collection"] == "documents"
    assert chunk.metadata["source_type"] == "document"
    assert chunk.metadata["source_label"] == "file.pdf"
    assert chunk.metadata["document_id"] == "doc-123"
    assert chunk.metadata["chunk_index"] == 0


# Test RetrievalWarning
def test_retrieval_warning_construction():
    warning = RetrievalWarning(
        code=WarningCode.LEGACY_METADATA_DEFAULTED,
        severity=WarningSeverity.MEDIUM,
        source="LegacyDocumentRetrievalAdapter",
        message="Using sentinel defaults for metadata",
        details={"field": "service_name"}
    )
    assert warning.code == WarningCode.LEGACY_METADATA_DEFAULTED
    assert warning.severity == WarningSeverity.MEDIUM
    assert warning.source == "LegacyDocumentRetrievalAdapter"
    assert warning.message == "Using sentinel defaults for metadata"
    assert warning.details == {"field": "service_name"}


def test_retrieval_warning_optional_details():
    warning = RetrievalWarning(
        code=WarningCode.EMPTY_RETRIEVAL_RESULT,
        severity=WarningSeverity.LOW,
        source="RetrieveUseCase",
        message="No results found"
    )
    assert warning.details is None


# Test RetrievalGatewayResult
def test_retrieval_gateway_result_construction():
    chunk = RetrievedChunk(
        chunk_id="doc-123:0",
        document_id="doc-123",
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
            "document_id": "doc-123",
            "chunk_index": 0
        }
    )
    warning = RetrievalWarning(
        code=WarningCode.LEGACY_METADATA_DEFAULTED,
        severity=WarningSeverity.MEDIUM,
        source="LegacyDocumentRetrievalAdapter",
        message="Using sentinel defaults"
    )
    result = RetrievalGatewayResult(
        chunks=[chunk],
        warnings=[warning],
        diagnostics={"retrieval_time_ms": 50}
    )
    assert len(result.chunks) == 1
    assert len(result.warnings) == 1
    assert result.diagnostics == {"retrieval_time_ms": 50}


# Test ScopeDecision
def test_scope_decision_construction():
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={}
    )
    decision = ScopeDecision(
        validated_scope=scope,
        policy_name="PassthroughScopePolicy",
        warnings=[]
    )
    assert decision.validated_scope == scope
    assert decision.policy_name == "PassthroughScopePolicy"
    assert decision.warnings == []


# Test RetrievalTrace
def test_retrieval_trace_construction_success():
    trace = RetrievalTrace(
        trace_id="trace-123",
        correlation_id="corr-123",
        status=TraceStatus.SUCCESS,
        failure_stage=None,
        request_summary={"query": "test", "mode": "DENSE", "limit": 5},
        timing={"start": "2026-06-08T10:00:00", "end": "2026-06-08T10:00:01", "duration_ms": 1000},
        result_count=3,
        warnings=[]
    )
    assert trace.trace_id == "trace-123"
    assert trace.correlation_id == "corr-123"
    assert trace.status == TraceStatus.SUCCESS
    assert trace.failure_stage is None
    assert trace.result_count == 3
    assert trace.diagnostics == {}


def test_retrieval_trace_construction_failed():
    trace = RetrievalTrace(
        trace_id="trace-123",
        correlation_id=None,
        status=TraceStatus.FAILED,
        failure_stage=FailureStage.REQUEST_VALIDATION,
        request_summary={"query": "", "mode": "DENSE", "limit": 5},
        timing={"start": "2026-06-08T10:00:00", "end": "2026-06-08T10:00:01", "duration_ms": 10},
        result_count=0,
        warnings=[],
        diagnostics={"validation_error": "empty query"}
    )
    assert trace.status == TraceStatus.FAILED
    assert trace.failure_stage == FailureStage.REQUEST_VALIDATION
    assert trace.result_count == 0
    assert trace.diagnostics == {"validation_error": "empty query"}


# Test Domain Errors
def test_retrieval_error_base():
    error = RetrievalError(
        code="RETRIEVAL_ERROR",
        trace_id="trace-123",
        internal_message="Something went wrong",
        details={"detail": "value"}
    )
    assert error.code == "RETRIEVAL_ERROR"
    assert error.trace_id == "trace-123"
    assert error.internal_message == "Something went wrong"
    assert error.details == {"detail": "value"}


def test_invalid_retrieval_request_error():
    error = InvalidRetrievalRequestError(
        trace_id="trace-123",
        internal_message="Empty query",
        details={"query": ""}
    )
    assert error.code == "INVALID_RETRIEVAL_REQUEST"
    assert isinstance(error, RetrievalError)


def test_unsupported_retrieval_mode_error():
    error = UnsupportedRetrievalModeError(
        trace_id="trace-123",
        internal_message="Invalid mode",
        details={"mode": "UNKNOWN"}
    )
    assert error.code == "UNSUPPORTED_RETRIEVAL_MODE"
    assert isinstance(error, RetrievalError)


def test_no_indexed_corpus_error():
    error = NoIndexedCorpusError(
        trace_id="trace-123",
        internal_message="No documents indexed",
        details={}
    )
    assert error.code == "NO_INDEXED_CORPUS"
    assert isinstance(error, RetrievalError)


def test_retrieval_execution_error():
    error = RetrievalExecutionError(
        trace_id="trace-123",
        internal_message="Gateway failed",
        details={"exception": "ConnectionError"}
    )
    assert error.code == "RETRIEVAL_EXECUTION_ERROR"
    assert isinstance(error, RetrievalError)


def test_retrieved_chunk_validation_error():
    error = RetrievedChunkValidationError(
        trace_id="trace-123",
        internal_message="Missing required metadata",
        details={"missing_field": "service_name"}
    )
    assert error.code == "RETRIEVED_CHUNK_VALIDATION_ERROR"
    assert isinstance(error, RetrievalError)


# Test InMemoryTraceSink
def test_in_memory_trace_sink():
    sink = InMemoryTraceSink()
    assert len(sink.traces) == 0

    trace = RetrievalTrace(
        trace_id="trace-123",
        correlation_id=None,
        status=TraceStatus.SUCCESS,
        failure_stage=None,
        request_summary={},
        timing={},
        result_count=3,
        warnings=[]
    )
    sink.emit(trace)

    assert len(sink.traces) == 1
    assert sink.traces[0] == trace


# Test NoOpTraceSink
def test_noop_trace_sink():
    sink = NoOpTraceSink()
    trace = RetrievalTrace(
        trace_id="trace-123",
        correlation_id=None,
        status=TraceStatus.SUCCESS,
        failure_stage=None,
        request_summary={},
        timing={},
        result_count=3,
        warnings=[]
    )
    # Should not raise any errors
    sink.emit(trace)


# Test StructuredLoggingTraceSink
def test_structured_logging_trace_sink(caplog):
    import logging
    caplog.set_level(logging.INFO)

    sink = StructuredLoggingTraceSink()
    trace = RetrievalTrace(
        trace_id="trace-123",
        correlation_id="corr-123",
        status=TraceStatus.SUCCESS,
        failure_stage=None,
        request_summary={"query": "test", "mode": "DENSE"},
        timing={"duration_ms": 100},
        result_count=3,
        warnings=[]
    )
    sink.emit(trace)

    # Check that log was emitted with key=value format
    assert "trace_id=trace-123" in caplog.text
    assert "status=success" in caplog.text


# Test PassthroughScopePolicy
def test_passthrough_scope_policy_valid():
    policy = PassthroughScopePolicy()
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={}
    )
    decision = policy.evaluate(scope)

    assert decision.validated_scope == scope
    assert decision.policy_name == "PassthroughScopePolicy"
    assert len(decision.warnings) == 0


def test_passthrough_scope_policy_empty_service_name():
    policy = PassthroughScopePolicy()
    scope = RetrievalScope(
        service_name="",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={}
    )

    with pytest.raises(RetrievalError) as exc_info:
        policy.evaluate(scope)

    assert "service_name" in exc_info.value.internal_message.lower()


def test_passthrough_scope_policy_empty_tenant_id():
    policy = PassthroughScopePolicy()
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="",
        collections=["documents"],
        filters={}
    )

    with pytest.raises(RetrievalError) as exc_info:
        policy.evaluate(scope)

    assert "tenant_id" in exc_info.value.internal_message.lower()


def test_namespace_policy_implements_scope_policy_protocol():
    policy: ScopePolicy = NamespacePolicy()
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={},
    )

    decision = policy.evaluate(scope)

    assert decision.policy_name == "NamespacePolicy"


def test_namespace_policy_empty_collections():
    policy = NamespacePolicy()
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=[],
        filters={},
    )

    with pytest.raises(InvalidRetrievalRequestError) as exc_info:
        policy.evaluate(scope)

    assert "collections" in exc_info.value.internal_message.lower()


def test_namespace_policy_empty_collection_name():
    policy = NamespacePolicy()
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents", ""],
        filters={},
    )

    with pytest.raises(InvalidRetrievalRequestError) as exc_info:
        policy.evaluate(scope)

    assert "collection" in exc_info.value.internal_message.lower()


def test_namespace_policy_empty_service_name():
    policy = NamespacePolicy()
    scope = RetrievalScope(
        service_name="",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={},
    )

    with pytest.raises(InvalidRetrievalRequestError) as exc_info:
        policy.evaluate(scope)

    assert "service_name" in exc_info.value.internal_message.lower()


def test_namespace_policy_empty_tenant_id():
    policy = NamespacePolicy()
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="",
        collections=["documents"],
        filters={},
    )

    with pytest.raises(InvalidRetrievalRequestError) as exc_info:
        policy.evaluate(scope)

    assert "tenant_id" in exc_info.value.internal_message.lower()


def test_namespace_policy_rejects_disallowed_collection():
    policy = NamespacePolicy(allowed_collections={"documents", "notes"})
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents", "secret"],
        filters={},
    )

    with pytest.raises(InvalidRetrievalRequestError) as exc_info:
        policy.evaluate(scope)

    assert exc_info.value.details["invalid_collections"] == ["secret"]


def test_namespace_policy_accepts_allowed_collections():
    policy = NamespacePolicy(allowed_collections={"documents", "notes"})
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents", "notes"],
        filters={"topic": "rag"},
    )

    decision = policy.evaluate(scope)

    assert decision.validated_scope == scope
    assert decision.warnings == []


def test_namespace_policy_passes_all_collections_when_unconfigured():
    policy = NamespacePolicy()
    scope = RetrievalScope(
        service_name="test-service",
        tenant_id="tenant-123",
        collections=["documents", "other-service"],
        filters={},
    )

    decision = policy.evaluate(scope)

    assert decision.validated_scope == scope
    assert decision.warnings == []


def test_namespace_policy_emits_default_scope_warning():
    policy = NamespacePolicy()
    scope = RetrievalScope(
        service_name="local-rag",
        tenant_id="default",
        collections=["documents"],
        filters={},
    )

    decision = policy.evaluate(scope)

    assert len(decision.warnings) == 1
    warning = decision.warnings[0]
    assert warning.code == WarningCode.NAMESPACE_DEFAULT_SCOPE
    assert warning.severity == WarningSeverity.LOW
    assert warning.source == "NamespacePolicy"
    assert "sentinel default scope" in warning.message.lower()


def test_namespace_policy_is_exported_from_package():
    from app.retrieval import NamespacePolicy as ExportedNamespacePolicy

    assert ExportedNamespacePolicy is NamespacePolicy


# Test SystemClock
def test_system_clock():
    clock = SystemClock()
    now = clock.now()
    assert isinstance(now, datetime)


# Test UuidTraceIdGenerator
def test_uuid_trace_id_generator():
    generator = UuidTraceIdGenerator()
    trace_id1 = generator.generate()
    trace_id2 = generator.generate()

    assert isinstance(trace_id1, str)
    assert isinstance(trace_id2, str)
    assert trace_id1 != trace_id2
    assert len(trace_id1) == 36  # UUID format


class TestRetrievedChunkDomainMetadata:
    def _chunk(self, metadata: dict) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            content="text",
            score=1.0,
            rank=0,
            retrieval_mode=RetrievalMode.DENSE,
            metadata=metadata,
        )

    def test_returns_only_non_core_keys(self):
        chunk = self._chunk(
            {
                "service_name": "svc",
                "tenant_id": "ten",
                "collection": "docs",
                "source_type": "text",
                "source_label": "f.pdf",
                "document_id": "d1",
                "original_filename": "f.pdf",
                "chunk_index": 0,
                "topic": "platform",
                "is_external": False,
            }
        )

        assert chunk.domain_metadata() == {"topic": "platform", "is_external": False}

    def test_empty_when_only_core_keys(self):
        chunk = self._chunk(
            {
                "service_name": "svc",
                "tenant_id": "ten",
                "collection": "docs",
                "source_type": "text",
                "source_label": "f.pdf",
            }
        )

        assert chunk.domain_metadata() == {}
