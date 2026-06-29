"""RetrieveUseCase - central orchestrator of the Retrieval Core."""
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from app.retrieval.types import (
    RetrieveRequest,
    RetrievalScope,
    EffectiveRetrieveRequest,
    RetrievedChunk,
    RetrievalWarning,
    RetrievalTrace,
    TraceStatus,
    FailureStage,
)
from app.retrieval.errors import (
    RetrievalError,
    InvalidRetrievalRequestError,
    RetrievalExecutionError,
    RetrievedChunkValidationError,
)
from app.retrieval.contracts import (
    RetrievalGateway,
    ScopePolicy,
    Clock,
    TraceIdGenerator,
    RetrievalTraceSink,
)


logger = logging.getLogger(__name__)


@dataclass
class RetrieveResult:
    """Result from RetrieveUseCase.execute()."""
    chunks: list[RetrievedChunk]
    warnings: list[RetrievalWarning]
    trace_id: str | None = None


class _TracedRetrieval:
    """Owns a single retrieval's trace: id, timing, current stage, and emission.

    Used as a context manager around the orchestration. The body advances the
    current ``stage`` and, on the happy path, calls ``record_success``. On exit
    with an error this:

    - stamps the trace_id onto any ``RetrievalError`` and emits a failed trace
      for the current stage, then lets it propagate;
    - at the gateway stage only, wraps an unexpected exception in
      ``RetrievalExecutionError`` (and emits a failed trace);
    - otherwise lets unexpected exceptions propagate untraced.

    Trace emission is best-effort: a failing sink is logged, never raised.
    """

    def __init__(
        self,
        request: RetrieveRequest,
        clock: Clock,
        trace_id_generator: TraceIdGenerator,
        trace_sink: RetrievalTraceSink,
    ) -> None:
        self._request = request
        self._clock = clock
        self._trace_sink = trace_sink
        self.trace_id = trace_id_generator.generate()
        self._start_time = clock.now()
        self._start_perf = perf_counter()
        self._stage: FailureStage | None = None
        # Summary before query normalization; replaced once validation passes.
        self._summary: dict = {
            "query": request.query,
            "retrieval_mode": request.retrieval_mode.value,
            "limit": request.limit,
            "service_name": request.scope.service_name,
            "tenant_id": request.scope.tenant_id,
        }
        # Partial result carried into a post-gateway failure trace.
        self._result_count = 0
        self._warnings: list[RetrievalWarning] = []

    def stage(self, stage: FailureStage) -> None:
        self._stage = stage

    def set_summary(self, summary: dict) -> None:
        self._summary = summary

    def set_partial(self, *, result_count: int, warnings: list[RetrievalWarning]) -> None:
        """Record gateway results so a later failure trace reflects them."""
        self._result_count = result_count
        self._warnings = warnings

    def record_success(
        self,
        *,
        result_count: int,
        warnings: list[RetrievalWarning],
        diagnostics: dict | None = None,
    ) -> None:
        self._emit(
            TraceStatus.SUCCESS,
            failure_stage=None,
            result_count=result_count,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def __enter__(self) -> "_TracedRetrieval":
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        if exc is None:
            return False

        if isinstance(exc, RetrievalError):
            exc.trace_id = self.trace_id
            self._emit_failed()
            return False

        # Unexpected exceptions are only traced + wrapped at the gateway stage;
        # elsewhere they propagate untouched (preserving prior behavior).
        if self._stage == FailureStage.GATEWAY_EXECUTION:
            self._emit_failed()
            raise RetrievalExecutionError(
                trace_id=self.trace_id,
                internal_message=f"Gateway execution failed: {str(exc)}",
                details={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            )

        return False

    def _emit_failed(self) -> None:
        self._emit(
            TraceStatus.FAILED,
            failure_stage=self._stage,
            result_count=self._result_count,
            warnings=self._warnings,
        )

    def _emit(
        self,
        status: TraceStatus,
        *,
        failure_stage: FailureStage | None,
        result_count: int,
        warnings: list[RetrievalWarning],
        diagnostics: dict | None = None,
    ) -> None:
        end_time = self._clock.now()
        duration_ms = round((perf_counter() - self._start_perf) * 1000, 2)

        trace = RetrievalTrace(
            trace_id=self.trace_id,
            correlation_id=self._request.correlation_id,
            status=status,
            failure_stage=failure_stage,
            request_summary=self._summary,
            timing={
                "start": self._start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_ms": duration_ms,
            },
            result_count=result_count,
            warnings=warnings,
            diagnostics=diagnostics or {},
        )

        try:
            self._trace_sink.emit(trace)
        except Exception as e:
            logger.warning(
                "Failed to emit %s trace trace_id=%s error=%s",
                status.value,
                self.trace_id,
                str(e),
                exc_info=True,
            )


class RetrieveUseCase:
    """Central orchestrator for retrieval operations.

    Validates requests, enforces scope, orchestrates retrieval through gateways,
    post-validates results, and emits traces.
    """

    def __init__(
        self,
        gateway: RetrievalGateway,
        scope_policy: ScopePolicy,
        clock: Clock,
        trace_id_generator: TraceIdGenerator,
        trace_sink: RetrievalTraceSink,
    ):
        self.gateway = gateway
        self.scope_policy = scope_policy
        self.clock = clock
        self.trace_id_generator = trace_id_generator
        self.trace_sink = trace_sink

    def execute(self, request: RetrieveRequest) -> RetrieveResult:
        """Execute retrieval with validation, scope enforcement, and tracing."""
        with _TracedRetrieval(
            request, self.clock, self.trace_id_generator, self.trace_sink
        ) as trace:
            # Validate request
            trace.stage(FailureStage.REQUEST_VALIDATION)
            normalized_query, original_query = self._normalize_and_validate_query(
                request.query, trace.trace_id
            )
            self._validate_limit(request.limit, trace.trace_id)
            trace.set_summary(
                self._build_request_summary(original_query, normalized_query, request)
            )

            # Evaluate scope
            trace.stage(FailureStage.SCOPE_POLICY)
            scope_decision = self.scope_policy.evaluate(request.scope)

            # Build effective request
            effective_request = EffectiveRetrieveRequest(
                normalized_query=normalized_query,
                original_query=original_query,
                retrieval_mode=request.retrieval_mode,
                limit=request.limit,
                validated_scope=scope_decision.validated_scope,
                correlation_id=request.correlation_id,
            )

            # Execute retrieval through gateway
            trace.stage(FailureStage.GATEWAY_EXECUTION)
            gateway_result = self.gateway.retrieve(effective_request)

            # Post-validate chunks (a failure here reports the gateway results)
            trace.stage(FailureStage.POST_VALIDATION)
            trace.set_partial(
                result_count=len(gateway_result.chunks),
                warnings=gateway_result.warnings,
            )
            self._post_validate_chunks(
                gateway_result.chunks, scope_decision.validated_scope, trace.trace_id
            )

            # Merge warnings from gateway and scope decision, emit success trace
            all_warnings = list(scope_decision.warnings) + list(gateway_result.warnings)
            trace.record_success(
                result_count=len(gateway_result.chunks),
                warnings=all_warnings,
                diagnostics=gateway_result.diagnostics,
            )

            return RetrieveResult(
                chunks=gateway_result.chunks,
                warnings=all_warnings,
                trace_id=trace.trace_id,
            )

    def _normalize_and_validate_query(
        self, query: str, trace_id: str
    ) -> tuple[str, str]:
        """Normalize query and validate it's not empty."""
        original_query = query
        normalized_query = query.strip()

        if not normalized_query:
            raise InvalidRetrievalRequestError(
                trace_id=trace_id,
                internal_message="Query must not be empty or whitespace-only",
                details={"original_query": original_query},
            )

        return normalized_query, original_query

    def _validate_limit(self, limit: int, trace_id: str) -> None:
        """Validate retrieval limit is within bounds."""
        if limit < 1:
            raise InvalidRetrievalRequestError(
                trace_id=trace_id,
                internal_message=f"Limit must be at least 1, got {limit}",
                details={"limit": limit, "min": 1},
            )

        if limit > 50:
            raise InvalidRetrievalRequestError(
                trace_id=trace_id,
                internal_message=f"Limit must not exceed 50, got {limit}",
                details={"limit": limit, "max": 50},
            )

    def _post_validate_chunks(
        self, chunks: list[RetrievedChunk], validated_scope: RetrievalScope, trace_id: str
    ) -> None:
        """Validate returned chunks have required metadata and match validated scope."""
        required_fields = [
            "service_name",
            "tenant_id",
            "collection",
            "source_type",
            "source_label",
        ]

        for i, chunk in enumerate(chunks):
            # Check required metadata fields
            missing_fields = [f for f in required_fields if f not in chunk.metadata]
            if missing_fields:
                raise RetrievedChunkValidationError(
                    trace_id=trace_id,
                    internal_message=f"Chunk {i} missing required metadata fields: {', '.join(missing_fields)}",
                    details={"chunk_index": i, "missing_fields": missing_fields},
                )

            # Check scope compliance - service_name and tenant_id
            if chunk.metadata["service_name"] != validated_scope.service_name:
                raise RetrievedChunkValidationError(
                    trace_id=trace_id,
                    internal_message=f"Chunk {i} service_name '{chunk.metadata['service_name']}' does not match validated scope '{validated_scope.service_name}'",
                    details={
                        "chunk_index": i,
                        "chunk_service_name": chunk.metadata["service_name"],
                        "scope_service_name": validated_scope.service_name,
                    },
                )

            if chunk.metadata["tenant_id"] != validated_scope.tenant_id:
                raise RetrievedChunkValidationError(
                    trace_id=trace_id,
                    internal_message=f"Chunk {i} tenant_id '{chunk.metadata['tenant_id']}' does not match validated scope '{validated_scope.tenant_id}'",
                    details={
                        "chunk_index": i,
                        "chunk_tenant_id": chunk.metadata["tenant_id"],
                        "scope_tenant_id": validated_scope.tenant_id,
                    },
                )

            # Check collection membership
            chunk_collection = chunk.metadata["collection"]
            if chunk_collection not in validated_scope.collections:
                raise RetrievedChunkValidationError(
                    trace_id=trace_id,
                    internal_message=f"Chunk {i} collection '{chunk_collection}' is not in validated scope collections {validated_scope.collections}",
                    details={
                        "chunk_index": i,
                        "chunk_collection": chunk_collection,
                        "scope_collections": validated_scope.collections,
                    },
                )

            # Check filter constraints
            for field, constraint_value in validated_scope.filters.items():
                chunk_value = chunk.metadata.get(field)

                # Support primitive equality: field == value
                if not isinstance(constraint_value, list):
                    if chunk_value != constraint_value:
                        raise RetrievedChunkValidationError(
                            trace_id=trace_id,
                            internal_message=f"Chunk {i} filter constraint failed: {field}={chunk_value} does not match required value {constraint_value}",
                            details={
                                "chunk_index": i,
                                "filter_field": field,
                                "chunk_value": chunk_value,
                                "required_value": constraint_value,
                            },
                        )
                # Support list membership: field IN [values]
                else:
                    if chunk_value not in constraint_value:
                        raise RetrievedChunkValidationError(
                            trace_id=trace_id,
                            internal_message=f"Chunk {i} filter constraint failed: {field}={chunk_value} is not in allowed values {constraint_value}",
                            details={
                                "chunk_index": i,
                                "filter_field": field,
                                "chunk_value": chunk_value,
                                "allowed_values": constraint_value,
                            },
                        )

    def _build_request_summary(
        self, original_query: str, normalized_query: str, request: RetrieveRequest
    ) -> dict:
        """Build request summary for trace."""
        return {
            "original_query": original_query,
            "normalized_query": normalized_query,
            "retrieval_mode": request.retrieval_mode.value,
            "limit": request.limit,
            "service_name": request.scope.service_name,
            "tenant_id": request.scope.tenant_id,
        }
