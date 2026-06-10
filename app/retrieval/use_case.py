"""RetrieveUseCase - central orchestrator of the Retrieval Core."""
import logging
from dataclasses import dataclass
from time import perf_counter

from app.retrieval.types import (
    RetrieveRequest,
    RetrievalScope,
    EffectiveRetrieveRequest,
    RetrievedChunk,
    RetrievalWarning,
    RetrievalTrace,
    TraceStatus,
    FailureStage,
    RetrievalMode,
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
        trace_id = self.trace_id_generator.generate()
        start_time = self.clock.now()
        start_perf = perf_counter()

        try:
            # Validate request
            try:
                normalized_query, original_query = self._normalize_and_validate_query(
                    request.query, trace_id
                )
                self._validate_limit(request.limit, trace_id)
            except InvalidRetrievalRequestError as e:
                # Request validation failed - emit trace before re-raising
                self._emit_failed_trace(
                    trace_id=trace_id,
                    correlation_id=request.correlation_id,
                    request_summary={
                        "query": request.query,
                        "retrieval_mode": request.retrieval_mode.value,
                        "limit": request.limit,
                        "service_name": request.scope.service_name,
                        "tenant_id": request.scope.tenant_id,
                    },
                    start_time=start_time,
                    start_perf=start_perf,
                    result_count=0,
                    warnings=[],
                    failure_stage=FailureStage.REQUEST_VALIDATION,
                )
                raise

            # Evaluate scope
            try:
                scope_decision = self.scope_policy.evaluate(request.scope)
            except RetrievalError as e:
                # Re-raise with proper trace_id and failure stage
                e.trace_id = trace_id
                self._emit_failed_trace(
                    trace_id=trace_id,
                    correlation_id=request.correlation_id,
                    request_summary=self._build_request_summary(
                        original_query, normalized_query, request
                    ),
                    start_time=start_time,
                    start_perf=start_perf,
                    result_count=0,
                    warnings=[],
                    failure_stage=FailureStage.SCOPE_POLICY,
                )
                raise

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
            try:
                gateway_result = self.gateway.retrieve(effective_request)
            except RetrievalError as e:
                # Domain error from gateway - propagate with proper trace_id
                e.trace_id = trace_id
                self._emit_failed_trace(
                    trace_id=trace_id,
                    correlation_id=request.correlation_id,
                    request_summary=self._build_request_summary(
                        original_query, normalized_query, request
                    ),
                    start_time=start_time,
                    start_perf=start_perf,
                    result_count=0,
                    warnings=[],
                    failure_stage=FailureStage.GATEWAY_EXECUTION,
                )
                raise
            except Exception as e:
                # Unexpected exception - wrap in RetrievalExecutionError
                self._emit_failed_trace(
                    trace_id=trace_id,
                    correlation_id=request.correlation_id,
                    request_summary=self._build_request_summary(
                        original_query, normalized_query, request
                    ),
                    start_time=start_time,
                    start_perf=start_perf,
                    result_count=0,
                    warnings=[],
                    failure_stage=FailureStage.GATEWAY_EXECUTION,
                )
                raise RetrievalExecutionError(
                    trace_id=trace_id,
                    internal_message=f"Gateway execution failed: {str(e)}",
                    details={"exception_type": type(e).__name__, "exception_message": str(e)},
                )

            # Post-validate chunks
            try:
                self._post_validate_chunks(
                    gateway_result.chunks, scope_decision.validated_scope, trace_id
                )
            except RetrievedChunkValidationError as e:
                self._emit_failed_trace(
                    trace_id=trace_id,
                    correlation_id=request.correlation_id,
                    request_summary=self._build_request_summary(
                        original_query, normalized_query, request
                    ),
                    start_time=start_time,
                    start_perf=start_perf,
                    result_count=len(gateway_result.chunks),
                    warnings=gateway_result.warnings,
                    failure_stage=FailureStage.POST_VALIDATION,
                )
                raise

            # Merge warnings from gateway and scope decision
            all_warnings = list(scope_decision.warnings) + list(gateway_result.warnings)

            # Emit success trace with gateway diagnostics
            self._emit_success_trace(
                trace_id=trace_id,
                correlation_id=request.correlation_id,
                request_summary=self._build_request_summary(
                    original_query, normalized_query, request
                ),
                start_time=start_time,
                start_perf=start_perf,
                result_count=len(gateway_result.chunks),
                warnings=all_warnings,
                diagnostics=gateway_result.diagnostics,
            )

            return RetrieveResult(
                chunks=gateway_result.chunks,
                warnings=all_warnings,
                trace_id=trace_id,
            )

        except RetrievalError:
            # Domain errors already have trace emitted - just re-raise
            raise

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

    def _emit_success_trace(
        self,
        trace_id: str,
        correlation_id: str | None,
        request_summary: dict,
        start_time,
        start_perf: float,
        result_count: int,
        warnings: list[RetrievalWarning],
        diagnostics: dict | None = None,
    ) -> None:
        """Emit a successful retrieval trace."""
        end_time = self.clock.now()
        duration_ms = round((perf_counter() - start_perf) * 1000, 2)

        trace = RetrievalTrace(
            trace_id=trace_id,
            correlation_id=correlation_id,
            status=TraceStatus.SUCCESS,
            failure_stage=None,
            request_summary=request_summary,
            timing={
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_ms": duration_ms,
            },
            result_count=result_count,
            warnings=warnings,
            diagnostics=diagnostics or {},
        )

        try:
            self.trace_sink.emit(trace)
        except Exception as e:
            logger.warning(
                "Failed to emit success trace trace_id=%s error=%s",
                trace_id,
                str(e),
                exc_info=True,
            )

    def _emit_failed_trace(
        self,
        trace_id: str,
        correlation_id: str | None,
        request_summary: dict,
        start_time,
        start_perf: float,
        result_count: int,
        warnings: list[RetrievalWarning],
        failure_stage: FailureStage,
    ) -> None:
        """Emit a failed retrieval trace."""
        end_time = self.clock.now()
        duration_ms = round((perf_counter() - start_perf) * 1000, 2)

        trace = RetrievalTrace(
            trace_id=trace_id,
            correlation_id=correlation_id,
            status=TraceStatus.FAILED,
            failure_stage=failure_stage,
            request_summary=request_summary,
            timing={
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_ms": duration_ms,
            },
            result_count=result_count,
            warnings=warnings,
        )

        try:
            self.trace_sink.emit(trace)
        except Exception as e:
            logger.warning(
                "Failed to emit failure trace trace_id=%s failure_stage=%s error=%s",
                trace_id,
                failure_stage.value,
                str(e),
                exc_info=True,
            )
