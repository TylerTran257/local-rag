"""LegacyDocumentRetrievalAdapter - bridges DocumentService to Retrieval Core."""
from fastapi import HTTPException

from app.retrieval.types import (
    EffectiveRetrieveRequest,
    RetrievalGatewayResult,
    RetrievedChunk,
    RetrievalWarning,
    WarningCode,
    WarningSeverity,
    RetrievalMode,
)
from app.retrieval.errors import (
    NoIndexedCorpusError,
    RetrievalExecutionError,
)


class LegacyDocumentRetrievalAdapter:
    """Retrieval gateway implementation that bridges DocumentService.

    Normalizes dict results from DocumentService into RetrievedChunk objects
    with sentinel default metadata, and translates exceptions into domain errors.
    """

    # Sentinel defaults from ADR 003
    SENTINEL_SERVICE_NAME = "local-rag"
    SENTINEL_TENANT_ID = "default"
    SENTINEL_COLLECTION = "documents"
    SENTINEL_SOURCE_TYPE = "document"

    def __init__(self, document_service):
        """Initialize adapter with DocumentService instance.

        Args:
            document_service: Instance of app.services.document_service.DocumentService
        """
        self.document_service = document_service

    def retrieve(self, request: EffectiveRetrieveRequest) -> RetrievalGatewayResult:
        """Execute retrieval through DocumentService and normalize results.

        Args:
            request: Effective retrieve request after validation and scope enforcement

        Returns:
            RetrievalGatewayResult with normalized chunks, warnings, and diagnostics

        Raises:
            NoIndexedCorpusError: When DocumentService raises HTTPException(409)
            RetrievalExecutionError: For other errors from DocumentService
        """
        try:
            # Dispatch to appropriate DocumentService method based on mode
            if request.retrieval_mode == RetrievalMode.DENSE:
                results = self.document_service.retrieve_context_dense(
                    request.normalized_query, request.limit
                )
            elif request.retrieval_mode == RetrievalMode.LEXICAL:
                results = self.document_service.retrieve_context_lexical(
                    request.normalized_query, request.limit
                )
            elif request.retrieval_mode == RetrievalMode.HYBRID:
                results = self.document_service.retrieve_context_hybrid(
                    request.normalized_query, request.limit
                )
            else:
                # Should not happen due to validation, but handle defensively
                raise RetrievalExecutionError(
                    trace_id="unknown",
                    internal_message=f"Unsupported retrieval mode: {request.retrieval_mode}",
                    details={"mode": str(request.retrieval_mode)}
                )

            # Normalize results to RetrievedChunk with sentinel defaults
            chunks = []
            warnings = []

            for result_dict in results:
                chunk = self._normalize_to_retrieved_chunk(result_dict)
                chunks.append(chunk)

                # Emit warning for legacy metadata
                warning = RetrievalWarning(
                    code=WarningCode.LEGACY_METADATA_DEFAULTED,
                    severity=WarningSeverity.MEDIUM,
                    source="LegacyDocumentRetrievalAdapter",
                    message="Using sentinel defaults for metadata fields",
                    details={
                        "service_name": self.SENTINEL_SERVICE_NAME,
                        "tenant_id": self.SENTINEL_TENANT_ID,
                        "collection": self.SENTINEL_COLLECTION,
                        "source_type": self.SENTINEL_SOURCE_TYPE,
                    }
                )
                warnings.append(warning)

            return RetrievalGatewayResult(
                chunks=chunks,
                warnings=warnings,
                diagnostics={}
            )

        except HTTPException as e:
            # Translate HTTPException from DocumentService to domain errors
            if e.status_code == 409:
                raise NoIndexedCorpusError(
                    trace_id="unknown",  # Will be set by RetrieveUseCase
                    internal_message=f"No indexed corpus available: {e.detail}",
                    details={"status_code": e.status_code, "detail": e.detail}
                )
            else:
                raise RetrievalExecutionError(
                    trace_id="unknown",
                    internal_message=f"DocumentService HTTP error: {e.detail}",
                    details={"status_code": e.status_code, "detail": e.detail}
                )

        except Exception as e:
            # Wrap unexpected exceptions
            raise RetrievalExecutionError(
                trace_id="unknown",
                internal_message=f"DocumentService execution failed: {str(e)}",
                details={"exception_type": type(e).__name__, "exception_message": str(e)}
            )

    def _normalize_to_retrieved_chunk(self, result_dict: dict) -> RetrievedChunk:
        """Normalize DocumentService dict result to RetrievedChunk with sentinel defaults.

        Args:
            result_dict: Dict from DocumentService with keys: text, score, document_id,
                        chunk_index, original_filename

        Returns:
            RetrievedChunk with normalized metadata
        """
        return RetrievedChunk(
            content=result_dict["text"],
            score=result_dict["score"],
            metadata={
                # From DocumentService
                "document_id": result_dict["document_id"],
                "chunk_index": result_dict["chunk_index"],
                # Sentinel defaults (ADR 003)
                "service_name": self.SENTINEL_SERVICE_NAME,
                "tenant_id": self.SENTINEL_TENANT_ID,
                "collection": self.SENTINEL_COLLECTION,
                "source_type": self.SENTINEL_SOURCE_TYPE,
                "source_label": result_dict["original_filename"],
            }
        )
