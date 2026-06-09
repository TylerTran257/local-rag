"""Scope policy implementations for the Retrieval Core."""
from app.retrieval.errors import InvalidRetrievalRequestError
from app.retrieval.types import (
    RetrievalScope,
    RetrievalWarning,
    ScopeDecision,
    WarningCode,
    WarningSeverity,
)


class PassthroughScopePolicy:
    """Validates structural correctness of scope only (no authorization)."""

    def evaluate(self, scope: RetrievalScope) -> ScopeDecision:
        """Evaluate scope for structural correctness."""
        # Validate required fields are non-empty
        if not scope.service_name or scope.service_name.strip() == "":
            raise InvalidRetrievalRequestError(
                trace_id="unknown",  # Will be replaced by use case
                internal_message="Scope validation failed: service_name must be non-empty",
                details={"service_name": scope.service_name}
            )

        if not scope.tenant_id or scope.tenant_id.strip() == "":
            raise InvalidRetrievalRequestError(
                trace_id="unknown",
                internal_message="Scope validation failed: tenant_id must be non-empty",
                details={"tenant_id": scope.tenant_id}
            )

        # Structural validation passed
        return ScopeDecision(
            validated_scope=scope,
            policy_name="PassthroughScopePolicy",
            warnings=[]
        )


class NamespacePolicy:
    def __init__(self, allowed_collections: set[str] | None = None):
        self._allowed_collections = set(allowed_collections) if allowed_collections else None

    def evaluate(self, scope: RetrievalScope) -> ScopeDecision:
        if not scope.service_name or scope.service_name.strip() == "":
            raise InvalidRetrievalRequestError(
                trace_id="unknown",
                internal_message="Scope validation failed: service_name must be non-empty",
                details={"service_name": scope.service_name},
            )

        if not scope.tenant_id or scope.tenant_id.strip() == "":
            raise InvalidRetrievalRequestError(
                trace_id="unknown",
                internal_message="Scope validation failed: tenant_id must be non-empty",
                details={"tenant_id": scope.tenant_id},
            )

        if not scope.collections:
            raise InvalidRetrievalRequestError(
                trace_id="unknown",
                internal_message="Scope validation failed: collections must be non-empty",
                details={"collections": scope.collections},
            )

        empty_collections = [collection for collection in scope.collections if not collection or collection.strip() == ""]
        if empty_collections:
            raise InvalidRetrievalRequestError(
                trace_id="unknown",
                internal_message="Scope validation failed: collection names must be non-empty",
                details={"collections": scope.collections},
            )

        if self._allowed_collections is not None:
            invalid_collections = [
                collection for collection in scope.collections if collection not in self._allowed_collections
            ]
            if invalid_collections:
                raise InvalidRetrievalRequestError(
                    trace_id="unknown",
                    internal_message="Scope validation failed: collections are not allowed",
                    details={
                        "collections": scope.collections,
                        "invalid_collections": invalid_collections,
                        "allowed_collections": sorted(self._allowed_collections),
                    },
                )

        warnings = []
        if scope.service_name == "local-rag" and scope.tenant_id == "default":
            warnings.append(
                RetrievalWarning(
                    code=WarningCode.NAMESPACE_DEFAULT_SCOPE,
                    severity=WarningSeverity.LOW,
                    source="NamespacePolicy",
                    message="Retrieve request used sentinel default scope values",
                    details={
                        "service_name": scope.service_name,
                        "tenant_id": scope.tenant_id,
                    },
                )
            )

        return ScopeDecision(
            validated_scope=scope,
            policy_name="NamespacePolicy",
            warnings=warnings,
        )
