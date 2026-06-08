"""Scope policy implementations for the Retrieval Core."""
from app.retrieval.errors import InvalidRetrievalRequestError
from app.retrieval.types import RetrievalScope, ScopeDecision


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
