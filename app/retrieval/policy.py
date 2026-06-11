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
    def __init__(
        self,
        allowed_collections: set[str] | None = None,
        allowed_services: set[str] | None = None,
        service_collection_rules: dict[str, set[str]] | None = None,
        service_required_filters: dict[str, set[str]] | None = None,
    ):
        self._allowed_collections = set(allowed_collections) if allowed_collections else None
        self._allowed_services = set(allowed_services) if allowed_services else None
        self._service_collection_rules = (
            {k: set(v) for k, v in service_collection_rules.items()}
            if service_collection_rules
            else None
        )
        self._service_required_filters = (
            {k: set(v) for k, v in service_required_filters.items()}
            if service_required_filters
            else None
        )

    def evaluate(self, scope: RetrievalScope) -> ScopeDecision:
        # Validate non-empty service_name
        if not scope.service_name or scope.service_name.strip() == "":
            raise InvalidRetrievalRequestError(
                trace_id="unknown",
                internal_message="Scope validation failed: service_name must be non-empty",
                details={"service_name": scope.service_name},
            )

        # Validate non-empty tenant_id
        if not scope.tenant_id or scope.tenant_id.strip() == "":
            raise InvalidRetrievalRequestError(
                trace_id="unknown",
                internal_message="Scope validation failed: tenant_id must be non-empty",
                details={"tenant_id": scope.tenant_id},
            )

        # Validate non-empty collections
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

        # Validate allowed services (fail closed)
        if self._allowed_services is not None:
            if scope.service_name not in self._allowed_services:
                raise InvalidRetrievalRequestError(
                    trace_id="unknown",
                    internal_message=f"Scope validation failed: service '{scope.service_name}' is not allowed",
                    details={
                        "service_name": scope.service_name,
                        "allowed_services": sorted(self._allowed_services),
                    },
                )

        # Validate global allowed collections
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

        # Validate service-specific collection rules (fail closed)
        denied_collections: list[str] = []
        if self._service_collection_rules is not None:
            service_rules = self._service_collection_rules.get(scope.service_name)
            if service_rules is not None:
                denied_collections = [
                    collection for collection in scope.collections
                    if collection not in service_rules
                ]
                if denied_collections:
                    raise InvalidRetrievalRequestError(
                        trace_id="unknown",
                        internal_message=(
                            f"Scope validation failed: collections {denied_collections} "
                            f"are not allowed for service '{scope.service_name}'"
                        ),
                        details={
                            "service_name": scope.service_name,
                            "collections": scope.collections,
                            "denied_collections": denied_collections,
                            "allowed_collections": sorted(service_rules),
                        },
                    )

        # Validate service-specific required filters (fail closed)
        enforced_filters: dict[str, str] = {}
        if self._service_required_filters is not None:
            required_keys = self._service_required_filters.get(scope.service_name)
            if required_keys is not None:
                missing_keys = [
                    key for key in sorted(required_keys)
                    if key not in scope.filters
                ]
                if missing_keys:
                    raise InvalidRetrievalRequestError(
                        trace_id="unknown",
                        internal_message=(
                            f"Scope validation failed: required filter keys {missing_keys} "
                            f"missing for service '{scope.service_name}'"
                        ),
                        details={
                            "service_name": scope.service_name,
                            "missing_filter_keys": missing_keys,
                            "required_filter_keys": sorted(required_keys),
                            "provided_filter_keys": sorted(scope.filters.keys()),
                        },
                    )
                # Record the enforced filters (the required keys and their values)
                enforced_filters = {
                    key: scope.filters[key]
                    for key in required_keys
                    if key in scope.filters
                }

        # Emit warnings for sentinel defaults
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
            denied_collections=denied_collections,
            enforced_filters=enforced_filters,
        )
