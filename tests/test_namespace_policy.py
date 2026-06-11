"""Tests for enhanced NamespacePolicy with service-level controls."""
import pytest
from app.retrieval import (
    RetrievalScope,
    ScopeDecision,
    InvalidRetrievalRequestError,
    WarningCode,
    WarningSeverity,
)
from app.retrieval.policy import NamespacePolicy, PassthroughScopePolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scope(
    service_name: str = "my-service",
    tenant_id: str = "tenant-1",
    collections: list[str] | None = None,
    filters: dict | None = None,
) -> RetrievalScope:
    return RetrievalScope(
        service_name=service_name,
        tenant_id=tenant_id,
        collections=["documents"] if collections is None else collections,
        filters=filters or {},
    )


# ---------------------------------------------------------------------------
# Test: allowed_services enforcement
# ---------------------------------------------------------------------------

class TestAllowedServices:
    def test_rejects_disallowed_service(self):
        policy = NamespacePolicy(allowed_services={"service-a", "service-b"})
        scope = _make_scope(service_name="service-x")

        with pytest.raises(InvalidRetrievalRequestError) as exc_info:
            policy.evaluate(scope)

        error = exc_info.value
        assert "service" in error.internal_message.lower()
        assert "service-x" in error.internal_message
        assert error.details["service_name"] == "service-x"
        assert "allowed_services" in error.details

    def test_accepts_allowed_service(self):
        policy = NamespacePolicy(allowed_services={"service-a", "service-b"})
        scope = _make_scope(service_name="service-a")

        decision = policy.evaluate(scope)

        assert isinstance(decision, ScopeDecision)
        assert decision.validated_scope == scope
        assert decision.policy_name == "NamespacePolicy"

    def test_no_allowed_services_configured_permits_any_service(self):
        policy = NamespacePolicy()  # allowed_services=None
        scope = _make_scope(service_name="any-service")

        decision = policy.evaluate(scope)

        assert decision.validated_scope == scope


# ---------------------------------------------------------------------------
# Test: service_collection_rules enforcement
# ---------------------------------------------------------------------------

class TestServiceCollectionRules:
    def test_rejects_collections_not_in_service_rules(self):
        policy = NamespacePolicy(
            service_collection_rules={
                "my-service": {"docs", "notes"},
            }
        )
        scope = _make_scope(
            service_name="my-service",
            collections=["docs", "secrets"],
        )

        with pytest.raises(InvalidRetrievalRequestError) as exc_info:
            policy.evaluate(scope)

        error = exc_info.value
        assert "secrets" in error.internal_message
        assert "my-service" in error.internal_message
        assert error.details["denied_collections"] == ["secrets"]

    def test_accepts_collections_in_service_rules(self):
        policy = NamespacePolicy(
            service_collection_rules={
                "my-service": {"docs", "notes"},
            }
        )
        scope = _make_scope(
            service_name="my-service",
            collections=["docs", "notes"],
        )

        decision = policy.evaluate(scope)

        assert decision.validated_scope == scope
        assert decision.denied_collections == []

    def test_no_rule_for_service_permits_all_collections(self):
        policy = NamespacePolicy(
            service_collection_rules={
                "other-service": {"restricted"},
            }
        )
        scope = _make_scope(
            service_name="my-service",
            collections=["anything-goes"],
        )

        decision = policy.evaluate(scope)

        assert decision.validated_scope == scope

    def test_returns_denied_collections_in_scope_decision_on_rejection(self):
        """The denied_collections list should be populated in the error details."""
        policy = NamespacePolicy(
            service_collection_rules={
                "my-service": {"docs"},
            }
        )
        scope = _make_scope(
            service_name="my-service",
            collections=["docs", "forbidden-a", "forbidden-b"],
        )

        with pytest.raises(InvalidRetrievalRequestError) as exc_info:
            policy.evaluate(scope)

        error = exc_info.value
        denied = error.details["denied_collections"]
        assert "forbidden-a" in denied
        assert "forbidden-b" in denied
        assert "docs" not in denied


# ---------------------------------------------------------------------------
# Test: service_required_filters enforcement
# ---------------------------------------------------------------------------

class TestServiceRequiredFilters:
    def test_rejects_missing_required_filter(self):
        policy = NamespacePolicy(
            service_required_filters={
                "my-service": {"department", "region"},
            }
        )
        scope = _make_scope(
            service_name="my-service",
            filters={"department": "eng"},
        )

        with pytest.raises(InvalidRetrievalRequestError) as exc_info:
            policy.evaluate(scope)

        error = exc_info.value
        assert "region" in error.internal_message
        assert "my-service" in error.internal_message
        assert "region" in error.details["missing_filter_keys"]

    def test_accepts_request_with_all_required_filters_present(self):
        policy = NamespacePolicy(
            service_required_filters={
                "my-service": {"department", "region"},
            }
        )
        scope = _make_scope(
            service_name="my-service",
            filters={"department": "eng", "region": "us-west"},
        )

        decision = policy.evaluate(scope)

        assert decision.validated_scope == scope
        assert decision.enforced_filters == {
            "department": "eng",
            "region": "us-west",
        }

    def test_returns_enforced_filters_in_scope_decision(self):
        policy = NamespacePolicy(
            service_required_filters={
                "my-service": {"department"},
            }
        )
        scope = _make_scope(
            service_name="my-service",
            filters={"department": "eng", "extra_filter": "value"},
        )

        decision = policy.evaluate(scope)

        # enforced_filters only contains the required keys, not extras
        assert decision.enforced_filters == {"department": "eng"}

    def test_no_rule_for_service_permits_any_filters(self):
        policy = NamespacePolicy(
            service_required_filters={
                "other-service": {"must-have"},
            }
        )
        scope = _make_scope(
            service_name="my-service",
            filters={},
        )

        decision = policy.evaluate(scope)

        assert decision.validated_scope == scope
        assert decision.enforced_filters == {}


# ---------------------------------------------------------------------------
# Test: backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_no_rules_configured_behaves_like_original_namespace_policy(self):
        """With no new parameters, NamespacePolicy behaves exactly as before."""
        policy = NamespacePolicy()
        scope = _make_scope(service_name="any-service")

        decision = policy.evaluate(scope)

        assert decision.validated_scope == scope
        assert decision.policy_name == "NamespacePolicy"
        assert decision.warnings == []
        assert decision.denied_collections == []
        assert decision.enforced_filters == {}

    def test_allowed_collections_still_works(self):
        policy = NamespacePolicy(allowed_collections={"docs", "notes"})
        scope = _make_scope(collections=["docs"])

        decision = policy.evaluate(scope)
        assert decision.validated_scope == scope

    def test_allowed_collections_rejects_disallowed(self):
        policy = NamespacePolicy(allowed_collections={"docs"})
        scope = _make_scope(collections=["forbidden"])

        with pytest.raises(InvalidRetrievalRequestError):
            policy.evaluate(scope)

    def test_emits_warnings_for_default_scope_values(self):
        policy = NamespacePolicy()
        scope = _make_scope(
            service_name="local-rag",
            tenant_id="default",
        )

        decision = policy.evaluate(scope)

        assert len(decision.warnings) == 1
        warning = decision.warnings[0]
        assert warning.code == WarningCode.NAMESPACE_DEFAULT_SCOPE
        assert warning.severity == WarningSeverity.LOW
        assert warning.source == "NamespacePolicy"
        assert "sentinel" in warning.message.lower() or "default" in warning.message.lower()

    def test_empty_service_name_rejected(self):
        policy = NamespacePolicy()
        scope = _make_scope(service_name="")

        with pytest.raises(InvalidRetrievalRequestError):
            policy.evaluate(scope)

    def test_empty_tenant_id_rejected(self):
        policy = NamespacePolicy()
        scope = _make_scope(tenant_id="")

        with pytest.raises(InvalidRetrievalRequestError):
            policy.evaluate(scope)

    def test_empty_collections_rejected(self):
        policy = NamespacePolicy()
        scope = _make_scope(collections=[])

        with pytest.raises(InvalidRetrievalRequestError):
            policy.evaluate(scope)


# ---------------------------------------------------------------------------
# Test: PassthroughScopePolicy still works unchanged
# ---------------------------------------------------------------------------

class TestPassthroughScopePolicyUnchanged:
    def test_passthrough_accepts_valid_scope(self):
        policy = PassthroughScopePolicy()
        scope = _make_scope()

        decision = policy.evaluate(scope)

        assert isinstance(decision, ScopeDecision)
        assert decision.validated_scope == scope
        assert decision.policy_name == "PassthroughScopePolicy"
        assert decision.warnings == []

    def test_passthrough_rejects_empty_service_name(self):
        policy = PassthroughScopePolicy()
        scope = _make_scope(service_name="")

        with pytest.raises(InvalidRetrievalRequestError):
            policy.evaluate(scope)

    def test_passthrough_rejects_empty_tenant_id(self):
        policy = PassthroughScopePolicy()
        scope = _make_scope(tenant_id="")

        with pytest.raises(InvalidRetrievalRequestError):
            policy.evaluate(scope)

    def test_passthrough_scope_decision_has_new_fields_with_defaults(self):
        """PassthroughScopePolicy returns ScopeDecision which now has
        denied_collections and enforced_filters with empty defaults."""
        policy = PassthroughScopePolicy()
        scope = _make_scope()

        decision = policy.evaluate(scope)

        assert decision.denied_collections == []
        assert decision.enforced_filters == {}


# ---------------------------------------------------------------------------
# Test: combined rules
# ---------------------------------------------------------------------------

class TestCombinedRules:
    def test_allowed_services_checked_before_collection_rules(self):
        """If service is not in allowed_services, reject even if collection rules exist."""
        policy = NamespacePolicy(
            allowed_services={"service-a"},
            service_collection_rules={
                "service-b": {"docs"},
            },
        )
        scope = _make_scope(
            service_name="service-b",
            collections=["docs"],
        )

        with pytest.raises(InvalidRetrievalRequestError) as exc_info:
            policy.evaluate(scope)

        # Should fail on allowed_services, not collection rules
        assert "not allowed" in exc_info.value.internal_message
        assert "service-b" in exc_info.value.internal_message

    def test_global_allowed_collections_checked_before_service_rules(self):
        """Global allowed_collections is checked before service_collection_rules."""
        policy = NamespacePolicy(
            allowed_collections={"docs"},
            service_collection_rules={
                "my-service": {"docs", "notes"},
            },
        )
        scope = _make_scope(
            service_name="my-service",
            collections=["notes"],  # allowed by service rules, denied by global
        )

        with pytest.raises(InvalidRetrievalRequestError) as exc_info:
            policy.evaluate(scope)

        assert "not allowed" in exc_info.value.internal_message.lower()

    def test_all_rules_pass_returns_enriched_decision(self):
        policy = NamespacePolicy(
            allowed_services={"my-service"},
            allowed_collections={"docs", "notes"},
            service_collection_rules={
                "my-service": {"docs", "notes"},
            },
            service_required_filters={
                "my-service": {"department"},
            },
        )
        scope = _make_scope(
            service_name="my-service",
            collections=["docs"],
            filters={"department": "eng"},
        )

        decision = policy.evaluate(scope)

        assert decision.validated_scope == scope
        assert decision.denied_collections == []
        assert decision.enforced_filters == {"department": "eng"}
