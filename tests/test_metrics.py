"""The /metrics endpoint exposes Prometheus counters and every response carries a trace id."""
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.auth import ApiKeyRegistry
from app.composition import MetadataAwareRuntime
from app.main import create_app


@pytest.fixture
def client():
    runtime = MetadataAwareRuntime(
        retrieve_use_case=Mock(), ingest_use_case=Mock(), gateway=Mock()
    )
    app = create_app(
        generation_service=Mock(),
        metadata_aware_runtime=runtime,
        api_key_registry=ApiKeyRegistry.from_entries([]),
    )
    return TestClient(app)


def test_metrics_endpoint_exposes_counters(client):
    # Generate at least one request to record.
    client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text
    assert "http_request_duration_seconds" in resp.text


def test_responses_carry_trace_id_header(client):
    resp = client.get("/health")
    assert resp.headers.get("X-Trace-Id")
