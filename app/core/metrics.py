"""Optional Prometheus metrics: request counters, latency, and a /metrics route."""
from __future__ import annotations

import logging
from time import perf_counter

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)


def _route_template(request: Request) -> str:
    """Return the matched route template to keep label cardinality bounded."""
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path


def register_metrics(app: FastAPI) -> None:
    """Attach request metrics middleware and expose ``GET /metrics``.

    Uses a dedicated registry so repeated app construction (tests) does not
    collide on the default global registry.
    """
    registry = CollectorRegistry()
    requests_total = Counter(
        "http_requests_total",
        "Total HTTP requests.",
        labelnames=("method", "path", "status"),
        registry=registry,
    )
    request_duration = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds.",
        labelnames=("method", "path"),
        registry=registry,
    )

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        started_at = perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            path = _route_template(request)
            request_duration.labels(request.method, path).observe(
                perf_counter() - started_at
            )
            requests_total.labels(request.method, path, str(status)).inc()

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
