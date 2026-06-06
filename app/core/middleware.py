import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)


def register_request_timing_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_request_timing(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        started_at = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%s",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "event=http_request_completed request_id=%s method=%s path=%s status_code=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
