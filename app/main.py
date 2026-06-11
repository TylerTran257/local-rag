import logging

from fastapi import FastAPI

from app.api.routes import (
    answer_router,
    documents_v2_router,
    health_router,
    retrieve_router,
)
from app.composition import MetadataAwareRuntime, build_metadata_aware_runtime
from app.core.logging import configure_logging
from app.core.middleware import register_request_timing_middleware
from app.db.database import Base, engine
from app.services.generation_service import GenerationService

logger = logging.getLogger(__name__)


def create_app(
    generation_service=None,
    *,
    metadata_aware_runtime: MetadataAwareRuntime | None = None,
) -> FastAPI:
    """Create and wire the FastAPI application.

    Args:
        generation_service: Override for GenerationService (primarily for tests).
        metadata_aware_runtime: Pre-built MetadataAwareRuntime to use.
            When omitted, the factory builds production defaults via
            ``build_metadata_aware_runtime()``.

    Returns:
        Configured FastAPI application.
    """
    configure_logging()
    logger.info("event=app_started")
    app = FastAPI()
    register_request_timing_middleware(app)

    Base.metadata.create_all(bind=engine)

    # Build metadata-aware runtime
    runtime = metadata_aware_runtime or build_metadata_aware_runtime()

    # Generation service
    resolved_generation_service = generation_service or GenerationService()

    # Wire app state
    app.state.retrieve_use_case = runtime.retrieve_use_case
    app.state.ingest_use_case = runtime.ingest_use_case
    app.state.generation_service = resolved_generation_service

    # Register narrowed route set
    app.include_router(health_router)
    app.include_router(documents_v2_router)
    app.include_router(retrieve_router)
    app.include_router(answer_router)

    return app
