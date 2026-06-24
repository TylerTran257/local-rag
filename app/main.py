import logging

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import (
    answer_router,
    delete_router,
    documents_v2_router,
    health_router,
    profiles_router,
    retrieve_router,
)
from app.auth import ApiKeyRegistry
from app.composition import MetadataAwareRuntime, build_metadata_aware_runtime
from app.core.logging import configure_logging
from app.core.metrics import register_metrics
from app.core.middleware import register_request_timing_middleware
from app.db.database import Base, engine
from app.services.generation_service import GenerationService
from app.settings import settings

logger = logging.getLogger(__name__)


def create_app(
    generation_service=None,
    *,
    metadata_aware_runtime: MetadataAwareRuntime | None = None,
    api_key_registry: ApiKeyRegistry | None = None,
) -> FastAPI:
    """Create and wire the FastAPI application.

    Args:
        generation_service: Override for GenerationService (primarily for tests).
        metadata_aware_runtime: Pre-built MetadataAwareRuntime to use. When
            omitted, the factory builds production defaults.
        api_key_registry: Override for the API key registry (primarily for
            tests). When omitted, it is loaded from config (file or env).

    Returns:
        Configured FastAPI application.
    """
    configure_logging()
    logger.info("event=app_started")
    app = FastAPI()
    register_request_timing_middleware(app)
    if settings.metrics_enabled:
        register_metrics(app)
    register_exception_handlers(app)

    Base.metadata.create_all(bind=engine)

    # Build metadata-aware runtime
    runtime = metadata_aware_runtime or build_metadata_aware_runtime()

    # Seed profiles from the configured file (idempotent).
    if runtime.profile_store is not None:
        runtime.profile_store.seed_from_file(settings.profiles_file)

    # Generation service
    resolved_generation_service = generation_service or GenerationService()

    # API key registry (auth)
    resolved_registry = api_key_registry or ApiKeyRegistry.from_config(
        api_keys_file=settings.api_keys_file
    )

    # Wire app state
    app.state.retrieve_use_case = runtime.retrieve_use_case
    app.state.ingest_use_case = runtime.ingest_use_case
    app.state.delete_use_case = runtime.delete_use_case
    app.state.generation_service = resolved_generation_service
    app.state.profile_store = runtime.profile_store
    app.state.api_key_registry = resolved_registry

    # Register narrowed route set
    app.include_router(health_router)
    app.include_router(documents_v2_router)
    app.include_router(retrieve_router)
    app.include_router(answer_router)
    app.include_router(profiles_router)
    app.include_router(delete_router)

    return app
