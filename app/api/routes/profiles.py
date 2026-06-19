from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.api.schemas import ProfileResponse, ProfileUpsertRequest
from app.auth import AuthorizationError, Principal, require_principal
from app.profiles import ServiceProfile
from app.profiles.store import ProfileStore

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_profile_access(principal: Principal, service_name: str) -> None:
    """A key may manage a profile only for services it is granted (or admin)."""
    if not principal.allows_service(service_name):
        raise AuthorizationError(
            f"Key '{principal.key_id}' may not manage profile for service '{service_name}'",
            details={"service_name": service_name},
        )


def _store(request: Request) -> ProfileStore:
    return request.app.state.profile_store


def _to_response(profile: ServiceProfile) -> ProfileResponse:
    return ProfileResponse(**profile.to_dict())


def _merge(body: ProfileUpsertRequest, existing: ServiceProfile) -> ServiceProfile:
    """Overlay the request's set fields onto the existing/default profile."""
    return ServiceProfile(
        service_name=body.service_name,
        embedding_model=body.embedding_model or existing.embedding_model,
        chunk_size=body.chunk_size if body.chunk_size is not None else existing.chunk_size,
        chunk_overlap=body.chunk_overlap if body.chunk_overlap is not None else existing.chunk_overlap,
        dense_limit=body.dense_limit if body.dense_limit is not None else existing.dense_limit,
        lexical_limit=body.lexical_limit if body.lexical_limit is not None else existing.lexical_limit,
        fusion_rrf_k=body.fusion_rrf_k if body.fusion_rrf_k is not None else existing.fusion_rrf_k,
        default_mode=body.default_mode or existing.default_mode,
        generation_overrides=body.generation_overrides or existing.generation_overrides,
    )


@router.post("/profiles", response_model=ProfileResponse)
def upsert_profile(
    request: Request,
    body: ProfileUpsertRequest,
    principal: Principal = Depends(require_principal),
):
    """Create or update the calling service's config profile.

    Other agents/services use this to register their own ingestion + retrieval
    configuration (chunking, embedding model, retrieval defaults). The embedding
    model is fixed once a profile exists.
    """
    _require_profile_access(principal, body.service_name)

    store = _store(request)
    existing = store.get(body.service_name)
    merged = _merge(body, existing)
    # ProfileEmbeddingModelImmutableError propagates to the exception handler.
    saved = store.upsert(merged)
    return _to_response(saved)


@router.get("/profiles/{service_name}", response_model=ProfileResponse)
def get_profile(
    request: Request,
    service_name: str,
    principal: Principal = Depends(require_principal),
):
    """Return a service's config profile (defaults if none is registered)."""
    _require_profile_access(principal, service_name)
    profile = _store(request).get(service_name)
    return _to_response(profile)
