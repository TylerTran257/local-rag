"""Social style retrieval API routes."""
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.retrieval_helpers import map_retrieval_error_to_response
from app.api.schemas import SocialStyleRequest, SocialStyleEntryResponse, SocialStyleResponse
from app.retrieval import (
    InvalidRetrievalRequestError,
    UnsupportedRetrievalModeError,
    NoIndexedCorpusError,
    RetrievalExecutionError,
    RetrievedChunkValidationError,
)
from app.social.types import StyleCategory, StyleRetrievalRequest

logger = logging.getLogger(__name__)

router = APIRouter()

VALID_STYLE_CATEGORIES = {cat.value for cat in StyleCategory}


def _parse_style_categories(raw_categories: list[str]) -> list[StyleCategory]:
    """Parse and validate style category strings into StyleCategory enums.

    Raises ValueError if any category string is not a valid StyleCategory value.
    """
    parsed = []
    invalid = []
    for raw in raw_categories:
        if raw not in VALID_STYLE_CATEGORIES:
            invalid.append(raw)
        else:
            parsed.append(StyleCategory(raw))

    if invalid:
        raise ValueError(
            f"Invalid style categories: {invalid}. "
            f"Valid values: {sorted(VALID_STYLE_CATEGORIES)}"
        )

    return parsed


def _map_entry(entry) -> dict:
    """Map a StyleEntry dataclass to a response dict."""
    return {
        "content": entry.content,
        "source_label": entry.source_label,
        "score": entry.score,
        "metadata": entry.metadata,
    }


def _map_source_reference(ref) -> dict:
    """Map a SourceReference dataclass to a response dict."""
    return {
        "source_label": ref.source_label,
        "document_id": ref.document_id,
    }


def _map_warning(warning) -> dict:
    """Map a RetrievalWarning dataclass to a response dict."""
    result = {
        "code": warning.code.value if hasattr(warning.code, "value") else str(warning.code),
        "severity": warning.severity.value if hasattr(warning.severity, "value") else str(warning.severity),
        "source": warning.source,
        "message": warning.message,
    }
    if warning.details is not None:
        result["details"] = warning.details
    return result


@router.post("/social-style/retrieve", response_model=SocialStyleResponse)
def retrieve_social_style(request: Request, body: SocialStyleRequest):
    """Retrieve social style context grouped by category."""
    # Validate style categories
    try:
        style_categories = _parse_style_categories(body.style_categories)
    except ValueError as e:
        return JSONResponse(
            status_code=422,
            content={"detail": str(e)},
        )

    # Build domain request
    domain_request = StyleRetrievalRequest(
        tenant_id=body.tenant_id,
        query=body.query,
        style_categories=style_categories,
        platform=body.platform,
        per_category_limit=body.per_category_limit,
        collection=body.collection,
    )

    # Execute retrieval
    try:
        context = request.app.state.social_style_service.retrieve(domain_request)
    except (InvalidRetrievalRequestError, UnsupportedRetrievalModeError,
            NoIndexedCorpusError, RetrievalExecutionError, RetrievedChunkValidationError) as e:
        return map_retrieval_error_to_response(e)

    # Map response
    return SocialStyleResponse(
        voice_rules=[_map_entry(e) for e in context.voice_rules],
        hook_patterns=[_map_entry(e) for e in context.hook_patterns],
        cta_patterns=[_map_entry(e) for e in context.cta_patterns],
        past_post_patterns=[_map_entry(e) for e in context.past_post_patterns],
        avoid_rules=[_map_entry(e) for e in context.avoid_rules],
        offer_positioning=[_map_entry(e) for e in context.offer_positioning],
        source_references=[_map_source_reference(ref) for ref in context.source_references],
        warnings=[_map_warning(w) for w in context.warnings],
        trace_ids=context.trace_ids,
        missing_categories=[cat.value for cat in context.missing_categories],
    )
