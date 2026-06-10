"""Type definitions for social style retrieval."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.retrieval.types import RetrievalWarning


class StyleCategory(Enum):
    """Social style memory categories."""
    VOICE_RULES = "voice_rules"
    HOOK_PATTERNS = "hook_patterns"
    CTA_PATTERNS = "cta_patterns"
    PAST_POST_PATTERNS = "past_post_patterns"
    AVOID_RULES = "avoid_rules"
    OFFER_POSITIONING = "offer_positioning"


@dataclass
class StyleEntry:
    """A single style memory entry."""
    content: str
    source_label: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceReference:
    """Reference to a source document."""
    source_label: str
    document_id: str


@dataclass
class StyleRetrievalRequest:
    """Request for social style retrieval."""
    tenant_id: str
    query: str
    style_categories: list[StyleCategory]
    platform: str | None = None
    per_category_limit: int = 5
    collection: str = "style_memory"


@dataclass
class StyleContext:
    """Social style context response grouped by category."""
    voice_rules: list[StyleEntry] = field(default_factory=list)
    hook_patterns: list[StyleEntry] = field(default_factory=list)
    cta_patterns: list[StyleEntry] = field(default_factory=list)
    past_post_patterns: list[StyleEntry] = field(default_factory=list)
    avoid_rules: list[StyleEntry] = field(default_factory=list)
    offer_positioning: list[StyleEntry] = field(default_factory=list)
    source_references: list[SourceReference] = field(default_factory=list)
    warnings: list[RetrievalWarning] = field(default_factory=list)
    trace_ids: list[str] = field(default_factory=list)
    missing_categories: list[StyleCategory] = field(default_factory=list)
