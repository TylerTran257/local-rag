from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=3, gt=0, le=10)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=3, gt=0, le=10)


class IngestDocumentRequest(BaseModel):
    text: str = Field(min_length=1)
    service_name: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    domain_metadata: dict[str, Any] = Field(default_factory=dict)


class IngestChunkItem(BaseModel):
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    service_name: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    domain_metadata: dict[str, Any] = Field(default_factory=dict)


class IngestChunksRequest(BaseModel):
    chunks: list[IngestChunkItem] = Field(min_length=1)


class IngestResponse(BaseModel):
    chunk_count: int


class SocialStyleRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    style_categories: list[str]
    platform: str | None = None
    per_category_limit: int = Field(default=5, gt=0, le=50)
    collection: str = "style_memory"


class SocialStyleEntryResponse(BaseModel):
    content: str
    source_label: str
    score: float
    metadata: dict[str, Any]


class SocialStyleResponse(BaseModel):
    voice_rules: list[SocialStyleEntryResponse]
    hook_patterns: list[SocialStyleEntryResponse]
    cta_patterns: list[SocialStyleEntryResponse]
    past_post_patterns: list[SocialStyleEntryResponse]
    avoid_rules: list[SocialStyleEntryResponse]
    offer_positioning: list[SocialStyleEntryResponse]
    source_references: list[dict[str, str]]
    warnings: list[dict[str, Any]]
    trace_ids: list[str]
    missing_categories: list[str]
