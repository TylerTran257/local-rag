from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.retrieval.types import RetrievalMode


NonEmptyString = Annotated[str, Field(min_length=1)]

# Scope-enforcement keys that request filters must not override
RESERVED_FILTER_KEYS = frozenset({"service_name", "tenant_id", "collection", "collections"})


class IngestDocumentRequest(BaseModel):
    text: str = Field(min_length=1)
    service_name: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    domain_metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    chunk_count: int


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    chunk_count: int
    source_label: NonEmptyString


class DocumentIngestRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: NonEmptyString
    service_name: NonEmptyString
    tenant_id: NonEmptyString
    collection: NonEmptyString
    source_type: NonEmptyString
    source_label: NonEmptyString
    domain_metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_count: int


class ChunkResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: NonEmptyString
    score: float
    source_label: NonEmptyString
    collection: NonEmptyString
    service_name: NonEmptyString
    tenant_id: NonEmptyString
    chunk_id: NonEmptyString
    domain_metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: NonEmptyString
    service_name: NonEmptyString
    tenant_id: NonEmptyString
    collections: list[NonEmptyString] = Field(min_length=1)
    filters: dict[str, str] = Field(default_factory=dict)
    limit: int = Field(default=5, gt=0)
    mode: RetrievalMode = RetrievalMode.HYBRID

    @field_validator("filters")
    @classmethod
    def filters_must_not_override_scope(cls, value: dict[str, str]) -> dict[str, str]:
        reserved = RESERVED_FILTER_KEYS.intersection(value)
        if reserved:
            raise ValueError(
                f"Filter keys {sorted(reserved)} are reserved for scope enforcement "
                "and cannot be set through filters"
            )
        return value


class RetrieveResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: list[ChunkResult]
    trace_id: NonEmptyString


class AnswerRequest(RetrieveRequest):
    model_config = ConfigDict(frozen=True)


class AnswerResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    sources: list[ChunkResult]
    trace_id: NonEmptyString


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event: NonEmptyString
    data: str
    done: bool


class DeleteCollectionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_name: NonEmptyString
    tenant_id: NonEmptyString
    collections: list[NonEmptyString] = Field(min_length=1)
    filters: dict[str, str] = Field(default_factory=dict)

    @field_validator("filters")
    @classmethod
    def filters_must_not_override_scope(cls, value: dict[str, str]) -> dict[str, str]:
        reserved = RESERVED_FILTER_KEYS.intersection(value)
        if reserved:
            raise ValueError(
                f"Filter keys {sorted(reserved)} are reserved for scope enforcement "
                "and cannot be set through filters"
            )
        return value


class DeleteCollectionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    deleted_count: int


class ProfileUpsertRequest(BaseModel):
    """Create or update a service config profile.

    Omitted fields fall back to the platform defaults. ``embedding_model`` is
    fixed once a profile exists (see the immutability rule in ProfileStore).
    """

    model_config = ConfigDict(frozen=True)

    service_name: NonEmptyString
    embedding_model: NonEmptyString | None = None
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)
    dense_limit: int | None = Field(default=None, gt=0)
    lexical_limit: int | None = Field(default=None, gt=0)
    fusion_rrf_k: int | None = Field(default=None, gt=0)
    default_mode: RetrievalMode | None = None
    generation_overrides: dict[str, Any] = Field(default_factory=dict)


class ProfileResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_name: NonEmptyString
    embedding_model: NonEmptyString
    chunk_size: int
    chunk_overlap: int
    dense_limit: int
    lexical_limit: int
    fusion_rrf_k: int
    default_mode: RetrievalMode
    generation_overrides: dict[str, Any] = Field(default_factory=dict)
