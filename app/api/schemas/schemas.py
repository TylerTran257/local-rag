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
