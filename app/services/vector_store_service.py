from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.db.models import DocumentChunk
from app.ingest import MetadataValidator, ValidatedMetadata
from app.retrieval.types import RetrievalScope
from app.settings import settings


STORED_CHUNK_FIELDS = {
    "document_id",
    "original_filename",
    "chunk_index",
    "text",
}


# Default vector dimension (sentence-transformers/all-MiniLM-L6-v2).
DEFAULT_VECTOR_SIZE = 384


class VectorStoreService:
    def __init__(
        self,
        qdrant_path: str | Path | None = None,
        collection_name: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.qdrant_collection_name
        # Prefer a remote Qdrant when a URL is configured (shared deployment);
        # otherwise fall back to the on-disk local store.
        resolved_url = qdrant_url if qdrant_url is not None else settings.qdrant_url
        if resolved_url:
            api_key = qdrant_api_key if qdrant_api_key is not None else settings.qdrant_api_key
            self.client = QdrantClient(url=resolved_url, api_key=api_key)
        else:
            qdrant_storage_path = qdrant_path or settings.qdrant_path
            self.client = QdrantClient(path=str(qdrant_storage_path))
        self.ensure_collection()

    def _resolve_collection(self, collection_name: str | None) -> str:
        return collection_name or self.collection_name

    def ensure_collection(
        self,
        collection_name: str | None = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        name = self._resolve_collection(collection_name)
        try:
            self.client.get_collection(name)
            return
        except (NotImplementedError, UnexpectedResponse, ValueError):
            pass

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_document_chunks(
        self,
        document_id: str,
        original_filename: str,
        chunks: Sequence[str | DocumentChunk],
        embeddings: list[list[float]],
        metadata: Sequence[ValidatedMetadata | dict[str, Any]] | ValidatedMetadata | dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> None:
        metadata_by_chunk: Sequence[ValidatedMetadata | dict[str, Any]] | None
        if metadata is None:
            metadata_by_chunk = None
        elif isinstance(metadata, ValidatedMetadata) or isinstance(metadata, dict):
            metadata_by_chunk = [metadata] * len(chunks)
        else:
            metadata_by_chunk = metadata

        if metadata_by_chunk is not None and len(metadata_by_chunk) != len(chunks):
            raise ValueError("metadata length must match chunks length")

        points = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_text = chunk if isinstance(chunk, str) else chunk.text
            payload = {
                "document_id": document_id,
                "original_filename": original_filename,
                "chunk_index": index,
                "text": chunk_text,
            }
            if metadata_by_chunk is not None:
                payload.update(self._build_metadata_payload(metadata_by_chunk[index]))

            points.append(
                PointStruct(
                    id=f"{uuid4()}",
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self._resolve_collection(collection_name), points=points
        )

    def build_query_filter(self, scope: RetrievalScope) -> Filter:
        must_conditions: list[Condition] = []
        must_conditions.append(
            FieldCondition(
                key="service_name",
                match=MatchValue(value=scope.service_name),
            )
        )
        must_conditions.append(
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=scope.tenant_id),
            )
        )
        must_conditions.append(
            FieldCondition(
                key="collection",
                match=MatchAny(any=scope.collections),
            )
        )
        must_conditions.extend(
            self._translate_filter_condition(field_name, value)
            for field_name, value in scope.filters.items()
        )
        return Filter(must=must_conditions)

    def delete_by_scope(
        self, scope: RetrievalScope, collection_name: str | None = None
    ) -> int:
        query_filter = self.build_query_filter(scope)
        name = self._resolve_collection(collection_name)
        count = self.client.count(name, count_filter=query_filter, exact=True).count
        if count > 0:
            self.client.delete(collection_name=name, points_selector=query_filter)
        return count

    def search(
        self,
        query_embedding: list[float],
        limit: int,
        query_filter: Filter | None = None,
        collection_name: str | None = None,
    ) -> list[dict]:
        hits = self.client.query_points(
            collection_name=self._resolve_collection(collection_name),
            query=query_embedding,
            limit=limit,
            query_filter=query_filter,
        ).points

        payloads = []
        for hit in hits:
            payload = hit.payload
            if payload is None:
                continue
            payloads.append({**payload, "score": hit.score})

        return payloads

    def has_indexed_chunks(self, collection_name: str | None = None) -> bool:
        try:
            collection_info = self.client.get_collection(
                self._resolve_collection(collection_name)
            )
        except (UnexpectedResponse, ValueError):
            return False

        points_count = collection_info.points_count or 0
        return points_count > 0

    def _build_metadata_payload(
        self, metadata: ValidatedMetadata | dict[str, Any]
    ) -> dict[str, Any]:
        validated_metadata = self._coerce_validated_metadata(metadata)
        domain_metadata = {
            key: value
            for key, value in validated_metadata.domain_metadata.items()
            if key not in STORED_CHUNK_FIELDS
        }
        return {
            "service_name": validated_metadata.service_name,
            "tenant_id": validated_metadata.tenant_id,
            "collection": validated_metadata.collection,
            "source_type": validated_metadata.source_type,
            "source_label": validated_metadata.source_label,
            **domain_metadata,
        }

    def _coerce_validated_metadata(
        self, metadata: ValidatedMetadata | dict[str, Any]
    ) -> ValidatedMetadata:
        if isinstance(metadata, ValidatedMetadata):
            return metadata
        return MetadataValidator.validate(metadata)

    def _translate_filter_condition(self, field_name: str, value: Any) -> Condition:
        if isinstance(value, bool):
            return FieldCondition(key=field_name, match=MatchValue(value=value))
        if isinstance(value, int):
            return FieldCondition(key=field_name, match=MatchValue(value=value))
        if isinstance(value, float):
            return FieldCondition(key=field_name, range=Range(gte=value, lte=value))
        if isinstance(value, str):
            return FieldCondition(key=field_name, match=MatchValue(value=value))
        if isinstance(value, list):
            if not value:
                raise ValueError(f"Unsupported filter value for field '{field_name}': {value!r}")

            bool_values = [item for item in value if isinstance(item, bool)]
            if len(bool_values) == len(value):
                return Filter(
                    should=[
                        FieldCondition(key=field_name, match=MatchValue(value=item))
                        for item in bool_values
                    ]
                )

            string_values = [item for item in value if isinstance(item, str)]
            if len(string_values) == len(value):
                return FieldCondition(key=field_name, match=MatchAny(any=string_values))

            integer_values = [
                item for item in value if isinstance(item, int) and not isinstance(item, bool)
            ]
            if len(integer_values) == len(value):
                return FieldCondition(key=field_name, match=MatchAny(any=integer_values))

            float_values = [item for item in value if isinstance(item, float)]
            if len(float_values) == len(value):
                return Filter(
                    should=[
                        FieldCondition(key=field_name, range=Range(gte=item, lte=item))
                        for item in float_values
                    ]
                )

        raise ValueError(f"Unsupported filter value for field '{field_name}': {value!r}")
