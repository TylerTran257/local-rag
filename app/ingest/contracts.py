from dataclasses import dataclass, field
from typing import Any


REQUIRED_METADATA_FIELDS = (
    "service_name",
    "tenant_id",
    "collection",
    "source_type",
    "source_label",
)


@dataclass
class ValidatedMetadata:
    service_name: str
    tenant_id: str
    collection: str
    source_type: str
    source_label: str
    domain_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestDocument:
    text: str
    service_name: str
    tenant_id: str
    collection: str
    source_type: str
    source_label: str
    domain_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestChunk:
    chunk_id: str
    text: str
    service_name: str
    tenant_id: str
    collection: str
    source_type: str
    source_label: str
    domain_metadata: dict[str, Any] = field(default_factory=dict)


IngestBatchItem = IngestDocument | IngestChunk
IngestBatch = list[IngestBatchItem]


class MetadataValidationError(ValueError):
    def __init__(self, invalid_fields: list[str], metadata: dict[str, Any]):
        self.invalid_fields = invalid_fields
        self.metadata = metadata
        super().__init__(f"Invalid metadata fields: {', '.join(invalid_fields)}")


class MetadataValidator:
    @staticmethod
    def validate(metadata: dict[str, Any]) -> ValidatedMetadata:
        invalid_fields = [
            field_name
            for field_name in REQUIRED_METADATA_FIELDS
            if field_name not in metadata or not MetadataValidator._is_non_empty_string(metadata[field_name])
        ]
        if invalid_fields:
            raise MetadataValidationError(invalid_fields=invalid_fields, metadata=metadata)

        domain_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in REQUIRED_METADATA_FIELDS
        }
        return ValidatedMetadata(
            service_name=metadata["service_name"].strip(),
            tenant_id=metadata["tenant_id"].strip(),
            collection=metadata["collection"].strip(),
            source_type=metadata["source_type"].strip(),
            source_label=metadata["source_label"].strip(),
            domain_metadata=domain_metadata,
        )

    @staticmethod
    def _is_non_empty_string(value: Any) -> bool:
        return isinstance(value, str) and value.strip() != ""
