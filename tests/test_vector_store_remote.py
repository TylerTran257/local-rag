"""VectorStoreService backend selection (local path vs remote URL) and dimensions."""
from unittest.mock import MagicMock

import pytest

import app.services.vector_store_service as vss
from app.services.vector_store_service import DEFAULT_VECTOR_SIZE, VectorStoreService


@pytest.fixture
def fake_client(monkeypatch):
    created = {}

    def _factory(*args, **kwargs):
        created["args"] = args
        created["kwargs"] = kwargs
        client = MagicMock()
        # Pretend the collection already exists so ensure_collection is a no-op.
        client.get_collection.return_value = MagicMock(points_count=0)
        created["client"] = client
        return client

    monkeypatch.setattr(vss, "QdrantClient", _factory)
    return created


class TestBackendSelection:
    def test_url_builds_remote_client(self, fake_client):
        VectorStoreService(qdrant_url="http://qdrant:6333", qdrant_api_key="secret")
        assert fake_client["kwargs"].get("url") == "http://qdrant:6333"
        assert fake_client["kwargs"].get("api_key") == "secret"
        assert "path" not in fake_client["kwargs"]

    def test_path_builds_local_client(self, fake_client, tmp_path):
        VectorStoreService(qdrant_path=str(tmp_path))
        assert fake_client["kwargs"].get("path") == str(tmp_path)
        assert "url" not in fake_client["kwargs"]


class TestCollectionDimensions:
    def test_ensure_collection_uses_given_dimension(self, fake_client, tmp_path):
        service = VectorStoreService(qdrant_path=str(tmp_path))
        client = fake_client["client"]
        # Force the create path by reporting the collection as missing.
        client.get_collection.side_effect = ValueError("missing")

        service.ensure_collection("custom_collection", vector_size=768)

        create_kwargs = client.create_collection.call_args.kwargs
        assert create_kwargs["collection_name"] == "custom_collection"
        assert create_kwargs["vectors_config"].size == 768

    def test_default_dimension_constant(self):
        assert DEFAULT_VECTOR_SIZE == 384
