from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.settings import settings


class VectorStoreService:
    def __init__(self) -> None:
        self.collection_name = settings.qdrant_collection_name
        self.client = QdrantClient(path=settings.qdrant_path)
        self.ensure_collection()

    def ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection_name)
            return
        except NotImplementedError:
            pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def upsert_document_chunks(
        self,
        document_id: str,
        original_filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        points = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=f"{uuid4()}",
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "original_filename": original_filename,
                        "chunk_index": index,
                        "text": chunk,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: list[float], limit: int) -> list[dict]:
        hits = self.client.query_points(
            collection_name=self.collection_name, query=query_embedding, limit=limit
        ).points

        payloads = []
        for hit in hits:
            payload = hit.payload
            if payload is None:
                continue
            payloads.append(
                {
                    "document_id": payload["document_id"],
                    "original_filename": payload["original_filename"],
                    "chunk_index": payload["chunk_index"],
                    "score": hit.score,
                    "text": payload["text"],
                }
            )

        return [
            {
                "document_id": payload["document_id"],
                "original_filename": payload["original_filename"],
                "chunk_index": payload["chunk_index"],
                "score": payload["score"],
                "text": payload["text"],
            }
            for payload in payloads
        ]

    def has_indexed_chunks(self) -> bool:
        try:
            collection_info = self.client.get_collection(self.collection_name)
        except UnexpectedResponse:
            return False

        points_count = collection_info.points_count or 0
        return points_count > 0
