from sentence_transformers import SentenceTransformer

from app.settings import settings


class EmbeddingService:
    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.embedding_model_name)

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
