from src.config import TOP_K
from src.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        query_embedding = self.store.embedder.embed([query])[0]
        return self.store.search(query_embedding, k=k)
