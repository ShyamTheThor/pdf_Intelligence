import json
import numpy as np
from pathlib import Path

from src.config import INDEX_PATH, METADATA_PATH
from src.embeddings.embedder import Embedder


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] = []
        self._load()

    def add_documents(self, chunks: list[dict]):
        texts = [c["text"] for c in chunks]
        emb = np.array(self.embedder.embed(texts), dtype=np.float32)

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        for c in chunks:
            self.documents.append({"text": c["text"], "source": c["source"]})

        self._save()

    def search(self, query_embedding: list[float], k: int = 4) -> list[dict]:
        if self.embeddings is None or len(self.documents) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q)
        sim = (self.embeddings @ q.T).flatten() / (norms.flatten() * q_norm + 1e-10)
        top_idx = np.argsort(sim)[-k:][::-1]

        return [
            {
                "text": self.documents[i]["text"],
                "source": self.documents[i]["source"],
                "score": float(sim[i]),
            }
            for i in top_idx
        ]

    def count(self) -> int:
        return len(self.documents)

    def delete_all(self):
        self.embeddings = None
        self.documents = []
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def _save(self):
        if self.embeddings is not None:
            np.savez_compressed(INDEX_PATH, embeddings=self.embeddings)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def _load(self):
        if INDEX_PATH.exists() and METADATA_PATH.exists():
            data = np.load(INDEX_PATH)
            self.embeddings = data["embeddings"]
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
