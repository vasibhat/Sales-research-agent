"""A tiny vector database: chunks + a numpy matrix of their embeddings, saved as JSON.

Good enough for a few thousand chunks. For millions, swap this class for a
real vector database (pgvector, Qdrant...) and keep the same methods.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunker import Chunk


@dataclass
class Hit:
    source: str
    text: str
    score: float  # cosine similarity: 1.0 = same meaning, 0 = unrelated


class VectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.embedding_model = ""
        self.chunks: list[Chunk] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        store = cls(path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            store.embedding_model = data.get("embedding_model", "")
            rows = data["chunks"]
            store.replace(
                [Chunk(id=r["id"], source=r["source"], text=r["text"]) for r in rows],
                [r["vector"] for r in rows],
            )
        return store

    def replace(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.chunks = list(chunks)
        if self.chunks:
            self._matrix = _normalise(np.array(vectors, dtype=np.float32))
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def save(self, embedding_model: str) -> None:
        self.embedding_model = embedding_model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{**asdict(c), "vector": v.tolist()} for c, v in zip(self.chunks, self._matrix)]
        self.path.write_text(json.dumps({"embedding_model": embedding_model, "chunks": rows}), encoding="utf-8")

    def is_empty(self) -> bool:
        return not self.chunks

    def search(self, query_vector: list[float], top_k: int) -> list[Hit]:
        if self.is_empty():
            return []
        query = _normalise(np.array([query_vector], dtype=np.float32))[0]
        scores = self._matrix @ query  # vectors are normalised, so dot product = cosine similarity
        best = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i].source, self.chunks[i].text, float(scores[i])) for i in best]


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms
