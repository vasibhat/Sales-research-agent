"""Build the search index: read docs -> chunk -> embed -> save."""

from __future__ import annotations

from pathlib import Path

from ..llm import LLM
from .chunker import Chunk, chunk_text
from .store import VectorStore

SUPPORTED = {".md", ".txt"}


def ingest(docs_dir: Path, store: VectorStore, llm: LLM, embedding_model: str, batch_size: int = 64) -> tuple[int, int]:
    """Returns (number of files, number of chunks)."""
    files = sorted(p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_text(path.read_text(encoding="utf-8"), source=str(path.relative_to(docs_dir))))

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):  # one API call per batch, not per chunk
        vectors.extend(llm.embed([c.text for c in chunks[start : start + batch_size]]))

    store.replace(chunks, vectors)
    store.save(embedding_model)
    return len(files), len(chunks)
