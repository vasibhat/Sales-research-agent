"""search_internal_docs: RAG exposed as a tool.

The agent decides WHEN to search our own documents (case studies, pricing,
objection handling) and WHAT to search for. That is "RAG as a tool".
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import LLM
from ..rag.store import VectorStore


class SearchDocsArgs(BaseModel):
    query: str = Field(description="What to look for, e.g. 'retail case study invoice automation'")
    top_k: int = Field(default=4, ge=1, le=8, description="How many passages to return")


def search_internal_docs(query: str, top_k: int, store: VectorStore, llm: LLM) -> dict:
    if store.is_empty():
        return {"error": "The internal document index is empty. Run: sales-agent ingest"}
    hits = store.search(llm.embed([query])[0], top_k)
    return {"results": [{"source": h.source, "score": round(h.score, 3), "text": h.text} for h in hits]}
