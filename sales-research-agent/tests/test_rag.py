from pathlib import Path

from fakes import ScriptedLLM

from sales_agent.rag.chunker import chunk_text
from sales_agent.rag.ingest import ingest
from sales_agent.rag.store import VectorStore
from sales_agent.tools.internal_docs import search_internal_docs

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


def test_chunks_keep_heading_and_overlap():
    para = "word " * 50  # 250 chars
    doc = f"# Title\n\n## Section A\n\n{para}\n\nsecond {para}\n\nthird {para}\n\n## Section B\n\nshort"
    chunks = chunk_text(doc, "doc.md", max_chars=600)

    assert all(len(c.text) <= 600 + 50 for c in chunks)
    assert chunks[0].text.startswith("Section A")
    assert chunks[1].text.startswith("Section A")
    assert "second" in chunks[0].text and "second" in chunks[1].text  # overlap
    assert chunks[-1].text == "Section B\n\nshort"
    assert [c.id for c in chunks] == [f"doc.md#{i}" for i in range(len(chunks))]


def test_ingest_and_search(tmp_path):
    store = VectorStore(tmp_path / "index.json")
    files, chunks = ingest(DOCS, store, ScriptedLLM(), "fake-embed")
    assert files == 6 and chunks > 6

    loaded = VectorStore.load(tmp_path / "index.json")  # survives save and load
    result = search_internal_docs("freight vendor invoices warehouses logistics", 2, store=loaded, llm=ScriptedLLM())
    assert result["results"][0]["source"] == "case_study_logistics.md"


def test_search_on_empty_index_explains_what_to_do(tmp_path):
    result = search_internal_docs("anything", 3, store=VectorStore(tmp_path / "none.json"), llm=ScriptedLLM())
    assert "sales-agent ingest" in result["error"]
