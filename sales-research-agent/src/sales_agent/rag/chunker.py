"""Split documents into small chunks that can be searched one by one."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    source: str
    text: str


def chunk_text(text: str, source: str, max_chars: int = 800) -> list[Chunk]:
    """Split a document into chunks made of whole paragraphs.

    - Each chunk starts with its nearest heading, so it still makes sense alone.
    - Each new chunk repeats the last paragraph of the one before (overlap),
      so an idea that crosses the boundary is not lost.
    """
    pieces: list[str] = []
    heading = ""
    current: list[str] = []
    has_new = False  # stops a chunk that holds only the overlap paragraph

    def flush() -> None:
        if has_new:
            body = "\n\n".join(current)
            pieces.append(f"{heading}\n\n{body}" if heading else body)

    for para in _paragraphs(text, max_chars):
        if para.startswith("#"):
            flush()
            current, has_new = [], False
            first, _, rest = para.partition("\n")
            heading = first.lstrip("#").strip()
            if not rest.strip():
                continue
            para = rest.strip()
        if current and len("\n\n".join([*current, para])) > max_chars:
            flush()
            overlap = current[-1]
            current = [overlap] if len(overlap) <= max_chars // 2 else []
            has_new = False
        current.append(para)
        has_new = True
    flush()

    return [Chunk(id=f"{source}#{i}", source=source, text=piece) for i, piece in enumerate(pieces)]


def _paragraphs(text: str, max_chars: int) -> list[str]:
    result: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        # A very long paragraph is cut into windows with a small overlap.
        while len(para) > max_chars:
            result.append(para[:max_chars])
            para = para[max_chars - 100 :]
        result.append(para)
    return result
