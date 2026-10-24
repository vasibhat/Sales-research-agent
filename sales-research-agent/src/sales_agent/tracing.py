"""Observability: record every model call and tool call of one run.

Each run writes one JSONL file (one JSON object per line) in logs/traces/.
Open it to see exactly what the agent did, how long each step took and what it cost.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PRICES_PER_MILLION
from .llm import LLM, LLMReply


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES_PER_MILLION.get(model)
    if price is None:
        return 0.0
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


class Tracer:
    def __init__(self, traces_dir: Path | None) -> None:
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.events: list[dict[str, Any]] = []
        self._started = time.perf_counter()
        self.path: Path | None = None
        if traces_dir is not None:
            traces_dir.mkdir(parents=True, exist_ok=True)
            self.path = traces_dir / f"{self.run_id}.jsonl"

    def log(self, kind: str, **data: Any) -> None:
        event = {"time": datetime.now(timezone.utc).isoformat(), "kind": kind, **data}
        self.events.append(event)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def summary(self) -> dict[str, Any]:
        llm_calls = [e for e in self.events if e["kind"] == "llm_call"]
        tool_calls = [e for e in self.events if e["kind"] == "tool_call"]
        return {
            "run_id": self.run_id,
            "llm_calls": len(llm_calls),
            "tool_calls": len(tool_calls),
            "tool_errors": sum(1 for e in tool_calls if not e["ok"]),
            "input_tokens": sum(e["input_tokens"] for e in llm_calls),
            "output_tokens": sum(e["output_tokens"] for e in llm_calls),
            "cost_usd": round(sum(e["cost_usd"] for e in llm_calls), 6),
            "seconds": round(time.perf_counter() - self._started, 2),
            "trace_file": str(self.path) if self.path else None,
        }


class TracedLLM:
    """Wraps any LLM and logs each call. The agent code does not change at all."""

    def __init__(self, inner: LLM, tracer: Tracer) -> None:
        self.inner = inner
        self.tracer = tracer

    def chat(self, messages, tools=None, json_mode=False) -> LLMReply:
        started = time.perf_counter()
        reply = self.inner.chat(messages, tools=tools, json_mode=json_mode)
        self.tracer.log(
            "llm_call",
            model=reply.model,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            cost_usd=round(estimate_cost(reply.model, reply.input_tokens, reply.output_tokens), 6),
            ms=round((time.perf_counter() - started) * 1000),
            tool_calls=[c.name for c in reply.tool_calls],
        )
        return reply

    def embed(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        vectors = self.inner.embed(texts)
        self.tracer.log("embed", count=len(texts), ms=round((time.perf_counter() - started) * 1000))
        return vectors
