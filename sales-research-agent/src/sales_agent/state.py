"""State: everything the agent knows during ONE research run."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Finding(BaseModel):
    tool: str
    arguments: str
    result: str
    ok: bool


class ResearchState(BaseModel):
    question: str
    company: str = ""
    plan: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    steps: int = 0
    stopped_early: bool = False

    def tools_used(self) -> list[str]:
        return sorted({f.tool for f in self.findings})

    def evidence_text(self, per_item_chars: int = 1500, total_chars: int = 12000) -> str:
        """All tool results as numbered notes, for the briefing writer."""
        lines = []
        for i, f in enumerate(self.findings, 1):
            status = "" if f.ok else " (FAILED)"
            lines.append(f"[{i}] {f.tool}({f.arguments}){status}:\n{f.result[:per_item_chars]}")
        return "\n\n".join(lines)[:total_chars] or "(no tool results)"
