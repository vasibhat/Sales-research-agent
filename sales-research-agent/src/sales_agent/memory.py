"""Long-term memory, saved in a JSON file: companies researched before and the user's preferences.

State (state.py) is forgotten after each run. Memory is kept between runs.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from typing import Any

from .briefing import SalesBriefing


class Memory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()  # the API may run two requests at the same time
        self._data = self._read()

    def _read(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data.setdefault("companies", {})
        data.setdefault("preferences", [])
        return data

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.path)  # swap in one step, so a crash never leaves half a file

    @staticmethod
    def _key(company: str) -> str:
        return " ".join(company.lower().split())

    def recall_company(self, company: str) -> dict[str, Any] | None:
        return self._data["companies"].get(self._key(company))

    def remember_company(self, company: str, briefing: SalesBriefing) -> None:
        with self._lock:
            self._data["companies"][self._key(company)] = {
                "name": briefing.company,
                "last_researched": date.today().isoformat(),
                "summary": briefing.summary,
                "lead_score": briefing.lead_score,
                "lead_grade": briefing.lead_grade,
            }
            self._write()

    def preferences(self) -> list[str]:
        return list(self._data["preferences"])

    def add_preference(self, text: str) -> None:
        with self._lock:
            self._data["preferences"].append(text.strip())
            self._write()

    def clear_preferences(self) -> None:
        with self._lock:
            self._data["preferences"] = []
            self._write()

    def context_note(self, company: str) -> str:
        """A short note for the system prompt, so the agent builds on past work."""
        parts = []
        past = self.recall_company(company)
        if past:
            parts.append(
                f"Memory: we researched {past['name']} on {past['last_researched']}. "
                f"Summary then: {past['summary']} Lead score then: {past['lead_score']}. "
                "Focus on what has changed since."
            )
        if self._data["preferences"]:
            parts.append("User preferences: " + "; ".join(self._data["preferences"]))
        return "\n".join(parts)
