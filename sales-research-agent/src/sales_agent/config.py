"""All settings in one place.

Values come from environment variables (or a .env file). Change behaviour
here, not deep inside the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Approximate USD price per 1 million tokens, used only for cost reports.
# Prices change - check OpenAI's pricing page and update these numbers.
PRICES_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


@dataclass(frozen=True)
class Settings:
    model: str
    embedding_model: str
    seller_name: str
    max_steps: int
    max_input_chars: int
    max_tool_output_chars: int
    data_dir: Path
    traces_dir: Path | None

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.json"

    @property
    def memory_path(self) -> Path:
        return self.data_dir / "memory.json"

    @property
    def icp_path(self) -> Path:
        return self.data_dir / "icp.json"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            seller_name=os.getenv("SELLER_NAME", "LedgerLeaf Technologies"),
            max_steps=int(os.getenv("MAX_STEPS", "8")),
            max_input_chars=int(os.getenv("MAX_INPUT_CHARS", "1000")),
            max_tool_output_chars=int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "4000")),
            data_dir=Path(os.getenv("DATA_DIR", "data")).resolve(),
            traces_dir=Path(os.getenv("TRACES_DIR", "logs/traces")).resolve(),
        )
