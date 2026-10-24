from pathlib import Path

import pytest

from sales_agent.config import Settings

ICP_FILE = Path(__file__).resolve().parents[1] / "data" / "icp.json"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    (tmp_path / "icp.json").write_text(ICP_FILE.read_text())
    return Settings(
        model="fake",
        embedding_model="fake-embed",
        seller_name="TestSeller",
        max_steps=4,
        max_input_chars=1000,
        max_tool_output_chars=2000,
        data_dir=tmp_path,
        traces_dir=tmp_path / "traces",
    )
