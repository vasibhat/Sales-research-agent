"""HTTP API (production entry point).

POST /research  {"question": "Research Infosys for Monday's meeting"}
GET  /health
Open http://127.0.0.1:8080/docs to try it in the browser.
"""

from __future__ import annotations

import logging
import os
import secrets
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .agent import SalesResearchAgent
from .factory import build_agent

log = logging.getLogger("sales_agent.api")
app = FastAPI(title="Sales Research Agent", version=__version__)


@lru_cache(maxsize=1)
def get_agent() -> SalesResearchAgent:
    return build_agent()


def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("APP_API_KEY")
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["Research Infosys for Monday's meeting"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/research", dependencies=[Depends(check_api_key)])
def research(request: ResearchRequest, agent: SalesResearchAgent = Depends(get_agent)) -> dict[str, Any]:
    try:
        result = agent.run(request.question)
    except Exception:
        log.exception("Research failed")
        raise HTTPException(status_code=502, detail="The research failed. Check the server logs.")
    if result.refused:
        raise HTTPException(status_code=400, detail=result.refusal_reason)
    return {
        "briefing": result.briefing.model_dump(),
        "stopped_early": result.stopped_early,
        "tools_used": result.tools_used,
        "trace": result.trace,
    }
