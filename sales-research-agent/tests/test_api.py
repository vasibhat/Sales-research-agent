import json

import pytest
from fakes import ScriptedLLM, text
from fastapi.testclient import TestClient
from test_agent import BRIEFING, PLAN

from sales_agent.api import app, get_agent
from sales_agent.factory import build_agent


@pytest.fixture
def client_with(settings):
    def make(replies):
        agent = build_agent(settings, llm=ScriptedLLM(replies))
        app.dependency_overrides[get_agent] = lambda: agent
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_health():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_research_returns_briefing(client_with):
    client = client_with([text(PLAN), text("notes"), text(BRIEFING)])
    response = client.post("/research", json={"question": "Research Acme Foods"})
    assert response.status_code == 200
    assert response.json()["briefing"]["company"] == "Acme Foods"


def test_off_topic_gives_400(client_with):
    plan = json.dumps({"on_topic": False, "reason": "not a company", "company": "", "steps": []})
    response = client_with([text(plan)]).post("/research", json={"question": "Tell me a joke"})
    assert response.status_code == 400


def test_api_key_required_when_set(client_with, monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "secret")
    client = client_with([text(PLAN), text("notes"), text(BRIEFING)])
    assert client.post("/research", json={"question": "Research Acme"}).status_code == 401
    ok = client.post("/research", json={"question": "Research Acme"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
