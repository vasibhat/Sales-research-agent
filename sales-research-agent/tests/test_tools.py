import json
from pathlib import Path

import httpx
import pytest

from sales_agent.guardrails import GuardrailError
from sales_agent.tools import Tool, ToolRegistry
from sales_agent.tools.company import SearchCompanyArgs, search_company
from sales_agent.tools.news import get_news
from sales_agent.tools.qualify import load_icp, score_lead
from sales_agent.tools.website import fetch_website

ICP = load_icp(Path(__file__).resolve().parents[1] / "data" / "icp.json")


def mock_client(handler, follow_redirects=True):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=follow_redirects)


def test_score_lead_hot():
    result = score_lead("Retail", 5000, "India", ["raised new funding", "opening a new warehouse"], icp=ICP)
    assert result["score"] == 35 + 25 + 15 + 10
    assert result["grade"] == "Hot"


def test_score_lead_cold():
    result = score_lead("Gaming", 20, "Brazil", [], icp=ICP)
    assert result["score"] == 0 and result["grade"] == "Cold"


def test_registry_rejects_bad_and_unknown_calls():
    registry = ToolRegistry([Tool("search_company", "x", SearchCompanyArgs, lambda company_name: {"ok": 1})])
    bad = registry.run("search_company", '{"wrong": 1}')
    assert not bad.ok and "Bad arguments" in bad.content
    unknown = registry.run("delete_database", "{}")
    assert not unknown.ok and "Unknown tool" in unknown.content
    good = registry.run("search_company", '{"company_name": "Infosys"}')
    assert good.ok


def test_search_company_uses_wikipedia():
    def handler(request: httpx.Request):
        if request.url.path == "/w/api.php":
            return httpx.Response(200, json={"query": {"search": [{"title": "Infosys"}]}})
        assert request.url.path == "/api/rest_v1/page/summary/Infosys"
        return httpx.Response(200, json={
            "title": "Infosys", "description": "Indian IT company", "extract": "Infosys is an IT company.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Infosys"}},
        })

    result = search_company("Infosys", client=mock_client(handler))
    assert result["found"] and result["description"] == "Indian IT company"


def test_search_company_not_found():
    result = search_company("Zzqx", client=mock_client(lambda r: httpx.Response(200, json={"query": {"search": []}})))
    assert result["found"] is False


def test_get_news_parses_rss():
    rss = """<rss><channel>
      <item><title>Acme opens plant</title><pubDate>Mon, 06 Oct 2026</pubDate><source>Paper</source><link>https://n/1</link></item>
      <item><title>Acme hires CFO</title><pubDate>Tue, 07 Oct 2026</pubDate><source>Paper</source><link>https://n/2</link></item>
    </channel></rss>"""
    result = get_news("Acme", 1, client=mock_client(lambda r: httpx.Response(200, text=rss)))
    assert [a["title"] for a in result["articles"]] == ["Acme opens plant"]


def test_fetch_website_extracts_visible_text():
    html = "<html><head><title>Acme</title><script>var x=1</script></head><body><h1>Snacks</h1><p>We make chips.</p></body></html>"
    client = mock_client(lambda r: httpx.Response(200, html=html), follow_redirects=False)
    result = fetch_website("https://acme.example", client=client, url_checker=lambda url: None)
    assert result["title"] == "Acme"
    assert result["text"] == "Snacks We make chips."


def test_fetch_website_checks_every_redirect():
    def checker(url):
        if "127.0.0.1" in url:
            raise GuardrailError("private address")

    client = mock_client(lambda r: httpx.Response(302, headers={"location": "http://127.0.0.1/admin"}), follow_redirects=False)
    with pytest.raises(GuardrailError):
        fetch_website("https://acme.example", client=client, url_checker=checker)


def test_tool_output_is_json():
    registry = ToolRegistry([Tool("search_company", "x", SearchCompanyArgs, lambda company_name: {"name": "₹ test"})])
    assert json.loads(registry.run("search_company", '{"company_name": "a"}').content) == {"name": "₹ test"}
