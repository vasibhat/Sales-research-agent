"""search_company: public company profile from Wikipedia (free, no API key)."""

from __future__ import annotations

from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


class SearchCompanyArgs(BaseModel):
    company_name: str = Field(description="Company name, for example 'Infosys'")


def search_company(company_name: str, client: httpx.Client) -> dict:
    response = client.get(
        SEARCH_URL,
        params={"action": "query", "list": "search", "srsearch": f"{company_name} company", "format": "json", "srlimit": 1},
    )
    response.raise_for_status()
    hits = response.json().get("query", {}).get("search", [])
    if not hits:
        return {"found": False, "message": f"No public profile for '{company_name}'. Try fetch_website or get_news."}

    title = hits[0]["title"]
    summary = client.get(SUMMARY_URL.format(title=quote(title.replace(" ", "_"), safe="")))
    summary.raise_for_status()
    data = summary.json()
    return {
        "found": True,
        "name": data.get("title"),
        "description": data.get("description"),
        "summary": data.get("extract"),
        "source_url": data.get("content_urls", {}).get("desktop", {}).get("page"),
        "note": "Check the name matches the company you want; search can return a similar name.",
    }
