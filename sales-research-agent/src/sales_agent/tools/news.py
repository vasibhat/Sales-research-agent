"""get_news: latest headlines from Google News RSS (free, no API key)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, Field

RSS_URL = "https://news.google.com/rss/search"


class GetNewsArgs(BaseModel):
    company_name: str = Field(description="Company name to find news about")
    max_items: int = Field(default=5, ge=1, le=10, description="How many headlines to return")


def get_news(company_name: str, max_items: int, client: httpx.Client) -> dict:
    response = client.get(RSS_URL, params={"q": f'"{company_name}"', "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    response.raise_for_status()
    root = ET.fromstring(response.text)

    articles = []
    for item in root.iter("item"):
        articles.append({
            "title": item.findtext("title"),
            "published": item.findtext("pubDate"),
            "source": item.findtext("source"),
            "link": item.findtext("link"),
        })
        if len(articles) >= max_items:
            break
    if not articles:
        return {"company": company_name, "articles": [], "message": "No recent news found."}
    return {"company": company_name, "articles": articles}
