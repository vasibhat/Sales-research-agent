"""fetch_website: read the visible text of a web page."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Callable
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field

from ..guardrails import check_url_is_safe

MAX_REDIRECTS = 3


class FetchWebsiteArgs(BaseModel):
    url: str = Field(description="Full link starting with http:// or https://, e.g. https://www.infosys.com")


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def fetch_website(
    url: str,
    client: httpx.Client,
    url_checker: Callable[[str], None] = check_url_is_safe,
    max_chars: int = 4000,
) -> dict:
    # Redirects are followed by hand so EVERY hop is safety-checked,
    # not just the first link.
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        url_checker(current)
        response = client.get(current)
        if not response.is_redirect:
            break
        current = urljoin(current, response.headers["location"])
    else:
        return {"error": f"Too many redirects starting from {url}"}

    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type:
        return {"error": f"Not a web page (content type: {content_type or 'unknown'})"}

    parser = _TextExtractor()
    parser.feed(response.text)
    return {"url": current, "title": parser.title.strip(), "text": " ".join(parser.parts)[:max_chars]}
