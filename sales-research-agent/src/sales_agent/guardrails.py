"""Safety checks on what goes IN, what tools are allowed to touch, and what comes OUT."""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urlparse


class GuardrailError(Exception):
    """Raised when a request or an action is blocked."""


# --- Input -----------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|above|prior|earlier)\s+instructions",
    r"(reveal|show|print)\s+(me\s+)?(your\s+|the\s+)?system\s+prompt",
    r"you\s+are\s+now\s+",
    r"developer\s+mode",
    r"jailbreak",
]


def check_input(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        raise GuardrailError("Please type a question, for example: 'Research Infosys for a meeting'.")
    if len(text) > max_chars:
        raise GuardrailError(f"The question is too long ({len(text)} characters). The limit is {max_chars}.")
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise GuardrailError("This request tries to change the agent's instructions, so it was blocked.")
    return text


# --- Tools -----------------------------------------------------------------

def clip(text: str, max_chars: int) -> str:
    """Keep tool output short so one big web page cannot fill the whole context."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [cut, {len(text) - max_chars} more characters]"


def check_url_is_safe(url: str) -> None:
    """Block links that point inside our own network (protects against SSRF attacks)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise GuardrailError(f"Only http and https links are allowed, got '{parsed.scheme or url}'.")
    host = parsed.hostname
    if not host:
        raise GuardrailError("The link has no host name.")
    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror as error:
        raise GuardrailError(f"Could not find the website '{host}'.") from error
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0].split("%")[0])
        if not ip.is_global:
            raise GuardrailError(f"Links to private or local addresses are not allowed ({host}).")


# --- Output ----------------------------------------------------------------

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Indian 10-digit mobile style (98765 43210, +91-9876543210) or any number starting with "+".
PHONE_RE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}(?!\w|\.\d)|\+\d[\d\s-]{8,}\d")


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[email removed]", text)
    return PHONE_RE.sub("[phone removed]", text)


def redact_pii_in(value: Any) -> Any:
    """Remove personal emails and phone numbers from every string inside dicts and lists."""
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, list):
        return [redact_pii_in(v) for v in value]
    if isinstance(value, dict):
        return {k: redact_pii_in(v) for k, v in value.items()}
    return value
