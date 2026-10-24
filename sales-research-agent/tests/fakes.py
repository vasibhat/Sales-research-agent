"""A fake LLM for tests: no API key, no cost, same answer every time."""

from __future__ import annotations

import copy
import json
import re
import zlib

from sales_agent.llm import LLMReply, ToolCall


class ScriptedLLM:
    """Returns pre-written replies in order and records every request it was sent."""

    def __init__(self, replies: list[LLMReply] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, json_mode=False) -> LLMReply:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "json_mode": json_mode})
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [bag_of_words(t) for t in texts]


def bag_of_words(text: str, dims: int = 256) -> list[float]:
    """A toy embedding: texts that share words get similar vectors."""
    vector = [0.0] * dims
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[zlib.crc32(word.encode()) % dims] += 1.0
    return vector


def text(content: str) -> LLMReply:
    return LLMReply(content=content, model="fake", input_tokens=100, output_tokens=20)


def tool_call(name: str, args: dict, call_id: str = "call_1") -> LLMReply:
    return LLMReply(content=None, tool_calls=[ToolCall(call_id, name, json.dumps(args))], model="fake")
