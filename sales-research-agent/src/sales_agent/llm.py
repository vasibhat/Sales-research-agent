"""The only file that talks to OpenAI.

Everything else uses the small `LLM` interface below. That keeps the agent
easy to test (tests plug in a fake LLM) and easy to switch to another provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError


@dataclass
class ToolCall:
    """One tool request from the model."""

    id: str  # the "token number": the tool result must carry the same id back
    name: str
    arguments: str  # JSON text, exactly as the model wrote it


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    def as_message(self) -> dict[str, Any]:
        """The assistant message to add to history. It must come BEFORE the tool results."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return message


class LLM(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> LLMReply: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAILLM:
    """Real LLM using OpenAI's Chat Completions and Embeddings APIs."""

    def __init__(self, model: str, embedding_model: str, client: Any = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(max_retries=3, timeout=60)  # reads OPENAI_API_KEY
        self.client = client
        self.model = model
        self.embedding_model = embedding_model

    def chat(self, messages, tools=None, json_mode=False) -> LLMReply:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
            for c in (message.tool_calls or [])
        ]
        usage = response.usage
        return LLMReply(
            content=message.content,
            tool_calls=calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in response.data]


T = TypeVar("T", bound=BaseModel)


def chat_json(llm: LLM, messages: list[dict[str, Any]], schema: type[T], retries: int = 1) -> T:
    """Ask for JSON and check it with Pydantic.

    If the JSON is wrong, show the model its mistake and ask once more.
    (JSON mode needs the word "JSON" somewhere in the prompt.)
    """
    messages = list(messages)
    for attempt in range(retries + 1):
        text = llm.chat(messages, json_mode=True).content or ""
        try:
            return schema.model_validate_json(text)
        except ValidationError as error:
            if attempt == retries:
                raise
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"That JSON was not valid: {error}. Return corrected JSON only."})
    raise AssertionError("unreachable")
