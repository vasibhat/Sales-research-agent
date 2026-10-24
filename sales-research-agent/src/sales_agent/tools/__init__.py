"""Tool registry: the list of tools the model may ask for, and the code that runs them.

Remember: the model never runs a tool. It only ASKS. This registry is where
our code checks the request and actually runs the function.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from pydantic import BaseModel, ValidationError

USER_AGENT = "Mozilla/5.0 (compatible; SalesResearchAgent/1.0; learning project)"


def make_http_client(follow_redirects: bool = True) -> httpx.Client:
    return httpx.Client(timeout=15, follow_redirects=follow_redirects, headers={"User-Agent": USER_AGENT})


@dataclass
class Tool:
    name: str
    description: str  # the model reads this to decide WHEN to use the tool
    args_model: type[BaseModel]  # Pydantic model = argument rules + JSON schema for the model
    func: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": parameters},
        }


@dataclass
class ToolResult:
    content: str
    ok: bool


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def run(self, name: str, arguments: str) -> ToolResult:
        """Run one tool. Never raises: errors go back to the model as text so it can try something else."""
        tool = self._tools.get(name)
        if tool is None:
            return _error(f"Unknown tool '{name}'. Available tools: {', '.join(self._tools)}")
        try:
            args = tool.args_model.model_validate_json(arguments or "{}")
        except ValidationError as error:
            return _error(f"Bad arguments for {name}: {error}")
        try:
            output = tool.func(**args.model_dump())
        except Exception as error:  # noqa: BLE001 - any tool failure is reported to the model
            return _error(f"{name} failed: {type(error).__name__}: {error}")
        text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
        return ToolResult(text, ok=not (isinstance(output, dict) and "error" in output))


def _error(message: str) -> ToolResult:
    return ToolResult(json.dumps({"error": message}, ensure_ascii=False), ok=False)
