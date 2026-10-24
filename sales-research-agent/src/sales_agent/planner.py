"""Planning: before doing anything, decide IF we should research and WHAT steps to take."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .llm import LLM, chat_json

PLANNER_PROMPT = """You plan research for a B2B sales team at {seller}.

Read the user's request and return JSON with these keys:
- "on_topic": true only if the user wants research about a company or organisation
  (for sales, a meeting or a lead check). Otherwise false.
- "reason": one short line explaining on_topic.
- "company": the company name to research ("" if none).
- "steps": 3 to 6 short research steps. Use only these tools: {tools}.

Example JSON:
{{"on_topic": true, "reason": "Meeting prep for a company", "company": "Infosys",
  "steps": ["Get the company profile with search_company", "Get latest news with get_news",
            "Read the company website with fetch_website",
            "Find matching case studies with search_internal_docs", "Score the lead with score_lead"]}}"""


class Plan(BaseModel):
    on_topic: bool
    reason: str = ""
    company: str = ""
    steps: list[str] = Field(default_factory=list)


def make_plan(llm: LLM, question: str, seller: str, tool_names: list[str]) -> Plan:
    messages = [
        {"role": "system", "content": PLANNER_PROMPT.format(seller=seller, tools=", ".join(tool_names))},
        {"role": "user", "content": question},
    ]
    return chat_json(llm, messages, Plan)
