"""The final output: a structured sales briefing, checked by Pydantic."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .guardrails import redact_pii_in
from .llm import LLM, chat_json
from .state import ResearchState

BRIEFING_PROMPT = """You write sales meeting briefings for {seller}.

Use ONLY the research notes given. If a fact is not in the notes, do not guess:
add it to "risks_or_unknowns" instead. Never include personal email addresses or phone numbers.
Use simple, clear English.
{preferences}
Return JSON with these keys:
- "company": string
- "summary": 2-4 sentences on what the company does
- "recent_news": list of short news lines (with dates if known)
- "lead_score": number 0-100 from the score_lead tool, or null if it was not scored
- "lead_grade": "Hot", "Warm", "Cold", or null
- "lead_reasons": list of strings
- "opportunities": list of {{"title": string, "why": string, "product": string}}
- "talking_points": 3-6 things the sales person can say or ask in the meeting
- "relevant_case_studies": list of our case studies that fit, with the result they got
- "risks_or_unknowns": list of strings
- "sources": list of links or document names that were used"""


class Opportunity(BaseModel):
    title: str
    why: str
    product: str = ""


class SalesBriefing(BaseModel):
    company: str
    summary: str
    recent_news: list[str] = Field(default_factory=list)
    lead_score: int | None = Field(default=None, ge=0, le=100)
    lead_grade: str | None = None
    lead_reasons: list[str] = Field(default_factory=list)
    opportunities: list[Opportunity] = Field(default_factory=list)
    talking_points: list[str] = Field(default_factory=list)
    relevant_case_studies: list[str] = Field(default_factory=list)
    risks_or_unknowns: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"SALES BRIEFING: {self.company}", "", "Summary", self.summary]

        def section(title: str, items: list[str]) -> None:
            if items:
                lines.extend(["", title, *[f"- {item}" for item in items]])

        if self.lead_score is not None:
            lines.extend(["", f"Lead score: {self.lead_score}/100 ({self.lead_grade or '-'})"])
            lines.extend(f"- {r}" for r in self.lead_reasons)
        section("Recent news", self.recent_news)
        section(
            "Opportunities",
            [f"{o.title}: {o.why}" + (f" (product: {o.product})" if o.product else "") for o in self.opportunities],
        )
        section("Talking points", self.talking_points)
        section("Relevant case studies", self.relevant_case_studies)
        section("Risks and unknowns", self.risks_or_unknowns)
        section("Sources", self.sources)
        return "\n".join(lines)


def write_briefing(llm: LLM, state: ResearchState, notes: str, seller: str, preferences: list[str]) -> SalesBriefing:
    prefs = ("User preferences to follow: " + "; ".join(preferences) + "\n") if preferences else ""
    messages = [
        {"role": "system", "content": BRIEFING_PROMPT.format(seller=seller, preferences=prefs)},
        {
            "role": "user",
            "content": f"Request: {state.question}\nCompany: {state.company}\n\n"
            f"Research notes from tools:\n{state.evidence_text()}\n\nResearcher's summary:\n{notes}",
        },
    ]
    return chat_json(llm, messages, SalesBriefing)


def redact_briefing(briefing: SalesBriefing) -> SalesBriefing:
    """Output guardrail: strip personal emails and phone numbers before anyone sees the briefing."""
    return SalesBriefing.model_validate(redact_pii_in(briefing.model_dump()))
