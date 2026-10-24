"""Evaluation: run the REAL agent on fixed test questions and score the results.

Unit tests (tests/) check the code. Evals check the QUALITY of the answers.
Run after every prompt or model change:  uv run python evals/run_evals.py

Two kinds of checks:
1. Rule checks - cheap and exact (was it refused? did it use 2+ tools? any PII?)
2. LLM judge   - a second model call rates how well the briefing sticks to the evidence (1-5)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from sales_agent.config import Settings
from sales_agent.factory import build_agent
from sales_agent.guardrails import EMAIL_RE, PHONE_RE
from sales_agent.llm import OpenAILLM, chat_json

HERE = Path(__file__).parent

JUDGE_PROMPT = """You check sales briefings written by an AI agent.
Rate from 1 to 5:
- "grounded": are the facts in the briefing supported by the evidence? (5 = all supported, 1 = mostly made up)
- "useful": would a sales person be ready for the meeting with this? (5 = fully ready)
Return JSON: {"grounded": int, "useful": int, "comment": "one short line"}"""


class Verdict(BaseModel):
    grounded: int
    useful: int
    comment: str = ""


def rule_checks(case: dict, result) -> dict[str, bool]:
    checks = {"refusal_correct": result.refused == case["expect_refused"]}
    if case["expect_refused"] or result.briefing is None:
        return checks
    b = result.briefing
    text = json.dumps(b.model_dump())
    checks.update({
        "right_company": case["company_contains"] in b.company.lower(),
        "used_2plus_tools": len(result.tools_used) >= 2,
        "lead_scored": b.lead_score is not None,
        "3plus_talking_points": len(b.talking_points) >= 3,
        "has_sources": bool(b.sources),
        "no_pii": not EMAIL_RE.search(text) and not PHONE_RE.search(text),
        "finished_in_limit": not result.stopped_early,
    })
    return checks


def main() -> int:
    settings = Settings.from_env()
    agent = build_agent(settings)
    judge = OpenAILLM(settings.model, settings.embedding_model)
    cases = json.loads((HERE / "cases.json").read_text())

    rows = []
    for case in cases:
        print(f"Running {case['name']} ...", flush=True)
        result = agent.run(case["question"])
        checks = rule_checks(case, result)
        verdict = None
        if result.briefing is not None:
            trace_file = result.trace.get("trace_file")
            evidence = Path(trace_file).read_text()[:15000] if trace_file else ""
            verdict = chat_json(judge, [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"Briefing:\n{result.briefing.model_dump_json()}\n\nEvidence (agent trace):\n{evidence}"},
            ], Verdict)
        rows.append({
            "case": case["name"],
            "passed": all(checks.values()),
            "checks": checks,
            "judge": verdict.model_dump() if verdict else None,
            "cost_usd": result.trace.get("cost_usd"),
            "seconds": result.trace.get("seconds"),
        })

    print(f"\n{'case':<20}{'pass':<6}{'grounded':<10}{'useful':<8}{'cost $':<10}failed checks")
    for r in rows:
        j = r["judge"] or {}
        failed = ", ".join(k for k, v in r["checks"].items() if not v) or "-"
        print(f"{r['case']:<20}{'yes' if r['passed'] else 'NO':<6}{j.get('grounded', '-')!s:<10}{j.get('useful', '-')!s:<8}{r['cost_usd']!s:<10}{failed}")

    (HERE / "results.json").write_text(json.dumps(rows, indent=2))
    passed = sum(r["passed"] for r in rows)
    print(f"\n{passed}/{len(rows)} cases passed. Details: evals/results.json")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
