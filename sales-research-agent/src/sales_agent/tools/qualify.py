"""score_lead: score a company against our Ideal Customer Profile (ICP).

This is plain Python rules, not the model. Scores must be the same every time
for the same input, so the sales team can trust and compare them.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ScoreLeadArgs(BaseModel):
    industry: str = Field(description="Company industry, e.g. 'retail' or 'manufacturing'")
    employee_count: int | None = Field(default=None, description="Number of employees, if known")
    country: str | None = Field(default=None, description="Head office country, if known")
    signals: list[str] = Field(
        default_factory=list,
        description="Buying signals found in research, e.g. 'raised new funding', 'opening new warehouse'",
    )


def load_icp(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_lead(industry: str, employee_count: int | None, country: str | None, signals: list[str], icp: dict) -> dict:
    score = 0
    reasons: list[str] = []

    industry_l = industry.lower()
    if any(t in industry_l or industry_l in t for t in icp["target_industries"]):
        score += 35
        reasons.append(f"Industry '{industry}' is one we target (+35)")
    else:
        reasons.append(f"Industry '{industry}' is not a main target (+0)")

    if employee_count is None:
        score += 5
        reasons.append("Company size unknown (+5)")
    elif employee_count >= icp["ideal_employees"]:
        score += 25
        reasons.append(f"{employee_count} employees: ideal size (+25)")
    elif employee_count >= icp["min_employees"]:
        score += 15
        reasons.append(f"{employee_count} employees: good size (+15)")
    else:
        reasons.append(f"{employee_count} employees: smaller than our minimum of {icp['min_employees']} (+0)")

    if not country:
        score += 5
        reasons.append("Country unknown (+5)")
    elif country.lower() in icp["target_countries"]:
        score += 15
        reasons.append(f"{country} is a target country (+15)")
    else:
        reasons.append(f"{country} is outside our target countries (+0)")

    matched = [k for k in icp["buying_signals"] if any(k in s.lower() for s in signals)]
    signal_points = min(25, 5 * len(matched))
    score += signal_points
    if matched:
        reasons.append(f"Buying signals: {', '.join(matched)} (+{signal_points})")

    grade = "Hot" if score >= 70 else "Warm" if score >= 45 else "Cold"
    return {"score": score, "grade": grade, "reasons": reasons}
