"""
Phase-4 Agent: Publication Strategist (投稿战略官)
Expected-value modeling for submission target selection.
"""

from __future__ import annotations

import json
from typing import Any

from rf.base.agent import BaseAgent


def _safe_dump(obj: Any, max_len: int = 3000) -> str:
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(obj)
    return text[:max_len]


class PublicationStrategist(BaseAgent):
    role = "PublicationStrategist"
    system_prompt = (
        "You are a Publication Strategist who models the expected return of submitting "
        "a research paper to various venues. You use the formula:\n\n"
        "  Expected Return = P(acceptance) × Impact Factor (or CCF rank score)\n\n"
        "You consider:\n"
        "1. **Novelty score** from the Novelty Auditor.\n"
        "2. **Quality score** from the Quality Inspector.\n"
        "3. **Experimental hardness** (how convincing the empirical results are).\n"
        "4. **Venue-specific fit** (e.g., ICLR rewards novelty, NeurIPS rewards breadth, "
        "   CVPR rewards visual results).\n\n"
        "For each venue, estimate:\n"
        "- P(acceptance): 0.0 to 1.0\n"
        "- Impact score: CCF-A=10, CCF-B=6, CCF-C=3, Workshop=1, arXiv-only=0.5\n"
        "- Expected return = P × Impact\n\n"
        "Output JSON with keys:\n"
        "- `venues`: list of {name, type: 'conference'|'journal'|'preprint', "
        "  p_acceptance, impact_score, expected_return, fit_reasoning}\n"
        "- `recommended_strategy`: {reach_venue, safe_venue, backup_venue}\n"
        "- `timeline`: {reach_deadline, safe_deadline, backup_deadline}\n"
        "- `strategic_advice`: string"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        quality_report = context.get("quality_report", {})
        novelty_audit = context.get("novelty_audit", {})
        final_plan = context.get("final_plan", {})

        plan_title = final_plan.get("final_plan_title", "N/A") if isinstance(final_plan, dict) else "N/A"
        go_decision = final_plan.get("go_no_go_decision", "N/A") if isinstance(final_plan, dict) else "N/A"

        user_msg = (
            f"## Quality Report\n{_safe_dump(quality_report)}\n\n"
            f"## Novelty Audit\n{_safe_dump(novelty_audit, 2000)}\n\n"
            f"## Final Plan Summary\n"
            f"Title: {plan_title}\n"
            f"Go/No-Go: {go_decision}\n\n"
            "Model the expected publication returns and recommend a submission strategy. "
            "Return JSON."
        )

        result = await self._ask_structured(user_msg)
        context["publication_strategy"] = result
        return context
