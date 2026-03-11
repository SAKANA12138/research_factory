"""
Phase-1 Agent: Novelty Auditor (查重比对官)
Checks research novelty against prior art.
"""

from __future__ import annotations

import json
from typing import Any

from rf.base.agent import BaseAgent


class NoveltyAuditor(BaseAgent):
    role = "NoveltyAuditor"
    system_prompt = (
        "You are a meticulous academic novelty auditor. You specialize in detecting "
        "whether a proposed research idea has been previously published or if highly "
        "similar work already exists.\n\n"
        "Your methodology:\n"
        "1. Compare the **core operator/module** being proposed against known prior art.\n"
        "2. Compare the **experimental pathway** (base model + grafting strategy + benchmarks).\n"
        "3. If a paper achieves the same goal via a similar grafting approach, flag as "
        "'LOW_NOVELTY' and suggest pivoting.\n\n"
        "Output JSON with keys:\n"
        "- `novelty_score`: 0-100\n"
        "- `novelty_level`: 'HIGH' | 'MEDIUM' | 'LOW'\n"
        "- `similar_works`: list of {title, arxiv_id, similarity_reason}\n"
        "- `differentiation_suggestions`: list of strings\n"
        "- `verdict`: summary paragraph"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        topic = context.get("topic", "")
        arxiv_papers = context.get("arxiv_papers", [])
        trend_report = context.get("trend_report", {})

        paper_summaries = []
        for p in arxiv_papers[:25]:
            paper_summaries.append(
                f"- **{p['title']}** ({p['arxiv_id']}): {p['abstract'][:200]}..."
            )
        corpus_text = "\n".join(paper_summaries) if paper_summaries else "(No papers found)"

        recommended = trend_report.get("recommended_focus", "")
        if isinstance(recommended, dict):
            recommended = json.dumps(recommended, ensure_ascii=False)

        user_msg = (
            f"## Proposed Research Idea\n{topic}\n\n"
            f"## Recommended Focus from Trend Analyst\n{recommended}\n\n"
            f"## Recent Related Papers (last 2 years)\n{corpus_text}\n\n"
            "Perform a novelty audit. Return JSON."
        )

        result = await self._ask_structured(user_msg)
        context["novelty_audit"] = result
        return context
