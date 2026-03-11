"""
Phase-1 Agent: Trend Analyst (热度分析师)
Captures technology burst points across arXiv, GitHub, OpenReview.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rf.base.agent import BaseAgent
from rf.tools import fetch_trending_repos, search_arxiv, search_openreview

logger = logging.getLogger("rf")


class TrendAnalyst(BaseAgent):
    role = "TrendAnalyst"
    system_prompt = (
        "You are a senior AI research trend analyst. Your mission is to identify "
        "technology burst points in deep learning — specifically around new attention "
        "mechanisms, efficient architectures, and quantization innovations.\n\n"
        "You will receive raw data from arXiv, GitHub, and OpenReview. Analyze the data "
        "across the following dimensions:\n"
        "1. **Social/community buzz**: GitHub star velocity, fork counts, discussion volume.\n"
        "2. **Academic momentum**: arXiv submission recency, OpenReview reviewer scores.\n"
        "3. **Technical novelty**: new operators, paradigms, or training recipes.\n\n"
        "Output a JSON object with keys:\n"
        "- `trending_topics`: list of {topic, score_0_to_100, evidence, key_papers}\n"
        "- `recommended_focus`: the single most promising research direction (string)\n"
        "- `reasoning`: detailed analysis paragraph (string)"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        topic = context.get("topic", "linear attention efficient LLM")

        # Gather raw intelligence (all three run concurrently)
        import asyncio

        arxiv_task = search_arxiv(
            query=f"({topic}) AND (cat:cs.CL OR cat:cs.LG)",
            max_results=context.get("arxiv_max_results", 50),
        )
        github_task = fetch_trending_repos(
            query=topic,
            min_stars=context.get("github_stars_threshold", 100),
        )
        openreview_task = search_openreview(query=topic)

        arxiv_papers, github_repos, openreview_papers = await asyncio.gather(
            arxiv_task, github_task, openreview_task
        )

        logger.info(
            "TrendAnalyst gathered: %d arXiv, %d GitHub, %d OpenReview",
            len(arxiv_papers), len(github_repos), len(openreview_papers),
        )

        # Build prompt — truncate to manage token limits
        user_msg = (
            f"## Research Topic\n{topic}\n\n"
            f"## arXiv Papers (latest {len(arxiv_papers)} found)\n"
            f"{json.dumps(arxiv_papers[:15], indent=2, ensure_ascii=False)}\n\n"
            f"## GitHub Trending Repos ({len(github_repos)} found)\n"
            f"{json.dumps(github_repos[:10], indent=2, ensure_ascii=False)}\n\n"
            f"## OpenReview Submissions ({len(openreview_papers)} found)\n"
            f"{json.dumps(openreview_papers[:10], indent=2, ensure_ascii=False)}\n\n"
            "Analyze the above data and produce your trend report in JSON."
        )

        result = await self._ask_structured(user_msg)
        context["trend_report"] = result
        context["arxiv_papers"] = arxiv_papers
        context["github_repos"] = github_repos
        context["openreview_papers"] = openreview_papers
        return context
