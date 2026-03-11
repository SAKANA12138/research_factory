"""
Phase-4 Agent: Quality Inspector (质量自检员)
Multi-dimensional self-assessment of the research output.
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


class QualityInspector(BaseAgent):
    role = "QualityInspector"
    system_prompt = (
        "You are a Quality Self-Inspector — the final gatekeeper before a research "
        "paper is submitted. You evaluate the work across multiple dimensions:\n\n"
        "1. **Logical rigor** (0-25): Are there mathematical gaps or unproven claims?\n"
        "2. **Experimental completeness** (0-25): Coverage of mainstream benchmarks, "
        "   ablation studies, statistical significance.\n"
        "3. **Figure/table quality** (0-25): Professional typography, clear axis labels, "
        "   consistent color schemes, reproducible plotting code.\n"
        "4. **Writing quality** (0-25): Academic English, clear motivation, well-structured "
        "   related work, concise yet complete.\n\n"
        "Output JSON with keys:\n"
        "- `scores`: {logical_rigor, experimental_completeness, figure_quality, writing_quality}\n"
        "- `total_score`: 0-100\n"
        "- `grade`: 'A' | 'B' | 'C' | 'D' | 'F'\n"
        "- `issues`: list of {dimension, severity, description, fix_action}\n"
        "- `pass_threshold_met`: bool (threshold = 70)\n"
        "- `executive_summary`: string"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        user_msg = (
            f"## Final Experiment Plan\n{_safe_dump(context.get('final_plan', {}), 5000)}\n\n"
            f"## Mathematical Formulation\n{_safe_dump(context.get('math_translation', {}))}\n\n"
            f"## Conflict Report\n{_safe_dump(context.get('conflict_report', {}), 2000)}\n\n"
            "Perform a comprehensive quality self-inspection. Return JSON."
        )

        result = await self._ask_structured(user_msg)
        context["quality_report"] = result
        return context
