"""
Phase-3 Agent: Mediator (调解员)
Synthesizes Proposer's plan and Critic's attacks into a final experiment design.
"""

from __future__ import annotations

import json
from typing import Any

from rf.base.agent import BaseAgent


def _safe_dump(obj: Any, max_len: int = 4000) -> str:
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(obj)
    return text[:max_len]


class Mediator(BaseAgent):
    role = "Mediator"
    system_prompt = (
        "You are the Mediator — a senior research scientist who resolves debates "
        "between the bold Proposer and the skeptical Critic. You produce a **final, "
        "actionable experiment plan** that:\n\n"
        "1. Keeps the Proposer's innovative core idea intact.\n"
        "2. Addresses every FATAL and MAJOR criticism from the Critic.\n"
        "3. Adds safety mechanisms (gradient clipping, loss spike detection, fallback).\n"
        "4. Produces a realistic timeline.\n\n"
        "Output JSON with keys:\n"
        "- `final_plan_title`: string\n"
        "- `addressed_criticisms`: list of {criticism_id, resolution}\n"
        "- `unresolved_risks`: list of {description, mitigation}\n"
        "- `final_grafting_strategy`: object\n"
        "- `final_training_recipe`: object\n"
        "- `final_implementation_code`: string (complete Python)\n"
        "- `final_benchmarks`: list of {name, metric}\n"
        "- `ablation_studies`: list of {variable, values_to_test}\n"
        "- `timeline_days`: int\n"
        "- `go_no_go_decision`: 'GO' | 'CONDITIONAL_GO' | 'NO_GO'\n"
        "- `rationale`: summary paragraph"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        user_msg = (
            f"## Proposer's Plan\n{_safe_dump(context.get('proposal', {}))}\n\n"
            f"## Critic's Review\n{_safe_dump(context.get('criticism', {}))}\n\n"
            f"## Hardware Budget\n{context.get('hardware_budget_gb', 24)} GB\n\n"
            "Synthesize both sides. Produce the final experiment plan. Return JSON."
        )

        result = await self._ask_structured(user_msg)
        context["final_plan"] = result
        return context
