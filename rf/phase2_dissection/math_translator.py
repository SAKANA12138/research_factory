"""
Phase-2 Agent: Math Translator (数学翻译官)
Converts grafting-point math formulas into PyTorch-aligned code.
"""

from __future__ import annotations

import json
from typing import Any

from rf.base.agent import BaseAgent


class MathTranslator(BaseAgent):
    role = "MathTranslator"
    system_prompt = (
        "You are a mathematical formalization expert who bridges theory and implementation. "
        "Given a set of grafting points and a target attention operator (e.g., linear "
        "attention with feature maps φ(Q)φ(K)ᵀV), you produce:\n\n"
        "1. **Mathematical specification**: LaTeX equations for the new operator.\n"
        "2. **PyTorch implementation**: Dimension-aligned code that can be drop-in "
        "   swapped at the identified grafting point.\n"
        "3. **Shape alignment proof**: Step-by-step verification that input/output shapes "
        "   match the original module.\n\n"
        "Output JSON with keys:\n"
        "- `operator_name`: string\n"
        "- `math_formulation`: LaTeX string\n"
        "- `pytorch_code`: complete Python/PyTorch code as a string\n"
        "- `shape_alignment`: list of {step, lhs_shape, rhs_shape, match: bool}\n"
        "- `numerical_stability_notes`: list of strings\n"
        "- `complexity_analysis`: {time, space} with big-O notation"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        source_analysis = context.get("source_analysis", {})
        trend_report = context.get("trend_report", {})

        grafting_points = source_analysis.get("grafting_points", [])
        architecture = source_analysis.get("model_architecture", {})
        recommended = trend_report.get("recommended_focus", "")
        if isinstance(recommended, dict):
            recommended = json.dumps(recommended, ensure_ascii=False)

        user_msg = (
            f"## Model Architecture\n"
            f"{json.dumps(architecture, indent=2, ensure_ascii=False)}\n\n"
            f"## Identified Grafting Points\n"
            f"{json.dumps(grafting_points, indent=2, ensure_ascii=False)[:15000]}\n\n"
            f"## Recommended Operator Focus\n{recommended}\n\n"
            "For the top-priority grafting point, produce:\n"
            "1. The mathematical formulation of the replacement attention operator.\n"
            "2. A complete PyTorch `nn.Module` implementation.\n"
            "3. A step-by-step shape alignment proof.\n\n"
            "Return as structured JSON."
        )

        result = await self._ask_structured(user_msg)
        context["math_translation"] = result
        return context
