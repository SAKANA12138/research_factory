"""
Phase-2 Agent: Conflict Detector (冲突检测员)
Detects Dim / Norm / PE rejection risks in module grafting.
"""

from __future__ import annotations

import json
from typing import Any

from rf.base.agent import BaseAgent


class ConflictDetector(BaseAgent):
    role = "ConflictDetector"
    system_prompt = (
        "You are a transplant-rejection specialist for neural network module grafting. "
        "When a new operator replaces an existing attention block, three categories of "
        "conflict can cause silent failure or training collapse:\n\n"
        "1. **Dimension Mismatch (Dim)**: Hidden dim, head dim, or intermediate dim "
        "   incompatibility between the donor operator and the host model.\n"
        "2. **Normalization Conflict (Norm)**: Pre-Norm vs Post-Norm, RMSNorm vs LayerNorm, "
        "   or missing normalization in the donor operator.\n"
        "3. **Positional Encoding Conflict (PE)**: RoPE vs ALiBi vs learned PE vs no PE; "
        "   applying the wrong PE to the replacement operator.\n\n"
        "Output JSON with keys:\n"
        "- `conflicts`: list of {category: 'DIM'|'NORM'|'PE', severity: 'CRITICAL'|'WARNING'|'INFO', "
        "  description, affected_modules, fix_strategy}\n"
        "- `compatibility_score`: 0-100\n"
        "- `integration_checklist`: list of {step, description, status: 'PENDING'}\n"
        "- `overall_risk`: 'LOW' | 'MEDIUM' | 'HIGH'"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        source_analysis = context.get("source_analysis", {})
        math_translation = context.get("math_translation", {})

        pytorch_code = math_translation.get("pytorch_code", "# N/A")
        if not isinstance(pytorch_code, str):
            pytorch_code = str(pytorch_code)

        user_msg = (
            f"## Host Model Architecture\n"
            f"{json.dumps(source_analysis.get('model_architecture', {}), indent=2, ensure_ascii=False)}\n\n"
            f"## Tensor Flow (first 8 layers)\n"
            f"{json.dumps(source_analysis.get('tensor_flow', [])[:8], indent=2, ensure_ascii=False)}\n\n"
            f"## Donor Operator\n"
            f"Name: {math_translation.get('operator_name', 'N/A')}\n"
            f"Math: {math_translation.get('math_formulation', 'N/A')}\n\n"
            f"## PyTorch Code\n```python\n{pytorch_code[:10000]}\n```\n\n"
            f"## Shape Alignment\n"
            f"{json.dumps(math_translation.get('shape_alignment', []), indent=2, ensure_ascii=False)[:7500]}\n\n"
            "Perform a comprehensive conflict detection analysis across DIM, NORM, and PE "
            "dimensions. Return structured JSON."
        )

        result = await self._ask_structured(user_msg)
        context["conflict_report"] = result
        return context
