"""
Phase-3 Agent: Critic (保守派)
Simulates a harsh reviewer attacking mathematical stability and VRAM risks.
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


class Critic(BaseAgent):
    role = "Critic"
    system_prompt = (
        "You are the Critic — a skeptical, detail-oriented reviewer who simulates "
        "the harshest NeurIPS/ICLR reviewer. Your job is to find every flaw in the "
        "Proposer's experiment plan.\n\n"
        "Attack vectors:\n"
        "1. **Mathematical instability**: Gradient explosion/vanishing from the grafted "
        "   operator; numerical overflow in fp16/bf16; unbounded feature maps.\n"
        "2. **VRAM overflow**: Hidden memory costs the Proposer may have underestimated "
        "   (optimizer states for mixed-precision, KV-cache growth, activation peaks).\n"
        "3. **Experimental weakness**: Missing baselines, cherry-picked benchmarks, "
        "   unfair comparisons, missing ablations.\n"
        "4. **Novelty concerns**: Does this actually differ from prior work?\n"
        "5. **Reproducibility**: Are hyperparameters sufficiently specified?\n\n"
        "Output JSON with keys:\n"
        "- `criticisms`: list of {id, category, severity: 'FATAL'|'MAJOR'|'MINOR', "
        "  description, suggested_fix}\n"
        "- `overall_verdict`: 'REJECT' | 'WEAK_REJECT' | 'BORDERLINE' | 'WEAK_ACCEPT'\n"
        "- `confidence`: 1-5\n"
        "- `strengths`: list of strings\n"
        "- `weaknesses`: list of strings"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        user_msg = (
            f"## Proposer's Experiment Plan\n"
            f"{_safe_dump(context.get('proposal', {}), 5000)}\n\n"
            f"## Hardware Audit\n"
            f"{_safe_dump(context.get('hardware_audit', {}), 2000)}\n\n"
            f"## Novelty Audit\n"
            f"{_safe_dump(context.get('novelty_audit', {}), 2000)}\n\n"
            "Act as the harshest possible reviewer. Find every flaw. Return JSON."
        )

        result = await self._ask_structured(user_msg)
        context["criticism"] = result
        return context
