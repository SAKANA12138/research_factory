"""
Phase-3 Agent: Proposer (激进派)
Proposes concrete module-stitching implementation with full code.
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


class Proposer(BaseAgent):
    role = "Proposer"
    system_prompt = (
        "You are the bold, innovative Proposer in an academic research debate. "
        "Your role is to take the analyzed grafting points, the translated math, "
        "and the conflict analysis, then propose a **concrete, executable experiment plan** "
        "with full implementation details.\n\n"
        "Your proposal must include:\n"
        "1. **Grafting Strategy**: Which layers to replace, partial vs full replacement.\n"
        "2. **Training Recipe**: Learning rate, scheduler, batch size, gradient accumulation, "
        "   LoRA rank (if applicable), number of training steps.\n"
        "3. **Implementation Code**: Complete Python code for the grafted model and training loop.\n"
        "4. **Benchmarks**: Which benchmarks to evaluate on (e.g., LongBench, RULER, MMLU, "
        "   perplexity on PG-19).\n"
        "5. **Expected Outcomes**: Quantitative predictions with confidence intervals.\n\n"
        "Output JSON with keys:\n"
        "- `proposal_title`: string\n"
        "- `grafting_strategy`: {method, layers_affected, replacement_ratio}\n"
        "- `training_recipe`: {lr, scheduler, batch_size, grad_accum, lora_rank, steps, "
        "  warmup_ratio, optimizer}\n"
        "- `implementation_code`: string (full Python)\n"
        "- `benchmarks`: list of {name, metric, expected_baseline, expected_ours}\n"
        "- `expected_outcomes`: string\n"
        "- `risk_factors`: list of strings"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        user_msg = (
            f"## Research Topic\n{context.get('topic', '')}\n\n"
            f"## Source Analysis\n{_safe_dump(context.get('source_analysis', {}))}\n\n"
            f"## Math Translation\n{_safe_dump(context.get('math_translation', {}))}\n\n"
            f"## Conflict Report\n{_safe_dump(context.get('conflict_report', {}), 2000)}\n\n"
            f"## Hardware Constraints\n"
            f"VRAM Budget: {context.get('hardware_budget_gb', 24)} GB\n"
            f"Quantization: {context.get('quantization', 'INT8')}\n"
            f"Target Model: {context.get('target_model_size', '8B-14B')}\n\n"
            "Propose a complete experiment plan with implementation code. "
            "Be aggressive and innovative. Return JSON."
        )

        result = await self._ask_structured(user_msg)
        context["proposal"] = result
        return context
