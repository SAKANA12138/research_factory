"""
Phase-1 Agent: Hardware Guardian (可行性审计师)
VRAM red-line audit for proposed architectures.
"""

from __future__ import annotations

import json
from typing import Any

from rf.base.agent import BaseAgent


class HardwareGuardian(BaseAgent):
    role = "HardwareGuardian"
    system_prompt = (
        "You are a hardware-aware deep learning engineer specializing in VRAM budgeting "
        "and memory optimization. Your job is to audit proposed research ideas for "
        "feasibility on a given GPU budget.\n\n"
        "Rules:\n"
        "1. Prioritize partial module grafting on 8B-14B parameter models.\n"
        "2. Apply fine-grained INT8/INT4 quantization estimates (GPTQ, AWQ, bitsandbytes).\n"
        "3. Consider gradient checkpointing, LoRA, and activation offloading.\n"
        "4. Estimate: model params (GB) + optimizer states (GB) + activations (GB) + KV-cache (GB).\n\n"
        "Output a JSON object with keys:\n"
        "- `vram_breakdown`: {params_gb, optimizer_gb, activations_gb, kv_cache_gb, total_gb}\n"
        "- `fits_budget`: bool\n"
        "- `optimizations_applied`: list of strings\n"
        "- `recommendations`: free-text suggestions if over budget\n"
        "- `risk_level`: 'LOW' | 'MEDIUM' | 'HIGH'"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        topic = context.get("topic", "")
        trend_report = context.get("trend_report", {})
        budget_gb = context.get("hardware_budget_gb", 24)
        target_size = context.get("target_model_size", "8B-14B")
        quant = context.get("quantization", "INT8")

        recommended = trend_report.get("recommended_focus", "")
        if isinstance(recommended, dict):
            recommended = json.dumps(recommended, ensure_ascii=False)

        user_msg = (
            f"## Hardware Budget\n"
            f"- GPU VRAM: {budget_gb} GB\n"
            f"- Target model size: {target_size}\n"
            f"- Quantization scheme: {quant}\n\n"
            f"## Research Topic\n{topic}\n\n"
            f"## Recommended Focus\n{recommended}\n\n"
            "Perform a VRAM feasibility audit. Return JSON."
        )

        result = await self._ask_structured(user_msg)
        context["hardware_audit"] = result
        return context
