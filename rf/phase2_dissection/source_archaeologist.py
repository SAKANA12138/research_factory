"""
Phase-2 Agent: Source Archaeologist (源码考古学家)
Extracts tensor flow topology from modeling.py source files.
"""

from __future__ import annotations

import json
from typing import Any

from rf.base.agent import BaseAgent


def _safe_dump(obj: Any, max_len: int = 3000) -> str:
    """Safely JSON-dump an object, truncating if needed."""
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(obj)
    return text[:max_len]


class SourceArchaeologist(BaseAgent):
    role = "SourceArchaeologist"
    system_prompt = (
        "You are an expert deep learning source code analyst — a 'Source Archaeologist'. "
        "Your mission is to dissect `modeling_*.py` files from HuggingFace Transformers "
        "or custom repositories to extract:\n\n"
        "1. **Tensor flow graph**: Input → Embedding → Attention → FFN → Output, with "
        "   exact tensor shapes at each stage (batch, seq_len, hidden_dim, num_heads, head_dim).\n"
        "2. **Grafting points**: Identify which submodules can be replaced (e.g., swap "
        "   `SdpaAttention` with a linear attention variant) and annotate the exact class "
        "   names, method signatures, and tensor shapes at the boundary.\n"
        "3. **Hidden dependencies**: Weight tying, shared buffers, custom CUDA kernels, "
        "   or non-standard forward hooks that could break after grafting.\n\n"
        "Output JSON with keys:\n"
        "- `model_architecture`: {name, num_layers, hidden_dim, num_heads, head_dim, vocab_size}\n"
        "- `tensor_flow`: list of {layer_name, input_shape, output_shape, operation}\n"
        "- `grafting_points`: list of {module_path, class_name, input_tensors, output_tensors, "
        "  swap_candidates}\n"
        "- `hidden_dependencies`: list of {type, description, risk_level}\n"
        "- `source_files_analyzed`: list of file paths"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        topic = context.get("topic", "")
        trend_report = context.get("trend_report", {})
        target_size = context.get("target_model_size", "8B-14B")

        recommended = trend_report.get("recommended_focus", "")
        if isinstance(recommended, dict):
            recommended = json.dumps(recommended, ensure_ascii=False)

        user_msg = (
            f"## Research Topic\n{topic}\n\n"
            f"## Target Model Size\n{target_size}\n\n"
            f"## Trend Report Focus\n{recommended}\n\n"
            "Analyze the typical HuggingFace `modeling_*.py` architecture for a model in "
            f"the {target_size} parameter class (e.g., LLaMA-3-8B, Qwen2.5-14B, Mistral-7B). "
            "Extract the tensor flow graph, identify grafting points where a linear attention "
            "operator could be inserted, and flag any hidden dependencies.\n\n"
            "Return your analysis as structured JSON."
        )

        result = await self._ask_structured(user_msg)
        context["source_analysis"] = result
        return context
