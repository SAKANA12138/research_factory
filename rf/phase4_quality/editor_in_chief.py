"""
Phase-4 Agent: Editor-in-Chief (总编辑 / EIC)
Compiles all outputs into LaTeX source and final report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rf.base.agent import BaseAgent
from rf.tools.latex_compiler import compile_pdf

logger = logging.getLogger("rf")


def _safe_dump(obj: Any, max_len: int = 2500) -> str:
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(obj)
    return text[:max_len]


class EditorInChief(BaseAgent):
    role = "EditorInChief"
    system_prompt = (
        "You are the Editor-in-Chief (EIC) of Research Factory. Your job is to compile "
        "all debate records, self-inspection results, experiment designs, and mathematical "
        "formulations into a complete, submission-ready LaTeX document.\n\n"
        "The paper must follow standard ML conference formatting with these sections:\n"
        "1. Title & Abstract\n"
        "2. Introduction (motivation, contributions)\n"
        "3. Related Work\n"
        "4. Method (with mathematical formulation and grafting strategy)\n"
        "5. Experimental Setup (benchmarks, baselines, hyperparameters)\n"
        "6. Results & Analysis (with placeholder tables/figures)\n"
        "7. Ablation Studies\n"
        "8. Conclusion & Future Work\n"
        "9. References\n\n"
        "Output JSON with keys:\n"
        "- `paper_title`: string\n"
        "- `abstract`: string\n"
        "- `latex_source`: complete LaTeX source as a string\n"
        "- `bibtex_entries`: string (BibTeX)\n"
        "- `figure_descriptions`: list of {figure_id, caption, description}\n"
        "- `compilation_notes`: list of strings"
    )

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        user_msg = (
            "## COMPILATION TASK\n"
            "Compile ALL of the following research artifacts into a complete LaTeX paper.\n\n"
            f"### Research Topic\n{context.get('topic', '')}\n\n"
            f"### Trend Report\n{_safe_dump(context.get('trend_report', {}))}\n\n"
            f"### Hardware Audit\n{_safe_dump(context.get('hardware_audit', {}), 1500)}\n\n"
            f"### Novelty Audit\n{_safe_dump(context.get('novelty_audit', {}), 1500)}\n\n"
            f"### Source Analysis\n{_safe_dump(context.get('source_analysis', {}), 2000)}\n\n"
            f"### Math Translation\n{_safe_dump(context.get('math_translation', {}), 2000)}\n\n"
            f"### Conflict Report\n{_safe_dump(context.get('conflict_report', {}), 1500)}\n\n"
            f"### Final Experiment Plan\n{_safe_dump(context.get('final_plan', {}))}\n\n"
            f"### Quality Report\n{_safe_dump(context.get('quality_report', {}), 1500)}\n\n"
            f"### Publication Strategy\n{_safe_dump(context.get('publication_strategy', {}), 1500)}\n\n"
            "Now compile the full LaTeX paper. Return JSON."
        )

        result = await self._ask_structured(user_msg)
        context["paper_draft"] = result

        # Write LaTeX source and attempt compilation
        latex_source = result.get("latex_source", "")
        if latex_source and isinstance(latex_source, str) and len(latex_source) > 100:
            output_dir = context.get("output_dir", "output/")
            tex_path = Path(output_dir) / "paper.tex"
            tex_path.parent.mkdir(parents=True, exist_ok=True)
            tex_path.write_text(latex_source, encoding="utf-8")
            context["tex_path"] = str(tex_path)
            logger.info("LaTeX source written to %s (%d chars)", tex_path, len(latex_source))

            # Compile PDF
            pdf_path = compile_pdf(str(tex_path))
            if pdf_path:
                context["pdf_path"] = pdf_path
                logger.info("PDF compiled: %s", pdf_path)

            # Write BibTeX
            bib_source = result.get("bibtex_entries", "")
            if bib_source and isinstance(bib_source, str):
                bib_path = tex_path.with_suffix(".bib")
                bib_path.write_text(bib_source, encoding="utf-8")
                context["bib_path"] = str(bib_path)
        else:
            logger.warning("EIC did not produce valid LaTeX source")

        return context
