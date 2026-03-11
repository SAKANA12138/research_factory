"""LaTeX source generation & compilation utility."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("rf.tools")


def compile_pdf(tex_path: str) -> str | None:
    """
    Compile .tex → .pdf via pdflatex (two passes for references).
    Returns PDF path or None.
    """
    tex = Path(tex_path)
    if not tex.exists():
        logger.warning("TeX file not found: %s", tex_path)
        return None
    try:
        for _pass in range(2):  # Two passes for references
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=str(tex.parent),
                capture_output=True,
                timeout=120,
                check=False,  # Don't raise on non-zero exit (common with missing packages)
            )
        pdf_path = tex.with_suffix(".pdf")
        return str(pdf_path) if pdf_path.exists() else None
    except FileNotFoundError:
        logger.info("pdflatex not found — skipping PDF compilation")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("pdflatex timed out")
        return None
