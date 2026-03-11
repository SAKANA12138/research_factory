"""
RF-1.5 Main Pipeline Orchestrator
===================================
Runs all four phases sequentially, passing enriched context between agents.
Each agent is a self-contained LLM-backed worker calling GitHub Models API.

Usage:
    python -m rf.main                          # Use default config
    python -m rf.main --topic "Your topic"     # Override topic
    python -m rf.main --config custom.yaml     # Custom config file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rf.llm.client import LLMClient, LLMConfig

# Phase-1 agents
from rf.phase1_intelligence import HardwareGuardian, NoveltyAuditor, TrendAnalyst

# Phase-2 agents
from rf.phase2_dissection import ConflictDetector, MathTranslator, SourceArchaeologist

# Phase-3 agents
from rf.phase3_debate import Critic, Mediator, Proposer

# Phase-4 agents
from rf.phase4_quality import EditorInChief, PublicationStrategist, QualityInspector

console = Console()
logger = logging.getLogger("rf")


# ──────────────────────────────────────────────
# Pipeline Definition
# ──────────────────────────────────────────────

PHASES = {
    "Phase 1 — Intelligence Gathering 🔍": {
        "enabled_key": "intelligence",
        "agents": [
            ("TrendAnalyst", TrendAnalyst),
            ("HardwareGuardian", HardwareGuardian),
            ("NoveltyAuditor", NoveltyAuditor),
        ],
    },
    "Phase 2 — Deep Dissection 🔬": {
        "enabled_key": "dissection",
        "agents": [
            ("SourceArchaeologist", SourceArchaeologist),
            ("MathTranslator", MathTranslator),
            ("ConflictDetector", ConflictDetector),
        ],
    },
    "Phase 3 — Adversarial Debate ⚔️": {
        "enabled_key": "debate",
        "agents": [
            ("Proposer", Proposer),
            ("Critic", Critic),
            ("Mediator", Mediator),
        ],
    },
    "Phase 4 — Quality & Publication 📝": {
        "enabled_key": "quality",
        "agents": [
            ("QualityInspector", QualityInspector),
            ("PublicationStrategist", PublicationStrategist),
            ("EditorInChief", EditorInChief),
        ],
    },
}


# ──────────────────────────────────────────────
# Default config generator
# ──────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "api_base": "https://models.github.ai/inference",
        "api_key": "${GITHUB_TOKEN}",
        "default_model": "openai/gpt-4.1",
        "temperature": 0.4,
        "max_tokens": 8192,
    },
    "pipeline": {
        "topic": (
            "Grafting Linear Attention Operators into Transformer-based LLMs "
            "for Efficient Long-Context Inference"
        ),
        "target_model_size": "8B-14B",
        "quantization": "INT8",
        "hardware_budget_gb": 24,
        "phases": {
            "intelligence": True,
            "dissection": True,
            "debate": True,
            "quality": True,
        },
    },
    "search": {
        "arxiv_max_results": 50,
        "github_stars_threshold": 100,
        "openreview_venues": [
            "ICLR.cc/2026/Conference",
            "NeurIPS.cc/2025/Conference",
        ],
    },
    "output": {
        "output_dir": "output/",
    },
}


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config; create default if not found."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[yellow]⚠ Config not found at {config_path}, generating default...[/yellow]")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, allow_unicode=True)
        console.print(f"[green]✓ Default config written to {path}[/green]")
        return DEFAULT_CONFIG.copy()

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_initial_context(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Construct the initial context dict from config + CLI overrides."""
    pipeline_cfg = config.get("pipeline", {})
    search_cfg = config.get("search", {})
    output_cfg = config.get("output", {})

    ctx: dict[str, Any] = {
        "topic": overrides.get("topic") or pipeline_cfg.get("topic", ""),
        "target_model_size": pipeline_cfg.get("target_model_size", "8B-14B"),
        "quantization": pipeline_cfg.get("quantization", "INT8"),
        "hardware_budget_gb": overrides.get("hardware_budget_gb")
        or pipeline_cfg.get("hardware_budget_gb", 24),
        "arxiv_max_results": search_cfg.get("arxiv_max_results", 50),
        "github_stars_threshold": search_cfg.get("github_stars_threshold", 100),
        "openreview_venues": search_cfg.get("openreview_venues", []),
        "output_dir": output_cfg.get("output_dir", "output/"),
    }
    return ctx


# ──────────────────────────────────────────────
# Core pipeline runner
# ──────────────────────────────────────────────


async def run_pipeline(
    config_path: str = "config/settings.yaml",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full RF-1.5 pipeline."""
    overrides = overrides or {}
    config = load_config(config_path)
    phase_flags = config.get("pipeline", {}).get("phases", {})

    # Initialize shared LLM client
    llm_config = LLMConfig.from_yaml(config_path)
    llm = LLMClient(llm_config)

    # Build initial context
    context = build_initial_context(config, overrides)

    console.print(
        Panel(
            f"[bold cyan]Research Factory RF-1.5[/bold cyan]\n"
            f"[white]Topic:[/white]  {context['topic']}\n"
            f"[white]Model:[/white]  {llm_config.default_model}\n"
            f"[white]VRAM:[/white]   {context['hardware_budget_gb']} GB\n"
            f"[white]Quant:[/white]  {context['quantization']}",
            title="🏭 RF-1.5 Initialized",
            border_style="green",
        )
    )

    total_start = time.time()
    agent_timings: list[tuple[str, float, bool]] = []

    for phase_name, phase_def in PHASES.items():
        enabled_key = phase_def["enabled_key"]
        if not phase_flags.get(enabled_key, True):
            console.print(f"\n[dim]⏭  Skipping {phase_name} (disabled)[/dim]")
            continue

        console.print(f"\n[bold yellow]{'═' * 60}[/bold yellow]")
        console.print(f"[bold yellow]▶ {phase_name}[/bold yellow]")
        console.print(f"[bold yellow]{'═' * 60}[/bold yellow]")

        for agent_name, agent_cls in phase_def["agents"]:
            agent = agent_cls(llm)
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]{agent_name}[/cyan] working..."),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("", total=None)
                t0 = time.time()
                success = True
                try:
                    context = await agent.run(context)
                except Exception as e:
                    success = False
                    console.print(f"  [red]✗ {agent_name} failed: {e}[/red]")
                    logger.exception("Agent %s failed", agent_name)
                    context[f"{agent_name}_error"] = str(e)
                elapsed = time.time() - t0
                agent_timings.append((agent_name, elapsed, success))
                if success:
                    console.print(
                        f"  [green]✓[/green] {agent_name} completed in [bold]{elapsed:.1f}s[/bold]"
                    )

    total_elapsed = time.time() - total_start

    # ── Final Summary ──
    console.print(f"\n[bold green]{'═' * 60}[/bold green]")
    console.print(
        f"[bold green]🏁 Pipeline Complete — {total_elapsed:.0f}s total[/bold green]"
    )
    console.print(f"[bold green]{'═' * 60}[/bold green]")

    _print_timing_table(agent_timings)
    _print_summary(context)

    # Save full context to disk
    output_dir = Path(context.get("output_dir", "output/"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rf_full_context.json"

    # Make context JSON-serializable
    serializable = _make_serializable(context)
    report_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\n[dim]Full context saved to {report_path}[/dim]")

    return context


# ──────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────


def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _print_timing_table(timings: list[tuple[str, float, bool]]) -> None:
    """Print agent execution timing table."""
    table = Table(title="⏱  Agent Execution Times", show_lines=False)
    table.add_column("Agent", style="cyan", width=25)
    table.add_column("Time (s)", justify="right", width=10)
    table.add_column("Status", justify="center", width=10)

    for name, elapsed, success in timings:
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        table.add_row(name, f"{elapsed:.1f}", status)

    console.print(table)


def _print_summary(context: dict[str, Any]) -> None:
    """Print a rich summary table of key results."""
    table = Table(title="📊 RF-1.5 Results Summary", show_lines=True)
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Key Result", style="white", min_width=40)

    # ── Phase 1 ──
    trend = context.get("trend_report", {})
    if isinstance(trend, dict) and not trend.get("_parse_error"):
        focus = trend.get("recommended_focus", "N/A")
        if isinstance(focus, dict):
            focus = focus.get("topic", str(focus))
        table.add_row("🔍 Trend Analyst", str(focus)[:80])
    else:
        table.add_row("🔍 Trend Analyst", "—")

    hw = context.get("hardware_audit", {})
    if isinstance(hw, dict) and not hw.get("_parse_error"):
        vram = hw.get("vram_breakdown", {})
        total_gb = vram.get("total_gb", "?") if isinstance(vram, dict) else "?"
        table.add_row(
            "🖥  HW Guardian",
            f"Total VRAM: {total_gb} GB | Fits: {hw.get('fits_budget', '?')} | Risk: {hw.get('risk_level', '?')}",
        )
    else:
        table.add_row("🖥  HW Guardian", "—")

    nov = context.get("novelty_audit", {})
    if isinstance(nov, dict) and not nov.get("_parse_error"):
        table.add_row(
            "🔎 Novelty Auditor",
            f"Score: {nov.get('novelty_score', '?')}/100 ({nov.get('novelty_level', '?')})",
        )
    else:
        table.add_row("🔎 Novelty Auditor", "—")

    # ── Phase 2 ──
    src = context.get("source_analysis", {})
    if isinstance(src, dict):
        arch = src.get("model_architecture", {})
        name = arch.get("name", "?") if isinstance(arch, dict) else "?"
        n_graft = len(src.get("grafting_points", []))
        table.add_row("🏛  Src Archaeologist", f"Model: {name} | Grafting points: {n_graft}")
    else:
        table.add_row("🏛  Src Archaeologist", "—")

    math_t = context.get("math_translation", {})
    if isinstance(math_t, dict):
        table.add_row("📐 Math Translator", math_t.get("operator_name", "—"))
    else:
        table.add_row("📐 Math Translator", "—")

    conflict = context.get("conflict_report", {})
    if isinstance(conflict, dict):
        table.add_row(
            "⚡ Conflict Detector",
            f"Risk: {conflict.get('overall_risk', '?')} | Score: {conflict.get('compatibility_score', '?')}/100",
        )
    else:
        table.add_row("⚡ Conflict Detector", "—")

    # ── Phase 3 ──
    crit = context.get("criticism", {})
    if isinstance(crit, dict):
        table.add_row(
            "👿 Critic Verdict",
            f"{crit.get('overall_verdict', '?')} (confidence: {crit.get('confidence', '?')}/5)",
        )
    else:
        table.add_row("👿 Critic Verdict", "—")

    plan = context.get("final_plan", {})
    if isinstance(plan, dict):
        table.add_row(
            "🤝 Mediator Decision",
            f"{plan.get('go_no_go_decision', '?')} — {plan.get('final_plan_title', '')[:60]}",
        )
    else:
        table.add_row("🤝 Mediator Decision", "—")

    # ── Phase 4 ──
    quality = context.get("quality_report", {})
    if isinstance(quality, dict):
        table.add_row(
            "✅ Quality Score",
            f"{quality.get('total_score', '?')}/100 (Grade: {quality.get('grade', '?')}) | Pass: {quality.get('pass_threshold_met', '?')}",
        )
    else:
        table.add_row("✅ Quality Score", "—")

    pub = context.get("publication_strategy", {})
    if isinstance(pub, dict):
        rec = pub.get("recommended_strategy", {})
        if isinstance(rec, dict):
            table.add_row(
                "🎯 Pub Strategy",
                f"Reach: {rec.get('reach_venue', '?')} | Safe: {rec.get('safe_venue', '?')} | Backup: {rec.get('backup_venue', '?')}",
            )
        else:
            table.add_row("🎯 Pub Strategy", str(rec)[:80])
    else:
        table.add_row("🎯 Pub Strategy", "—")

    table.add_row("📄 LaTeX Output", context.get("tex_path", "—"))
    table.add_row("📕 PDF Output", context.get("pdf_path", "(pdflatex not available or failed)"))

    console.print(table)


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────


def cli_entry() -> None:
    parser = argparse.ArgumentParser(
        description="🏭 Research Factory RF-1.5 — Multi-Agent Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m rf.main\n"
            "  python -m rf.main --topic 'Mamba-in-LLaMA grafting'\n"
            "  python -m rf.main --vram 48 --model openai/gpt-4.1\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to configuration YAML file (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Override the research topic",
    )
    parser.add_argument(
        "--vram",
        type=int,
        default=None,
        help="Override VRAM budget in GB",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override LLM model name (e.g., openai/gpt-4.1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("rf_debug.log", encoding="utf-8"),
        ],
    )

    overrides: dict[str, Any] = {}
    if args.topic:
        overrides["topic"] = args.topic
    if args.vram:
        overrides["hardware_budget_gb"] = args.vram

    # Model override: patch environment for LLMConfig
    if args.model:
        import os
        os.environ["RF_MODEL_OVERRIDE"] = args.model

    try:
        asyncio.run(run_pipeline(args.config, overrides))
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red bold]Pipeline failed: {e}[/red bold]")
        logger.exception("Pipeline fatal error")
        raise SystemExit(1)


if __name__ == "__main__":
    cli_entry()
