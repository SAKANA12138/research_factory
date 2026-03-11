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
import sys
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


def load_config(config_path: str) -> dict[str, Any]:
    """Load and return the full YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        console.print("[yellow]Creating default config...[/yellow]")
        path.parent.mkdir(parents=True, exist_ok=True)
        default_config = {
            "llm": {
                "api_base": "https://models.github.ai/inference",
                "api_key": "${GITHUB_TOKEN}",
                "default_model": "openai/gpt-4.1",
                "temperature": 0.4,
                "max_tokens": 8192,
            },
            "pipeline": {
                "topic": "Grafting Linear Attention Operators into
