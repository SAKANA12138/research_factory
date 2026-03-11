#!/usr/bin/env bash
# ──────────────────────────────────────────
# RF-1.5 Quick Start Script
# ──────────────────────────────────────────
set -euo pipefail

echo "🏭 Research Factory RF-1.5 — Quick Start"
echo "========================================="

# Check Python version
python3 -c "
import sys
v = sys.version_info
assert v >= (3, 11), f'Python 3.11+ required, got {v.major}.{v.minor}'
print(f'✓ Python {v.major}.{v.minor}.{v.micro}')
"

# Check GITHUB_TOKEN
if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "❌ GITHUB_TOKEN not set. Please run:"
    echo "   export GITHUB_TOKEN='ghp_xxxxxxxxxxxxxxxx'"
    exit 1
fi
echo "✓ GITHUB_TOKEN is set"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -e ".[dev]" --quiet 2>/dev/null || pip install -e . --quiet

# Run the pipeline
echo ""
echo "🚀 Launching RF-1.5 pipeline..."
echo ""
python -m rf.main "$@"
