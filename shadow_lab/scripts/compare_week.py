#!/usr/bin/env python3
"""CLI utility to generate weekly side-by-side comparison CSVs and report.
"""
import sys
from pathlib import Path

# Setup sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "shadow_lab" / "src"))

from shadow_identity.cli import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "compare":
        sys.argv.insert(1, "compare")
    elif len(sys.argv) == 1:
        sys.argv.append("compare")
    main()
