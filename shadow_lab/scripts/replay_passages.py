#!/usr/bin/env python3
"""CLI utility to replay passage fixtures through the shadow identity evaluator.
"""
import sys
from pathlib import Path

# Setup sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "shadow_lab" / "src"))

from shadow_identity.cli import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "replay":
        sys.argv.insert(1, "replay")
    elif len(sys.argv) == 1:
        sys.argv.append("replay")
    main()
