"""End to end: tactical ATD -> baseline labels -> submissions/<name>.csv

    python scripts/run_baseline.py --window 50 --out submissions/baseline_w50.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from possession.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["baseline", *sys.argv[1:]]))
