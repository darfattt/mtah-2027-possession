"""Score every CSV in submissions/ with the OFFICIAL scorer against the public 5-minute GT.

The official script scans its own evaluation/results/ folder, so we copy our files in,
run it, and clean up. Output is its ranked board.

    python scripts/score.py            # all of submissions/*.csv
    python scripts/score.py a.csv b.csv
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
RESULTS = UP / "evaluation" / "results"


def main(argv: list[str]) -> int:
    if not (UP / "evaluation" / "run_evaluation.py").exists():
        print("upstream/ missing — run scripts/bootstrap_upstream.sh first", file=sys.stderr)
        return 1
    files = [Path(a) for a in argv] or sorted((ROOT / "submissions").glob("*.csv"))
    if not files:
        print("no submissions to score", file=sys.stderr)
        return 1
    copied = []
    for f in files:
        dst = RESULTS / f"ours_{f.name}"
        shutil.copy(f, dst)
        copied.append(dst)
    try:
        return subprocess.call([sys.executable, "evaluation/run_evaluation.py"], cwd=UP)
    finally:
        for c in copied:
            c.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
