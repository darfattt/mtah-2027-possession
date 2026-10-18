"""`possession` command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import baseline
from .config import SUBMISSIONS_DIR
from .io import load_tracking, read_labels
from .submit import summary, validate_labels, write_submission


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="possession")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="nearest-player + majority vote")
    b.add_argument("--window", type=int, default=50, help="vote half-width, +/- frames (upstream)")
    b.add_argument("--ball-gap-fill", type=int, default=0, help="bridge ball gaps up to N frames")
    b.add_argument("--out", type=Path, default=SUBMISSIONS_DIR / "baseline.csv")

    c = sub.add_parser("check", help="validate + summarise a submission CSV")
    c.add_argument("csv", type=Path)

    args = ap.parse_args(argv)

    if args.cmd == "baseline":
        tr = load_tracking("tactical")
        labels = baseline.predict(tr, window=args.window, ball_gap_fill=args.ball_gap_fill)
        out = write_submission(labels, args.out)
        print(f"wrote {out}")
        print(json.dumps(summary(labels), indent=2))
        return 0

    if args.cmd == "check":
        labels = read_labels(args.csv)
        validate_labels(labels)
        print(json.dumps(summary(labels), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
