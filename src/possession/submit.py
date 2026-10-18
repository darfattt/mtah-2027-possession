"""Write and sanity-check a `frame,label` submission. Mirrors upstream validate_submission.py rules.

Always ALSO run the official validator before submitting:
    python upstream/evaluation/validate_submission.py <file>.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS, N_FRAMES
from .smoothing import runs


class SubmissionError(ValueError):
    pass


def validate_labels(labels: np.ndarray, n_frames: int = N_FRAMES) -> None:
    if len(labels) != n_frames:
        raise SubmissionError(
            f"expected {n_frames} rows, got {len(labels)}"
            + (" — ~164k rows means you indexed the broadcast video" if 160_000 < len(labels) < 170_000 else "")
        )
    bad = set(np.unique(labels)) - set(LABELS)
    if bad:
        raise SubmissionError(f"unknown labels {sorted(bad)}; allowed {LABELS}")


def write_submission(labels: np.ndarray, path: Path) -> Path:
    validate_labels(labels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)
    return path


def summary(labels: np.ndarray, fps: int = 25) -> dict:
    """Possession count / mean duration — compare against the public GT's 17 @ 14.2 s (5 min only)."""
    poss = [(s, e) for s, e, lab in runs(labels) if lab in ("H", "A")]
    durs = [(e - s + 1) / fps for s, e in poss]
    counts = {lab: int((labels == lab).sum()) for lab in LABELS}
    return {
        "possessions": len(poss),
        "mean_duration_s": float(np.mean(durs)) if durs else 0.0,
        "label_counts": counts,
    }
