"""Constants and paths. Mirrors the official `games.py` so both repos agree."""
from __future__ import annotations

import os
from pathlib import Path

GAME = "BAR-ATH"
FPS = 25
N_FRAMES = 199_319
PITCH_LENGTH_M = 105.3  # measured from corrected tracking (official games.py)
PITCH_WIDTH_M = 68.0

HOME, AWAY, DEAD, ABSTAIN = "H", "A", "D", "X"
LABELS = (HOME, AWAY, DEAD, ABSTAIN)
SCORED_LABELS = (HOME, AWAY, DEAD)

# Graded halves in video frames, inclusive. Everything else is never scored.
HALVES = [(1, 11_185, 90_651), (2, 110_808, 190_197)]

# Smart Tagging time base: seconds from first frame of the tactical video.
SMART_TAGGING_TEAM_HOME = "barcelona"
SMART_TAGGING_TEAM_AWAY = "atletic club"  # sic — upstream spelling

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO_ROOT / "upstream"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"


def data_dir() -> Path:
    """`MTAH_DATA_DIR` if set, else <repo>/data. Same env var the official repo uses."""
    return Path(os.environ.get("MTAH_DATA_DIR", REPO_ROOT / "data"))


def asset_paths(source: str = "tactical") -> dict[str, Path]:
    """Paths to the match bundle. source in {"tactical", "broadcast"}."""
    if source not in ("tactical", "broadcast"):
        raise ValueError(f"unknown source {source!r}")
    base = data_dir() / source
    stem = f"{GAME}_{source}"
    paths = {
        "video": base / f"{stem}.mp4",
        "atd_meta": base / f"{stem}_atd.xml",
        "atd_raw": base / f"{stem}_atd.txt",
        "homography": base / f"{stem}_homography.csv",
    }
    if source == "tactical":
        paths["smart_tagging"] = base / f"{stem}_smart_tagging.xml"
    return paths


def graded_mask(n_frames: int = N_FRAMES):
    """Boolean array, True on frames that the private ground truth scores."""
    import numpy as np

    m = np.zeros(n_frames, dtype=bool)
    for _, a, b in HALVES:
        m[a : b + 1] = True
    return m


def half_of_frame(frame: int) -> int:
    """1, 2, or 0 if the frame is outside both halves."""
    for h, a, b in HALVES:
        if a <= frame <= b:
            return h
    return 0
