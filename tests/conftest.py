"""Synthetic tracking fixture — no match data needed."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from possession.io import Tracking  # noqa: E402


@pytest.fixture
def synthetic_tracking():
    """600 frames (24 s). Home holds the ball for 300 frames, then away. 4 slots, 2 per team.

    Ball sits on top of slot 0 (H) then slot 2 (A). Slot positions are static.
    Ball is missing on frames 100-124 (short gap) and 400-459 (long gap).
    """
    rng = np.random.default_rng(0)
    n, S = 600, 4
    xy = np.empty((n, S, 2))
    xy[:, 0] = [0.30, 0.50]
    xy[:, 1] = [0.20, 0.40]
    xy[:, 2] = [0.70, 0.50]
    xy[:, 3] = [0.80, 0.60]
    xy += rng.normal(0, 0.002, xy.shape)
    ball = np.empty((n, 2))
    ball[:300] = [0.31, 0.50]
    ball[300:] = [0.71, 0.50]
    ball += rng.normal(0, 0.002, ball.shape)
    ball[100:125] = np.nan
    ball[400:460] = np.nan
    team = np.array(["H", "H", "A", "A"])
    speed = np.zeros((n, S))
    return Tracking(ball=ball, xy=xy, team=team, speed=speed)
