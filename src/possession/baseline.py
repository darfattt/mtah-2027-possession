"""Reference pipeline: nearest tracked player to the ball -> team -> majority vote.

Equivalent to upstream `starter/baseline.py` (spellF1 ≈ 0.2433 at window 50 on the full match):
distances in metres, a +/- `window` vote, and D wherever the window never sees the ball.
Everything beyond this file is where the score comes from.
"""
from __future__ import annotations

import numpy as np

from .config import ABSTAIN, DEAD, N_FRAMES, graded_mask
from .features import interpolate_ball, nearest_team
from .io import Tracking
from .smoothing import majority_vote


def predict(tr: Tracking, window: int = 50, ball_gap_fill: int = 0) -> np.ndarray:
    """(N,) labels. window is a half-width (+/- frames).

    ball_gap_fill>0 linearly bridges ball gaps up to that many frames first (0 = upstream).
    """
    ball = interpolate_ball(tr, ball_gap_fill) if ball_gap_fill > 0 else None
    raw = nearest_team(tr, ball)
    out = majority_vote(raw, window)
    out[out == ABSTAIN] = DEAD  # no ball anywhere in the window -> dead, as upstream
    n = len(out)
    if n < N_FRAMES:
        out = np.concatenate([out, np.full(N_FRAMES - n, ABSTAIN, dtype="<U1")])
    out = out[:N_FRAMES]
    out[~graded_mask(N_FRAMES)] = ABSTAIN  # free, and keeps the file honest
    return out
