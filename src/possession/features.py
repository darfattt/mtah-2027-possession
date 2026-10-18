"""Per-frame features built from Tracking. Everything is vectorised over frames.

Nothing here knows about labels. Keep it that way so features are reusable by any model.
"""
from __future__ import annotations

import numpy as np

from .io import Tracking


def ball_present(tr: Tracking) -> np.ndarray:
    """(N,) bool — ball observed on this frame."""
    return ~np.isnan(tr.ball).any(axis=1)


def interpolate_ball(tr: Tracking, max_gap: int = 25) -> np.ndarray:
    """Linear-fill ball gaps up to `max_gap` frames; longer gaps stay NaN.

    Returns a new (N, 2) array. Short occlusions are safe to bridge; long ones are not.
    """
    ball = tr.ball.copy()
    present = ~np.isnan(ball).any(axis=1)
    idx = np.arange(len(ball))
    if present.sum() < 2:
        return ball
    for d in range(2):
        filled = np.interp(idx, idx[present], ball[present, d])
        ball[:, d] = np.where(present, ball[:, d], filled)
    # re-blank gaps longer than max_gap
    gap_id = np.cumsum(present)  # constant inside a gap
    for g in np.unique(gap_id[~present]):
        run = np.where((gap_id == g) & ~present)[0]
        if len(run) > max_gap:
            ball[run] = np.nan
    return ball


def nearest_slot(tr: Tracking, ball: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """For each frame: index of nearest tracked slot to the ball and its distance (normalised units).

    Returns (slot (N,) int, -1 where undefined ; dist (N,) float, NaN where undefined).
    """
    ball = tr.ball if ball is None else ball
    d = tr.xy - ball[:, None, :]  # (N, S, 2)
    dist = np.sqrt((d**2).sum(axis=2))  # (N, S), NaN where slot or ball missing
    all_nan = np.isnan(dist).all(axis=1)
    dist_safe = np.where(np.isnan(dist), np.inf, dist)
    slot = dist_safe.argmin(axis=1)
    best = dist_safe[np.arange(len(slot)), slot]
    slot = np.where(all_nan, -1, slot)
    best = np.where(all_nan, np.nan, best)
    return slot, best


def nearest_team(tr: Tracking, ball: np.ndarray | None = None) -> np.ndarray:
    """(N,) array of 'H' / 'A' / 'X' — team of the nearest slot, 'X' when undefined."""
    slot, _ = nearest_slot(tr, ball)
    out = np.full(len(slot), "X", dtype="<U1")
    ok = slot >= 0
    out[ok] = tr.team[slot[ok]]
    return out


def team_centroids(tr: Tracking) -> dict[str, np.ndarray]:
    """{'H': (N,2), 'A': (N,2)} mean position of visible slots per team, NaN if none."""
    out = {}
    for tag in ("H", "A"):
        m = tr.team == tag
        with np.errstate(invalid="ignore"):
            out[tag] = np.nanmean(tr.xy[:, m, :], axis=1)
    return out


def n_visible(tr: Tracking) -> np.ndarray:
    """(N,) number of tracked slots with a position on each frame."""
    return (~np.isnan(tr.xy[:, :, 0])).sum(axis=1)


def ball_speed(ball: np.ndarray, fps: int = 25, pitch=(105.3, 68.0)) -> np.ndarray:
    """(N,) ball speed in m/s from consecutive frames, NaN where either frame missing."""
    xy_m = ball * np.array(pitch)
    v = np.full(len(ball), np.nan)
    d = np.diff(xy_m, axis=0)
    v[1:] = np.sqrt((d**2).sum(axis=1)) * fps
    return v
