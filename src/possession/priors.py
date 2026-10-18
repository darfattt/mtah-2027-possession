"""Priors from Smart Tagging. These are *evidence*, not answers — tagging is automatic and noisy.

Useful facts (upstream DATA.md):
  - every SET PIECES instance is a restart: a dead-ball stretch ends at its start
  - every phase carries a Team tag except 36 N/A (all throw-ins among them)
  - intervals are phases, not possessions — never copy them straight to labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SMART_TAGGING_TEAM_AWAY, SMART_TAGGING_TEAM_HOME


def team_tag_to_label(tag: str) -> str:
    t = (tag or "").strip().lower()
    if t == SMART_TAGGING_TEAM_HOME:
        return "H"
    if t == SMART_TAGGING_TEAM_AWAY:
        return "A"
    return "X"


def restart_frames(tags: pd.DataFrame) -> np.ndarray:
    """Start frames of SET PIECES instances — candidate dead→alive boundaries."""
    sp = tags[tags["code"].str.upper() == "SET PIECES"]
    return sp["start_frame"].to_numpy().astype(int)


def team_prior(tags: pd.DataFrame, n_frames: int) -> np.ndarray:
    """(N, 2) soft counts of [H, A] phase coverage per frame. Overlapping phases accumulate."""
    prior = np.zeros((n_frames, 2), dtype=np.float32)
    if "team" not in tags.columns:
        return prior
    for _, r in tags.iterrows():
        lab = team_tag_to_label(r["team"])
        if lab == "X":
            continue
        a, b = max(0, r["start_frame"]), min(n_frames - 1, r["end_frame"])
        prior[a : b + 1, 0 if lab == "H" else 1] += 1.0
    return prior


def setpiece_type_at(tags: pd.DataFrame, frame: int) -> str | None:
    sp = tags[(tags["code"].str.upper() == "SET PIECES") & (tags["start_frame"] <= frame) & (tags["end_frame"] >= frame)]
    if sp.empty or "type" not in sp.columns:
        return None
    return str(sp.iloc[0]["type"])
