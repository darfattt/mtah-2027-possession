"""Loaders. Every loader returns something indexed by *video frame* (0 … N_FRAMES-1).

Frame alignment (from upstream DATA.md):
  ATD           frame_id 1..N        -> video_frame = frame_id - 1
  Smart Tagging seconds from start   -> video_frame = round(seconds * 25)
  homography    frame 0..N-1         -> already a video frame
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FPS, N_FRAMES, asset_paths


# --------------------------------------------------------------------------- ATD
@dataclass
class Tracking:
    """Wide per-frame tracking.

    ball: (N, 2) normalised 0-1, NaN where missing
    xy:   (N, S, 2) per track slot, NaN where missing
    team: (S,) array of 'H' / 'A' / '?' per slot (from metadata)
    speed:(N, S) m/s, NaN where missing
    """

    ball: np.ndarray
    xy: np.ndarray
    team: np.ndarray
    speed: np.ndarray

    @property
    def n_frames(self) -> int:
        return self.ball.shape[0]

    @property
    def n_slots(self) -> int:
        return self.xy.shape[1]


def load_tracking(source: str = "tactical") -> Tracking:
    """Read Metrica FIFA/EPTS ATD with kloppy and reshape into dense arrays.

    kloppy's to_df() yields frame_id, ball_x, ball_y, then <slot>_x, <slot>_y, <slot>_s per slot.
    Team per slot comes from dataset.metadata.teams.
    """
    from kloppy import metrica  # imported lazily: tests run without kloppy's optional deps

    p = asset_paths(source)
    ds = metrica.load_tracking_epts(meta_data=str(p["atd_meta"]), raw_data=str(p["atd_raw"]))
    slots = [
        (player.player_id, "H" if team.ground.name.lower() == "home" else "A")
        for team in ds.metadata.teams
        for player in team.players
    ]
    return tracking_from_df(ds.to_df(), slots, min_frames=N_FRAMES if source == "tactical" else 0)


def tracking_from_df(df: pd.DataFrame, slots: list[tuple[str, str]], min_frames: int = 0) -> Tracking:
    """kloppy to_df() + [(slot_id, 'H'|'A'), ...] in metadata order -> dense Tracking.

    Slots without columns in df are dropped together with their team tag, so team stays
    aligned with xy.
    """
    frame_idx = df["frame_id"].to_numpy().astype(int) - 1
    n = max(int(frame_idx.max()) + 1, min_frames)

    slots = [(sid, tag) for sid, tag in slots if f"{sid}_x" in df.columns]
    slot_ids = [sid for sid, _ in slots]
    S = len(slot_ids)

    ball = np.full((n, 2), np.nan)
    xy = np.full((n, S, 2), np.nan)
    speed = np.full((n, S), np.nan)

    ball[frame_idx, 0] = df["ball_x"].to_numpy()
    ball[frame_idx, 1] = df["ball_y"].to_numpy()
    for j, sid in enumerate(slot_ids):
        xy[frame_idx, j, 0] = df[f"{sid}_x"].to_numpy()
        xy[frame_idx, j, 1] = df[f"{sid}_y"].to_numpy()
        if f"{sid}_s" in df.columns:
            speed[frame_idx, j] = df[f"{sid}_s"].to_numpy()

    return Tracking(ball=ball, xy=xy, team=np.array([tag for _, tag in slots]), speed=speed)


# ----------------------------------------------------------------- Smart Tagging
def load_smart_tagging(path: Path | None = None) -> pd.DataFrame:
    """SportsCode XML -> DataFrame with start_frame, end_frame, code, and one column per tag group.

    Pure stdlib so it works without kloppy. Tag groups are lower-cased column names;
    values are kept as-is (title case upstream) — compare case-insensitively.
    """
    path = path or asset_paths("tactical")["smart_tagging"]
    root = ET.parse(path).getroot()
    rows = []
    for inst in root.iter("instance"):
        start = float(inst.findtext("start"))
        end = float(inst.findtext("end"))
        row = {
            "id": inst.findtext("ID"),
            "start_s": start,
            "end_s": end,
            "start_frame": round(start * FPS),
            "end_frame": round(end * FPS),
            "code": (inst.findtext("code") or "").strip(),
        }
        for lab in inst.iter("label"):
            group = (lab.findtext("group") or "tag").strip().lower()
            text = (lab.findtext("text") or "").strip()
            row[group] = text
        rows.append(row)
    df = pd.DataFrame(rows)
    if "team" in df.columns:
        df["team"] = df["team"].fillna("N/A")
    return df.sort_values("start_frame").reset_index(drop=True)


# ------------------------------------------------------------------- homography
@dataclass
class Homography:
    H: np.ndarray  # (N, 3, 3), NaN where invalid
    reprojection_error: np.ndarray  # (N,)
    valid: np.ndarray  # (N,) bool


def load_homography(source: str = "tactical") -> Homography:
    df = pd.read_csv(asset_paths(source)["homography"])
    n = int(df["frame"].max()) + 1
    H = np.full((n, 3, 3), np.nan)
    err = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    f = df["frame"].to_numpy().astype(int)
    cols = [f"h{i}{j}" for i in range(3) for j in range(3)]
    H[f] = df[cols].to_numpy().reshape(-1, 3, 3)
    err[f] = df["reprojection_error"].to_numpy()
    valid[f] = df["valid"].to_numpy().astype(bool)
    return Homography(H=H, reprojection_error=err, valid=valid)


def project(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Normalised pitch (0-1) -> pixel using one 3x3 matrix."""
    v = H @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


# ------------------------------------------------------------------ submissions
def read_labels(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return df["label"].to_numpy().astype("<U1")
