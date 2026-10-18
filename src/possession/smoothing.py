"""Segment-aware smoothers. All operate on (N,) label arrays of 'H'/'A'/'D'/'X'.

The ranking metric scores possessions as whole objects, so the enemy is *fragmentation*:
a vote that flips for a few frames splits one true possession into three predicted ones.
"""
from __future__ import annotations

import numpy as np

from .config import ABSTAIN, SCORED_LABELS

_IDX = {lab: i for i, lab in enumerate(SCORED_LABELS)}


def majority_vote(labels: np.ndarray, window: int) -> np.ndarray:
    """Centered majority vote over `window` frames, ignoring 'X'. Reproduces the upstream baseline."""
    if window <= 1:
        return labels.copy()
    n = len(labels)
    onehot = np.zeros((n, len(SCORED_LABELS)), dtype=np.int32)
    for lab, i in _IDX.items():
        onehot[labels == lab, i] = 1
    cs = np.vstack([np.zeros((1, onehot.shape[1]), dtype=np.int32), np.cumsum(onehot, axis=0)])
    half = window // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    counts = cs[hi] - cs[lo]
    out = np.array(SCORED_LABELS, dtype="<U1")[counts.argmax(axis=1)]
    out[counts.sum(axis=1) == 0] = ABSTAIN
    return out


def fill_abstain(labels: np.ndarray) -> np.ndarray:
    """Forward-fill 'X' with the previous non-X label, back-fill the head. Mirrors the scorer."""
    out = labels.copy()
    last = None
    for i in range(len(out)):
        if out[i] == ABSTAIN:
            if last is not None:
                out[i] = last
        else:
            last = out[i]
    # back-fill leading X
    first = next((l for l in out if l != ABSTAIN), None)
    if first is not None:
        for i in range(len(out)):
            if out[i] != ABSTAIN:
                break
            out[i] = first
    return out


def runs(labels: np.ndarray) -> list[tuple[int, int, str]]:
    """Maximal runs as (start, end_inclusive, label)."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change - 1, [len(labels) - 1]])
    return [(int(s), int(e), str(labels[s])) for s, e in zip(starts, ends)]


def remove_short_runs(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Merge any run shorter than `min_len` into its longer neighbour. Repeats until stable."""
    out = labels.copy()
    changed = True
    while changed:
        changed = False
        rs = runs(out)
        for k, (s, e, lab) in enumerate(rs):
            if e - s + 1 >= min_len:
                continue
            left = rs[k - 1] if k > 0 else None
            right = rs[k + 1] if k + 1 < len(rs) else None
            if left is None and right is None:
                break
            cand = max([r for r in (left, right) if r is not None], key=lambda r: r[1] - r[0])
            out[s : e + 1] = cand[2]
            changed = True
            break
    return out


def viterbi(emission_logp: np.ndarray, switch_cost: float, stay_cost: float = 0.0) -> np.ndarray:
    """Most-likely state path with a constant penalty for changing state.

    emission_logp: (N, K) log-probability of each state per frame (use 0 / -inf for hard votes,
                   or log of a soft score). NaN rows are treated as uninformative (all zeros).
    switch_cost:   penalty (>0) subtracted when the state changes between frames.
    Returns (N,) int state indices.
    """
    E = np.nan_to_num(emission_logp, nan=0.0, neginf=-1e9)
    n, k = E.shape
    score = E[0].copy()
    back = np.zeros((n, k), dtype=np.int32)
    trans = -switch_cost * (1 - np.eye(k)) - stay_cost * np.eye(k)
    for t in range(1, n):
        cand = score[:, None] + trans  # (from, to)
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + E[t]
    path = np.empty(n, dtype=np.int32)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_labels(emission_logp: np.ndarray, switch_cost: float) -> np.ndarray:
    """Convenience: Viterbi over SCORED_LABELS order (H, A, D) -> label array."""
    return np.array(SCORED_LABELS, dtype="<U1")[viterbi(emission_logp, switch_cost)]
