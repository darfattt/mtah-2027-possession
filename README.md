# MTAH 2027 — Phase 1: Ball Possession

Team repository for **More Than A Hack 2027, Initial Challenge** (Metrica Sports × Barça Innovation Hub × MWCapital).

Task: for every frame of FC Barcelona vs Athletic Club (199,319 frames @ 25 fps), output one label —
`H` (home in possession), `A` (away), `D` (dead ball) or `X` (abstain). Ranked on **spellF1**
(possession-level temporal IoU F1), tie-break on **macroF1**. Baseline to beat: **spellF1 0.2433**.

Official starter + scorer: https://github.com/metrica-sports/more-than-a-hack-2027

## Key dates

| | |
|---|---|
| Registration closes | 10 Nov 2026 |
| Data released | 20 Oct 2026 |
| Submission deadline | **17 Nov 2026** |
| Results | before 26 Nov 2026 |

Teams must have 3–5 members.

## Layout

```
mtah-2027-possession/
├─ src/possession/          our pipeline (installable package)
│  ├─ config.py             paths, constants (fps, graded halves, labels)
│  ├─ io.py                 load ATD / Smart Tagging / homography → DataFrames
│  ├─ features.py           per-frame features (ball gaps, nearest player, team centroids…)
│  ├─ baseline.py           nearest-player-to-ball + majority-vote window
│  ├─ smoothing.py          segment-aware smoothers (min-duration, HMM/Viterbi)
│  ├─ priors.py             Smart Tagging priors (restarts → D boundaries, Team tags)
│  ├─ submit.py             build + validate the frame,label CSV
│  └─ cli.py                `possession` command
├─ scripts/
│  ├─ bootstrap_upstream.sh clone official repo into upstream/ and symlink data/
│  ├─ run_baseline.py       end-to-end: data → submissions/baseline.csv
│  └─ score.py              score every CSV in submissions/ with the official scorer
├─ notebooks/               exploration (keep outputs stripped)
├─ tests/                   unit tests on synthetic data (no match data needed)
├─ docs/
│  ├─ TECHNICAL_DOCUMENT.md required deliverable — fill as we go
│  └─ DECISIONS.md          running log of what we tried and the score
├─ data/                    gitignored — unpack the match bundle here
├─ submissions/             gitignored CSVs; keep the final one tracked
└─ upstream/                gitignored clone of the official repo (scorer lives there)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
bash scripts/bootstrap_upstream.sh                   # clones official repo into upstream/
# unpack the match bundle into data/ (tactical/ and broadcast/ subfolders)
pytest                                               # should pass with no data
```

Set `MTAH_DATA_DIR` if the data lives elsewhere; both this repo and the official starter honour it.

## Daily loop

```bash
python scripts/run_baseline.py --window 50 --out submissions/baseline_w50.csv   # +/-50 frames = upstream baseline_smooth50
python scripts/score.py                      # official scorer on the public 5-minute GT
python upstream/evaluation/validate_submission.py submissions/baseline_w50.csv
```

**Never tune on the public 5 minutes** — it is 17 possessions; one miss moves spellF1 by ~6 points.
Use it to check alignment and file validity only (see EVALUATION.md upstream).

## Rules we must respect

- Labels must come from an algorithm; no hand-labelling of the submission.
- Repository must be public and must **regenerate the exact submitted CSV**.
- Match data and any frame-level derivative must never be committed.
- Any model/data we use must be inspectable by the organisers and described in `docs/TECHNICAL_DOCUMENT.md`.

## Approach roadmap

1. **Baseline** — nearest tracked player to ball (in metres) → team, majority vote over ±`window`
   frames, `D` where the window never sees the ball. `--window 50` reproduces upstream ≈0.24.
   `--window` is a half-width, the same convention as upstream `starter/baseline.py`.
2. **Ball gap handling** — interpolate / Kalman short gaps; abstain (`X`) inside long gaps, never at transitions.
3. **Dead-ball detection** — Smart Tagging `SET PIECES` + ball out-of-bounds + player-cluster stillness.
4. **Transition model** — possession as a hidden state with a switch cost (Viterbi) instead of a per-frame vote.
5. **Collective features** — team centroids, velocity toward ball, compactness — no player IDs exist.
6. (Stretch) broadcast feed / own ball detector if ball gaps dominate error.
