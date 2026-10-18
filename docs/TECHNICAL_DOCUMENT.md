# Technical Document — Team <NAME> — MTAH 2027 Phase 1

> Required deliverable. Keep it short and concrete: what we built, why, and how to reproduce it.

## 1. Summary

One paragraph: approach in plain words, final public-window scores, how many possessions we predict.

## 2. Inputs used

| asset | used? | how |
|---|---|---|
| Tactical ATD (players + ball) | yes | … |
| Smart Tagging | | … |
| Homography | | … |
| Tactical video | | … |
| Broadcast feed | | … |
| External data / models | | … |

## 3. Pipeline

1. Loading and frame alignment (ATD `frame_id - 1`, tagging `round(s*25)`).
2. Ball-gap handling.
3. Dead-ball detection.
4. Team-in-possession assignment.
5. Transition / smoothing model.
6. Abstention policy (`X`).

Diagram or pseudo-code welcome.

## 4. Our definition of "control"

CHALLENGE.md leaves "controlled reception vs rebound" open. State the criterion we chose and
where it disagrees with the 5-minute public GT.

## 5. Experiments

| id | change | public spellF1 | public macroF1 | #poss | notes |
|---|---|---|---|---|---|
| b50 | upstream baseline, window 50 | 0.3256 | 0.6207 | 26 | reference |
| … | | | | | |

Remember: public window = 17 possessions; differences < ~0.06 are noise. Don't tune on it.

## 6. Reproduction

```bash
pip install -e .
# data in ./data (or MTAH_DATA_DIR)
python scripts/run_baseline.py --out submissions/FINAL_<name>.csv   # or the final command
python upstream/evaluation/validate_submission.py submissions/FINAL_<name>.csv
```

Deterministic: seeds fixed at … ; no network calls at inference.

## 7. Limitations and what we'd do next

…

## 8. Team

Names, roles, contact.
