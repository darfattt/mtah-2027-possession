import numpy as np
import pandas as pd
import pytest

from possession import config
from possession.features import ball_present, interpolate_ball, nearest_team
from possession.io import Tracking, tracking_from_df
from possession.smoothing import (
    fill_abstain,
    majority_vote,
    remove_short_runs,
    runs,
    viterbi_labels,
)
from possession.submit import SubmissionError, summary, validate_labels, write_submission


def test_graded_mask_counts():
    m = config.graded_mask()
    assert m.sum() == 158_857  # per upstream DATA.md
    assert config.half_of_frame(11_185) == 1
    assert config.half_of_frame(100_000) == 0
    assert config.half_of_frame(190_197) == 2


def test_nearest_team_respects_ball_gaps(synthetic_tracking):
    lab = nearest_team(synthetic_tracking)
    assert (lab[:100] == "H").all()
    assert (lab[100:125] == "X").all()
    assert (lab[300:400] == "A").all()


def test_interpolate_ball_bridges_only_short_gaps(synthetic_tracking):
    b = interpolate_ball(synthetic_tracking, max_gap=30)
    present = ~np.isnan(b).any(axis=1)
    assert present[100:125].all()  # 25-frame gap bridged
    assert not present[400:460].any()  # 60-frame gap left open
    assert ball_present(synthetic_tracking)[:100].all()


def test_majority_vote_removes_flicker():
    lab = np.array(list("HHHHHAHHHHH"), dtype="<U1")
    out = majority_vote(lab, window=5)
    assert (out == "H").all()


def test_majority_vote_matches_raw_when_window_0():
    lab = np.array(list("HHAADD"), dtype="<U1")
    assert (majority_vote(lab, 0) == lab).all()


def test_majority_vote_window_is_half_width():
    # upstream convention: window=3 means +/- 3 frames, so one H reaches 3 frames each side
    lab = np.array(list("XXXHXXXX"), dtype="<U1")
    assert "".join(majority_vote(lab, 3)) == "HHHHHHHX"


def test_nearest_team_measures_in_metres():
    # normalised units put A closer (0.06 vs 0.08); in metres H is (5.4 m vs 6.3 m)
    xy = np.array([[[0.50, 0.58], [0.56, 0.50]]])
    tr = Tracking(ball=np.array([[0.5, 0.5]]), xy=xy, team=np.array(["H", "A"]),
                  speed=np.zeros((1, 2)))
    assert nearest_team(tr)[0] == "H"


def test_tracking_from_df_keeps_team_aligned_when_slot_missing():
    df = pd.DataFrame({
        "frame_id": [1, 2], "ball_x": [0.5, 0.5], "ball_y": [0.5, 0.5],
        "p1_x": [0.1, 0.1], "p1_y": [0.1, 0.1],
        "p3_x": [0.9, 0.9], "p3_y": [0.9, 0.9],
    })  # p2 is in the metadata but has no columns
    tr = tracking_from_df(df, [("p1", "H"), ("p2", "H"), ("p3", "A")])
    assert tr.xy.shape == (2, 2, 2)
    assert list(tr.team) == ["H", "A"]


def test_fill_abstain_mirrors_scorer():
    lab = np.array(list("XXHHXXAAX"), dtype="<U1")
    assert "".join(fill_abstain(lab)) == "HHHHHHAAA"


def test_runs_and_short_run_removal():
    lab = np.array(list("HHHHAHHHHDDDD"), dtype="<U1")
    assert runs(lab)[1] == (4, 4, "A")
    out = remove_short_runs(lab, min_len=3)
    assert "".join(out) == "HHHHHHHHHDDDD"


def test_viterbi_prefers_fewer_switches():
    # hard votes: mostly H with a one-frame A blip
    n = 20
    E = np.full((n, 3), -5.0)
    E[:, 0] = 0.0
    E[10, 0], E[10, 1] = -5.0, 0.0
    assert (viterbi_labels(E, switch_cost=0.0) == np.array(list("H" * 10 + "A" + "H" * 9))).all()
    assert (viterbi_labels(E, switch_cost=6.0) == "H").all()


def test_submission_roundtrip(tmp_path, synthetic_tracking):
    labels = np.full(config.N_FRAMES, "X", dtype="<U1")
    labels[config.HALVES[0][1] : config.HALVES[0][1] + 500] = "H"
    p = write_submission(labels, tmp_path / "s.csv")
    assert p.exists()
    s = summary(labels)
    assert s["possessions"] == 1
    assert s["mean_duration_s"] == pytest.approx(20.0)


def test_validate_rejects_broadcast_length():
    with pytest.raises(SubmissionError, match="broadcast"):
        validate_labels(np.full(164_488, "H", dtype="<U1"))


def test_validate_rejects_bad_label():
    lab = np.full(config.N_FRAMES, "H", dtype="<U1")
    lab[0] = "Z"
    with pytest.raises(SubmissionError):
        validate_labels(lab)


def test_baseline_end_to_end(synthetic_tracking, monkeypatch):
    from possession import baseline

    monkeypatch.setattr(config, "N_FRAMES", 600)
    monkeypatch.setattr(baseline, "N_FRAMES", 600)
    monkeypatch.setattr(baseline, "graded_mask", lambda n: np.ones(n, dtype=bool))
    out = baseline.predict(synthetic_tracking, window=25, ball_gap_fill=30)
    assert len(out) == 600
    assert (out[:300] == "H").all()
    assert (out[300:400] == "A").all()
    # 60-frame gap 400-459 is not bridged; frames whose +/-25 window never sees the ball are D
    assert (out[426:434] == "D").all()
    assert "X" not in out
