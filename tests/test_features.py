import numpy as np
import pandas as pd
import pytest

from src import features
from src.features import FEATURES, XG_FEATURES, EloConfig

ALL_FEATS = ["elo_home", "elo_away"] + FEATURES


def _mini(rows, with_xg=False):
    """rows: (date, season, home, away, hg, ag[, hxg, axg])"""
    cols = ["date", "season", "home", "away", "hg", "ag"] + (["hxg", "axg"] if with_xg else [])
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_goal_multiplier():
    assert features._goal_mult(0) == 1.0
    assert features._goal_mult(1) == 1.0
    assert features._goal_mult(-1) == 1.0
    assert features._goal_mult(2) == 1.5
    assert features._goal_mult(-2) == 1.5
    assert features._goal_mult(3) == pytest.approx(14 / 8)
    assert features._goal_mult(5) == pytest.approx(16 / 8)


@pytest.mark.parametrize("hg,ag", [(2, 0), (1, 1), (0, 3), (4, 1)])
def test_elo_update_zero_sum_and_expected_score(hg, ag):
    cfg = EloConfig()
    st = features.new_state(cfg)
    st.ratings.update({"A": 1550.0, "B": 1480.0})
    features.update(st, "A", "B", hg, ag, cfg)
    exp_h = 1 / (1 + 10 ** ((1480 - 1550 - cfg.home_adv) / 400))
    score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    delta = cfg.k * features._goal_mult(hg - ag) * (score - exp_h)
    assert st.ratings["A"] == pytest.approx(1550 + delta)
    assert st.ratings["B"] == pytest.approx(1480 - delta)
    assert st.ratings["A"] + st.ratings["B"] == pytest.approx(3030.0)
    # form: (goals for, goals against, points)
    hp = 3 if hg > ag else 1 if hg == ag else 0
    ap = 3 if ag > hg else 1 if hg == ag else 0
    assert st.form["A"][-1] == (hg, ag, hp)
    assert st.form["B"][-1] == (ag, hg, ap)


def test_first_match_defaults_and_elo_diff_includes_home_adv():
    df = _mini([("2020-08-20", 2020, "A", "B", 1, 0)])
    out, _ = features.build_features(df)
    r = out.iloc[0]
    assert r.elo_home == 1500 and r.elo_away == 1500
    assert r.elo_diff == pytest.approx(EloConfig().home_adv)
    assert (r.home_gf5, r.home_ga5, r.home_pts5) == (1.3, 1.3, 1.4)


def test_no_leakage_features_unchanged_when_future_results_change(history):
    base, _ = features.build_features(history)
    rng = np.random.default_rng(5)
    for i in [0, 37, 150, len(history) - 5]:
        alt = history.copy()
        alt.loc[i:, "hg"] = rng.integers(0, 7, len(alt) - i)
        alt.loc[i:, "ag"] = rng.integers(0, 7, len(alt) - i)
        out, _ = features.build_features(alt)
        pd.testing.assert_frame_equal(out.loc[:i, ALL_FEATS], base.loc[:i, ALL_FEATS])
        # and the change does matter later on (sanity of the test itself)
        if i < len(history) - 20:
            assert not out.loc[i + 1:, ALL_FEATS].equals(base.loc[i + 1:, ALL_FEATS])


def test_no_leakage_xg_features(history):
    rng = np.random.default_rng(3)
    h = history.copy()
    h["hxg"] = rng.uniform(0.2, 3, len(h))
    h["axg"] = rng.uniform(0.2, 3, len(h))
    base, _ = features.build_features(h)
    i = 200
    alt = h.copy()
    alt.loc[i:, ["hxg", "axg"]] = 9.0
    out, _ = features.build_features(alt)
    pd.testing.assert_frame_equal(out.loc[:i, XG_FEATURES], base.loc[:i, XG_FEATURES])


def test_season_regression():
    cfg = EloConfig()
    rows = [("2020-08-20", 2020, "A", "B", 3, 0), ("2020-08-27", 2020, "B", "A", 0, 2),
            ("2021-08-20", 2021, "A", "B", 1, 1)]
    out, _ = features.build_features(_mini(rows))
    _, st1 = features.build_features(_mini(rows[:2]))
    ra, rb = st1.ratings["A"], st1.ratings["B"]
    assert ra > 1500 > rb
    r = out.iloc[2]
    assert r.elo_home == pytest.approx(ra + cfg.season_regress * (1500 - ra))
    assert r.elo_away == pytest.approx(rb + cfg.season_regress * (1500 - rb))


def test_promoted_and_returning_team_reset():
    rows = [
        ("2019-08-20", 2019, "A", "R", 0, 4),   # R strong in 2019...
        ("2019-08-27", 2019, "R", "A", 5, 0),
        ("2020-08-20", 2020, "A", "N", 1, 0),   # ...absent in 2020; N is new in 2020
        ("2020-08-27", 2020, "N", "A", 1, 1),
        ("2021-08-20", 2021, "A", "R", 1, 1),   # R returns in 2021
        ("2021-08-20", 2021, "N", "B", 2, 2),   # B never seen before
    ]
    out, _ = features.build_features(_mini(rows))
    new_n = out.iloc[2]
    assert new_n.elo_away == 1420
    assert (new_n.away_gf5, new_n.away_ga5, new_n.away_pts5) == (1.3, 1.3, 1.4)
    ret = out.iloc[4]
    assert ret.elo_away == 1420, "returning team must be reset to the promoted rating"
    assert (ret.away_gf5, ret.away_ga5, ret.away_pts5) == (1.3, 1.3, 1.4), "stale form must be cleared"
    assert out.iloc[5].elo_away == 1420
    # a team that stayed up keeps its (regressed) rating and its form
    assert out.iloc[4].elo_home != 1420
    assert out.iloc[5].home_pts5 != 1.4


def test_promoted_reset_on_synthetic_league(featured):
    df, _ = featured
    for season, team in [(2018, "T09"), (2019, "T08"), (2020, "T10")]:
        first = df[(df.season == season) & ((df.home == team) | (df.away == team))].iloc[0]
        side = "home" if first.home == team else "away"
        assert first[f"elo_{side}"] == 1420
        assert first[f"{side}_pts5"] == 1.4


def test_xg_features_use_only_prior_matches():
    rows = [
        ("2020-08-20", 2020, "A", "B", 1, 0, 2.0, 0.5),
        ("2020-08-27", 2020, "B", "A", 1, 1, 1.0, 3.0),
        ("2020-09-03", 2020, "A", "B", 0, 0, np.nan, np.nan),   # no xG for this one
        ("2020-09-10", 2020, "B", "A", 0, 0, 0.4, 0.6),
    ]
    out, _ = features.build_features(_mini(rows, with_xg=True))
    cfg = EloConfig()
    assert out.loc[0, XG_FEATURES].tolist() == [cfg.xg_default] * 4
    # match 1: B home has history from match 0 only (xgf 0.5, xga 2.0)
    assert out.loc[1, "home_xgf"] == pytest.approx(0.5)
    assert out.loc[1, "home_xga"] == pytest.approx(2.0)
    assert out.loc[1, "away_xgf"] == pytest.approx(2.0)
    # match 2: A home = mean of (2.0, 0.5) and (3.0, 1.0)
    assert out.loc[2, "home_xgf"] == pytest.approx(2.5)
    assert out.loc[2, "home_xga"] == pytest.approx(0.75)
    # match 3: the NaN-xG match did not change the history
    assert out.loc[3, "away_xgf"] == pytest.approx(2.5)


def test_xg_features_absent_without_xg_columns(featured):
    df, _ = featured
    assert not set(XG_FEATURES) & set(df.columns)
    assert set(FEATURES) <= set(df.columns)
    assert df[FEATURES].notna().all(axis=None)


def test_build_features_keeps_row_order_and_index(history, featured):
    df, _ = featured
    assert len(df) == len(history)
    pd.testing.assert_frame_equal(df[history.columns], history)


def test_predict_snapshots_matches_build_features_state(history):
    """Predicting the next match from the final state == feature row of that match."""
    df, _ = features.build_features(history)
    last = len(history) - 1
    _, st = features.build_features(history.iloc[:last])
    snap = features.predict_snapshots(st, history.iloc[[last]])
    for c in ALL_FEATS:
        assert snap.iloc[0][c] == pytest.approx(df.iloc[last][c])


def test_predict_snapshots_new_season_applies_regression_and_promotion():
    """Fixtures of a new season (before any of its games is played) must get the
    same treatment build_features gives a new season: ratings regressed toward the
    mean and a team absent last season reset to the promoted rating with no form.
    """
    rows = [
        ("2019-08-20", 2019, "A", "R", 0, 4),
        ("2019-08-27", 2019, "R", "A", 5, 0),
        ("2020-08-20", 2020, "A", "B", 3, 0),
        ("2020-08-27", 2020, "B", "A", 0, 3),
    ]
    hist = _mini(rows)
    _, st = features.build_features(hist)
    fixtures = pd.DataFrame({"date": pd.to_datetime(["2021-08-20"]), "season": [None],
                             "home": ["A"], "away": ["R"]})
    snap = features.predict_snapshots(st, fixtures).iloc[0]
    # what build_features would produce if the game were in the data
    full, _ = features.build_features(_mini(rows + [("2021-08-20", 2021, "A", "R", 0, 0)]))
    expected = full.iloc[-1]
    assert snap.elo_away == 1420, "R was not in the 2020 league: promoted rating expected"
    assert snap.away_pts5 == 1.4, "R's 2019 form must not carry over"
    assert snap.elo_home == pytest.approx(expected.elo_home), "new-season regression missing"


def test_build_features_empty_frame():
    out, st = features.build_features(_mini([]))
    assert out.empty and set(FEATURES) <= set(out.columns)


def test_predict_snapshots_does_not_change_state():
    _, st = features.build_features(_mini([("2019-08-20", 2019, "A", "B", 1, 0)]))
    before = dict(st.ratings)
    fx = pd.DataFrame({"date": pd.to_datetime(["2020-08-20"]), "home": ["A"], "away": ["C"]})
    features.predict_snapshots(st, fx)
    assert st.ratings == before and st.season == 2019
