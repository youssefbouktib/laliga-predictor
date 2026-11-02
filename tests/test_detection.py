"""Tests for the value-detection upgrade: shot features, market and
Dixon-Coles inputs, the stacked models, line shopping, fair CLV, the
no-model control and bootstrap intervals."""
import argparse

import numpy as np
import pandas as pd
import pytest

import run
import synth
from src import backtest, data, features, model

OUT = ["h", "d", "a"]


# ---------------------------------------------------------------- data

def test_clean_adds_sharp_best_and_stats(history):
    for c in ["sharp_h", "best_h", "h_shots", "a_shots", "h_sot", "a_sot"]:
        assert c in history.columns and history[c].notna().all(), c
    # the best price is never worse than the paired bookmaker's price
    for o in OUT:
        assert (history[f"best_{o}"] >= history[f"odds_{o}"]).all()
    # sharp pre-match odds: Pinnacle while it exists, then the next source
    ps = history[history.book == "PS"]
    np.testing.assert_allclose(ps.sharp_h, ps.odds_h)


def test_best_price_below_book_is_replaced():
    raw = pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["B"], "Date": ["01/09/2024"],
                        "FTHG": [1], "FTAG": [0],
                        "B365H": [2.0], "B365D": [3.4], "B365A": [4.0],
                        "MaxH": [1.5], "MaxD": [3.6], "MaxA": [4.2]})   # broken MaxH
    out = data.clean(raw, 2024)
    assert out.best_h[0] == 2.0 and out.best_d[0] == 3.6


# ---------------------------------------------------------------- shot features

def test_shot_features_use_only_earlier_matches(history):
    base, _ = features.build_features(history)
    i = len(history) // 2
    changed = history.copy()
    changed.loc[i:, ["h_shots", "h_sot"]] = 99   # alter this match and everything after
    alt, _ = features.build_features(changed)
    pd.testing.assert_frame_equal(base.loc[:i, features.SHOT_FEATURES], alt.loc[:i, features.SHOT_FEATURES])
    assert not base.loc[i + 1:, features.SHOT_FEATURES].equals(alt.loc[i + 1:, features.SHOT_FEATURES])


def test_shot_features_are_rolling_means(history):
    df, _ = features.build_features(history)
    team = df.home.iloc[-1]
    before = df.iloc[:-1]
    played = before[(before.home == team) | (before.away == team)].tail(8)
    sot_for = np.where(played.home == team, played.h_sot, played.a_sot)
    assert df.home_sotf.iloc[-1] == pytest.approx(sot_for.mean())


def test_no_shot_data_means_no_shot_features(history):
    df, _ = features.build_features(history.assign(h_shots=np.nan, a_shots=np.nan,
                                                   h_sot=np.nan, a_sot=np.nan))
    assert not set(features.SHOT_FEATURES) & set(df.columns)
    assert "logistic_shots" not in model.available_models(df)


def test_promoted_team_shot_history_cleared(featured):
    df, _ = featured
    # T09 is promoted in synth: its first match uses the league defaults
    first = df[(df.home == "T09") | (df.away == "T09")].iloc[0]
    side = "home" if first.home == "T09" else "away"
    cfg = features.EloConfig()
    assert first[f"{side}_shf"] == cfg.shots_default
    assert first[f"{side}_sotf"] == cfg.sot_default


# ---------------------------------------------------------------- model inputs

def test_market_features_match_fair_probs(featured):
    df = model.add_market_features(featured[0])
    p = data.fair_probs(df[["sharp_h", "sharp_d", "sharp_a"]].to_numpy(float))
    np.testing.assert_allclose(df.mkt_lh, np.log(p[:, 0] / p[:, 1]))
    np.testing.assert_allclose(df.mkt_la, np.log(p[:, 2] / p[:, 1]))


@pytest.fixture(scope="module")
def stacked(featured):
    return model.add_dc_features(model.add_market_features(featured[0]))


def test_dc_features_missing_only_in_first_season(stacked):
    first = stacked.season.min()
    assert stacked.loc[stacked.season == first, "dc_lh"].isna().all()
    assert stacked.loc[stacked.season > first, model.DC_FEATURES].notna().all().all()


def test_dc_features_do_not_see_the_match(featured):
    """Changing a match's result (and later ones) must not change its own
    Dixon-Coles input, nor that of any earlier match."""
    df = featured[0]
    base = model.add_dc_features(df)
    season = sorted(df.season.unique())[2]
    i = df.index[df.season == season][len(df[df.season == season]) // 2]
    changed = df.copy()
    changed.loc[i:, "hg"] = changed.loc[i:, "hg"] + 3
    alt = model.add_dc_features(changed)
    # the same calendar week is predicted by one fit made before it starts
    week_start = df.date[i] - pd.Timedelta(days=df.date[i].dayofweek)
    safe = df.index[df.date < week_start + pd.Timedelta(days=7)]
    safe = safe[safe <= i]
    pd.testing.assert_frame_equal(base.loc[safe, model.DC_FEATURES], alt.loc[safe, model.DC_FEATURES])


def test_stacked_models_available_and_trainable(stacked):
    avail = model.available_models(stacked)
    for m in ["ensemble", "market", "market_plus"]:
        assert m in avail
    rows = model.train_rows(stacked, "market_plus")
    assert stacked.loc[rows, model.TABULAR_MODELS["market_plus"][1]].notna().all().all()
    assert not rows[stacked.season == stacked.season.min()].any()


@pytest.mark.parametrize("name", ["ensemble", "market", "market_plus"])
def test_stacked_walk_forward(stacked, name):
    pred = model.walk_forward(stacked, 2, name)
    np.testing.assert_allclose(pred[model.PROB_COLS].sum(axis=1), 1.0)
    comp = model.compare({name: pred})
    assert comp.uses_market[0] == (name in model.MARKET_MODELS)


def test_market_model_close_to_market(stacked):
    """A model of the market alone should land near the market's own
    probabilities (it can only recalibrate them)."""
    pred = model.walk_forward(stacked, 2, "market")
    p = data.fair_probs(pred[["sharp_h", "sharp_d", "sharp_a"]].to_numpy(float))
    # small synthetic league (a few hundred training games): allow some noise
    assert np.abs(pred[model.PROB_COLS].to_numpy() - p).mean() < 0.06


# ---------------------------------------------------------------- backtest

def _pred(**kw):
    base = dict(date=pd.to_datetime(["2024-01-01", "2024-01-02"]), season=[2023, 2023],
                result=["H", "A"], p_h=[0.6, 0.2], p_d=[0.2, 0.3], p_a=[0.2, 0.5],
                odds_h=[1.9, 4.0], odds_d=[3.5, 3.4], odds_a=[4.5, 1.9],
                best_h=[2.0, 4.2], best_d=[3.6, 3.5], best_a=[4.6, 2.1],
                close_h=[1.8, 4.1], close_d=[3.5, 3.4], close_a=[4.8, 1.95],
                bench_h=[1.85, 4.0], bench_d=[3.6, 3.5], bench_a=[4.9, 2.0])
    base.update(kw)
    return pd.DataFrame(base)


def test_find_value_best_price_and_fair_clv():
    pred = _pred()
    book = backtest.find_value(pred, 0.0, "book")
    best = backtest.find_value(pred, 0.0, "best")
    assert (best.bet_odds >= book.bet_odds).all()
    assert book.clv.notna().all() and best.clv.isna().all()   # like-for-like CLV only at the book
    implied = 1 / pred[["bench_h", "bench_d", "bench_a"]].to_numpy(float)
    fair = implied / implied.sum(axis=1, keepdims=True)
    # match 0 picks home at 2.0: fair CLV = 2.0 * fair P(H) - 1
    assert best.clv_fair.iloc[0] == pytest.approx(2.0 * fair[0, 0] - 1)
    assert (best.price == "best").all()


def test_find_value_best_falls_back_to_book_without_best_columns():
    pred = _pred().drop(columns=["best_h", "best_d", "best_a"])
    v = backtest.find_value(pred, 0.0, "best")
    assert v.bet_odds.iloc[0] == 1.9


def test_max_edge_skips_too_good_to_be_true():
    pred = _pred(p_h=[0.9, 0.2], p_d=[0.05, 0.3], p_a=[0.05, 0.5])   # EV 0.71 on match 0
    capped = backtest.find_value(pred, 0.05, "book", max_edge=0.15)
    assert 0 not in capped.index
    uncapped = backtest.find_value(pred, 0.05, "book", max_edge=None)
    assert 0 in uncapped.index


def test_bootstrap_roi_interval():
    rng = np.random.default_rng(3)
    profit = np.where(rng.random(4000) < 0.5, 1.0, -1.0)      # fair coin at even odds
    lo, hi = backtest.bootstrap_roi(profit, np.ones_like(profit))
    assert lo < 0 < hi and hi - lo < 0.08
    assert np.isnan(backtest.bootstrap_roi(np.array([1.0]), np.array([1.0]))[0])


def test_summary_has_ci_and_fair_clv():
    bets = backtest.settle(backtest.find_value(_pred(), 0.0, "best"))
    s = backtest.summary(bets, 0.0)
    assert s["price"] == "best"
    assert s["roi_ci_low"] <= s["roi"] <= s["roi_ci_high"]
    assert s["avg_clv_fair"] == pytest.approx(bets.clv_fair.mean())
    assert np.isnan(s["avg_clv"]) and np.isnan(s["pct_beat_close"])


def test_control_clv():
    pred = _pred()
    c = backtest.control_clv(pred)
    implied = 1 / pred[["bench_h", "bench_d", "bench_a"]].to_numpy(float)
    fair = implied / implied.sum(axis=1, keepdims=True)
    assert c["book"] == pytest.approx((pred[["odds_h", "odds_d", "odds_a"]].to_numpy() * fair - 1).mean())
    assert c["best"] > c["book"]


def test_by_season():
    bets = backtest.settle(backtest.find_value(_pred(season=[2022, 2023]), 0.0, "best"))
    out = backtest.by_season(bets)
    assert out.season.tolist() == ["2022/23", "2023/24"]
    assert out.bets.tolist() == [1, 1]
    assert backtest.by_season(bets.iloc[:0]).empty


def test_settle_paper_trades_fair_clv():
    hist = pd.DataFrame({"date": pd.to_datetime(["2026-10-04"]), "home": ["A"], "away": ["B"],
                         "result": ["H"], "book": ["B365"], "close_h": [1.9], "close_d": [3.5],
                         "close_a": [4.0], "bench_h": [1.95], "bench_d": [3.6], "bench_a": [4.2]})
    trades = pd.DataFrame({"date": ["2026-10-04"], "home": ["A"], "away": ["B"], "book": ["B365"],
                           "price": ["best"], "pick": ["h"], "bet_odds": [2.1]})
    out = backtest.settle_paper_trades(trades, hist)
    implied = 1 / np.array([1.95, 3.6, 4.2])
    assert out.clv_fair[0] == pytest.approx(2.1 * implied[0] / implied.sum() - 1)
    assert "bench_h" not in out.columns


# ---------------------------------------------------------------- run.py end to end

@pytest.fixture
def env(tmp_path, raw_copy, monkeypatch):
    monkeypatch.setattr(run, "RAW", raw_copy)
    monkeypatch.setattr(run, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(run, "DOCS_IMG", tmp_path / "docs" / "img")
    monkeypatch.setattr(run, "today", lambda: pd.Timestamp("2000-01-01"))
    return tmp_path


def test_evaluate_default_main_model_is_market_plus(env):
    run.cmd_evaluate(argparse.Namespace(models=None, test_seasons=2, model=None))
    bt = pd.read_csv(env / "reports" / "backtest_summary.csv")
    main_rows = bt[bt.model == model.DEFAULT_MODEL]
    assert len(main_rows) == 2 * 2 * len(run.EDGES)
    comp = pd.read_csv(env / "reports" / "model_comparison.csv")
    assert {"market", "market_plus", "ensemble", "dixon_coles"} <= set(comp.model)
    assert (env / "reports" / "clv_by_season.png").exists()


def test_predict_market_plus_logs_best_price(env):
    fx = synth.fixtures_frame([synth.CURRENT_PLAYED_ROUNDS + 1], edge=0.08)
    synth.write_csv(fx, run.RAW / "fixtures.csv", bom=True)
    run.cmd_predict(argparse.Namespace(model="market_plus", min_edge=0.02, price="best",
                                       max_edge=run.MAX_EDGE))
    up = pd.read_csv(env / "reports" / "upcoming.csv")
    assert (up.model == "market_plus").all()
    np.testing.assert_allclose(up[["p_h", "p_d", "p_a"]].sum(axis=1), 1.0)
    log = env / "reports" / run.PAPER_LOG
    if log.exists():
        trades = pd.read_csv(log)
        assert (trades.price == "best").all() and (trades.ev <= run.MAX_EDGE).all()


def test_old_paper_log_gets_price_column(env):
    rep = env / "reports"
    rep.mkdir()
    pd.DataFrame([{"logged_at": "2023-01-01T09:00", "model": "dixon_coles", "date": "2023-01-02",
                   "home": "X1", "away": "X2", "book": "B365", "pick": "h", "p_model": 0.5,
                   "bet_odds": 2.4, "ev": 0.2}]).to_csv(rep / run.PAPER_LOG, index=False)
    synth.write_csv(synth.fixtures_frame([synth.CURRENT_PLAYED_ROUNDS + 1]), run.RAW / "fixtures.csv")
    run.cmd_predict(argparse.Namespace(model="dixon_coles", min_edge=0.05, price="book", max_edge=None))
    log = pd.read_csv(rep / run.PAPER_LOG)
    assert "price" in log.columns and log.price.iloc[0] == "book"
    assert len(log) > 1
