import numpy as np
import pandas as pd
import pytest

from src import backtest


def _pred(rows):
    """rows: (date, p_h, p_d, p_a, odds_h, odds_d, odds_a, close_h, close_d, close_a, result)"""
    cols = ["date", "p_h", "p_d", "p_a", "odds_h", "odds_d", "odds_a",
            "close_h", "close_d", "close_a", "result"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df["home"], df["away"] = "A", "B"
    return df


def test_find_value_picks_max_ev_and_respects_min_edge():
    pred = _pred([
        # EVs: 0.1, -0.1, 0.2 -> pick away
        ("2024-01-01", 0.5, 0.3, 0.2, 2.2, 3.0, 6.0, 2.0, 3.0, 5.0, "A"),
        # best EV 0.04 -> below min_edge 0.05
        ("2024-01-02", 0.5, 0.3, 0.2, 2.08, 3.0, 4.0, 2.0, 3.0, 4.0, "H"),
        # all EV negative
        ("2024-01-03", 0.4, 0.3, 0.3, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, "D"),
        # NaN odds on the best-looking outcome are ignored; draw EV 0.2
        ("2024-01-04", 0.5, 0.4, 0.1, np.nan, 3.0, np.nan, 2.0, 3.2, 9.0, "D"),
        # all odds NaN
        ("2024-01-05", 0.5, 0.4, 0.1, np.nan, np.nan, np.nan, 2.0, 3.2, 9.0, "D"),
    ])
    bets = backtest.find_value(pred, 0.05)
    assert bets.index.tolist() == [0, 3]
    assert bets.pick.tolist() == ["a", "d"]
    np.testing.assert_allclose(bets.ev, [0.2, 0.2])
    np.testing.assert_allclose(bets.p_model, [0.2, 0.4])
    np.testing.assert_allclose(bets.bet_odds, [6.0, 3.0])
    np.testing.assert_allclose(bets.close_odds, [5.0, 3.2])
    np.testing.assert_allclose(bets.clv, [6.0 / 5.0 - 1, 3.0 / 3.2 - 1])
    assert len(backtest.find_value(pred, 0.0)) == 3   # row 1 now qualifies (0.04 > 0)


def test_find_value_nan_close_gives_nan_clv_and_empty_input():
    pred = _pred([("2024-01-01", 0.6, 0.2, 0.2, 2.0, 3.0, 6.0, np.nan, np.nan, np.nan, "H")])
    bets = backtest.find_value(pred, 0.05)
    assert len(bets) == 1 and np.isnan(bets.clv.iloc[0])
    empty = backtest.find_value(pred.iloc[:0], 0.05)
    assert empty.empty


def test_settle_flat():
    pred = _pred([
        ("2024-01-02", 0.5, 0.3, 0.2, 2.2, 3.0, 6.0, 2.0, 3.0, 5.0, "A"),   # bet a, wins 5
        ("2024-01-01", 0.6, 0.3, 0.1, 2.0, 3.0, 6.0, 2.0, 3.0, 5.0, "D"),   # bet h, loses
    ])
    bets = backtest.settle(backtest.find_value(pred, 0.05))
    assert bets.date.is_monotonic_increasing
    assert bets.profit.tolist() == [-1.0, 5.0]
    assert bets.cum_profit.tolist() == [-1.0, 4.0]
    assert (bets.stake == 1.0).all()


def test_kelly_fraction_formula_and_cap():
    p = np.array([0.5, 0.5, 0.3, 0.9])
    o = np.array([2.2, 2.0, 3.0, 2.0])
    f = backtest.kelly_fraction(p, o, fraction=0.25, cap=0.05)
    full = (p * o - 1) / (o - 1)
    np.testing.assert_allclose(f[:3], np.clip(full[:3] * 0.25, 0, 0.05))
    assert f[0] == pytest.approx(0.25 * 0.1 / 1.2)
    assert f[1] == 0.0 and f[2] == 0.0      # no edge / negative edge -> no stake
    assert f[3] == 0.05                     # capped
    assert backtest.kelly_fraction(np.array([0.5]), np.array([2.2]), 1.0, 1.0)[0] == pytest.approx(0.1 / 1.2)


def test_settle_kelly_same_day_sizing():
    bets = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        "pick": ["h", "h", "a"], "result": ["H", "A", "A"],
        "p_model": [0.6, 0.6, 0.6], "bet_odds": [2.0, 2.0, 2.0],
    })
    out = backtest.settle_kelly(bets, bankroll=100, fraction=0.25, cap=1.0)
    f = 0.25 * (0.6 * 2 - 1) / 1      # 0.05
    # both day-1 bets sized from the same 100
    np.testing.assert_allclose(out.stake[:2], [100 * f, 100 * f])
    np.testing.assert_allclose(out.profit[:2], [5.0, -5.0])
    np.testing.assert_allclose(out.bankroll[:2], [105.0, 100.0])
    # day 2 sized from the bankroll after day 1
    assert out.stake.iloc[2] == pytest.approx(100 * f)
    assert out.bankroll.iloc[2] == pytest.approx(105.0)
    np.testing.assert_allclose(out.cum_profit, np.cumsum(out.profit))


def test_max_drawdown():
    assert backtest.max_drawdown(np.array([])) == 0.0
    assert backtest.max_drawdown(np.array([100, 120, 90, 130, 65])) == pytest.approx(0.5)
    assert backtest.max_drawdown(np.array([100, 110, 120])) == 0.0


def test_summary_empty_and_full():
    s = backtest.summary(pd.DataFrame(), 0.05)
    assert s == {"staking": "flat", "min_edge": 0.05, "bets": 0}
    pred = _pred([
        ("2024-01-01", 0.6, 0.3, 0.1, 2.0, 3.0, 6.0, 1.8, 3.0, 5.0, "D"),
        ("2024-01-02", 0.5, 0.3, 0.2, 2.2, 3.0, 6.0, 2.0, 3.0, 6.5, "A"),
    ])
    value = backtest.find_value(pred, 0.05)
    flat = backtest.summary(backtest.settle(value), 0.05)
    assert flat["bets"] == 2 and flat["profit_units"] == pytest.approx(4.0)
    assert flat["roi"] == pytest.approx(2.0)
    assert flat["pct_beat_close"] == 0.5
    assert flat["max_drawdown_units"] == pytest.approx(1.0)
    kelly = backtest.summary(backtest.settle_kelly(value), 0.05, "kelly")
    assert kelly["final_bankroll"] < 100 + 6 and "max_drawdown" in kelly


def test_summary_of_empty_find_value_result():
    pred = _pred([("2024-01-01", 0.3, 0.3, 0.4, 2.0, 3.0, 2.0, 2.0, 3.0, 2.0, "H")])
    value = backtest.find_value(pred, 0.05)
    assert value.empty
    assert backtest.summary(backtest.settle(value), 0.05)["bets"] == 0
    assert backtest.summary(backtest.settle_kelly(value), 0.05, "kelly")["bets"] == 0


# ---------------------------------------------------------------- paper trades

def _history():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-10-04", "2026-10-25", "2026-10-26", "2025-10-05"]),
        "home": ["A", "C", "E", "A"], "away": ["B", "D", "F", "B"],
        "result": ["H", "A", "D", "A"],
        "book": ["B365", "Avg", "B365", "PS"],
        "close_h": [1.9, 2.5, 2.0, 1.5], "close_d": [3.5, 3.2, 3.1, 4.0], "close_a": [4.0, 2.8, 3.9, 6.0],
    })


def _trades():
    return pd.DataFrame({
        "logged_at": ["2026-09-30T09:00"] * 4,
        "model": ["dixon_coles"] * 4,
        "date": ["2026-10-01", "2026-10-25", "2026-11-08", "2026-10-26"],
        "home": ["A", "C", "G", "E"], "away": ["B", "D", "H", "F"],
        "book": ["B365", "B365", "B365", "B365"],
        "pick": ["h", "a", "d", "d"],
        "p_model": [0.6, 0.4, 0.3, 0.3], "bet_odds": [2.0, 3.0, 3.4, 3.3], "ev": [0.2, 0.2, 0.02, 0.1],
    })


def test_settle_paper_trades():
    out = backtest.settle_paper_trades(_trades(), _history())
    assert len(out) == 4
    # 0: rescheduled 3 days later, matched (not to last season's A v B)
    assert out.status[0] == "won" and out.profit[0] == pytest.approx(1.0)
    assert out.played[0] == pd.Timestamp("2026-10-04")
    assert out.close_odds[0] == pytest.approx(1.9)
    assert out.clv[0] == pytest.approx(2.0 / 1.9 - 1)
    # 1: different bookmaker in history -> no CLV, but still settled
    assert out.status[1] == "won" and out.profit[1] == pytest.approx(2.0)
    assert np.isnan(out.close_odds[1]) and np.isnan(out.clv[1])
    # 2: not played yet -> open
    assert out.status[2] == "open" and np.isnan(out.profit[2])
    # 3: draw
    assert out.status[3] == "won" and out.clv[3] == pytest.approx(3.3 / 3.1 - 1)
    for c in ["logged_at", "model", "date", "home", "away", "book", "pick", "bet_odds"]:
        assert c in out.columns
    assert not {"index", "close_h", "hist_book"} & set(out.columns)


def test_settle_paper_trades_long_postponement_is_settled():
    # logged for 20 Sep, played on 4 Oct: still the same season's A v B
    t = _trades().iloc[[0]].assign(date="2026-09-20")
    out = backtest.settle_paper_trades(t, _history())
    assert out.status.tolist() == ["won"]
    assert out.played[0] == pd.Timestamp("2026-10-04")


def test_settle_paper_trades_game_played_before_trade_date_stays_open():
    # A v B was played on 4 Oct; a trade logged for 20 Oct can't be that game
    t = _trades().iloc[[0]].assign(date="2026-10-20")
    out = backtest.settle_paper_trades(t, _history())
    assert out.status.tolist() == ["open"]


def test_settle_paper_trades_never_uses_last_season():
    # only last season's A v B exists in history -> the trade stays open
    h = _history().iloc[[3]]
    out = backtest.settle_paper_trades(_trades().iloc[[0]], h)
    assert out.status.tolist() == ["open"]


def test_settle_paper_trades_picks_nearest_game():
    h = _history()
    h = pd.concat([h, h.iloc[[0]].assign(date=pd.Timestamp("2026-09-29"), result="A")], ignore_index=True)
    out = backtest.settle_paper_trades(_trades().iloc[[0]], h)
    assert out.played[0] == pd.Timestamp("2026-09-29") and out.status[0] == "lost"


def test_kelly_fraction_zero_for_odds_at_most_one():
    f = backtest.kelly_fraction(np.array([0.9, 0.9, 0.5]), np.array([1.0, 0.5, 2.5]))
    assert f[0] == 0 and f[1] == 0 and f[2] > 0
