"""Value-betting backtest: flat or fractional-Kelly stakes, with closing line value."""
from __future__ import annotations

import numpy as np
import pandas as pd

OUT = ["h", "d", "a"]


def find_value(pred: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    """One bet per match at most: the outcome with the highest expected value,
    if that EV is above `min_edge`. EV = model_prob * odds - 1."""
    p = pred[[f"p_{o}" for o in OUT]].to_numpy(dtype=float)
    odds = pred[[f"odds_{o}" for o in OUT]].to_numpy(dtype=float)
    ev = p * odds - 1
    ev = np.where(np.isnan(ev), -np.inf, ev)
    pick = ev.argmax(axis=1)
    best_ev = ev[np.arange(len(ev)), pick]
    keep = best_ev > min_edge
    bets = pred.loc[keep].copy()
    pick = pick[keep]
    idx = np.arange(len(bets))
    bets["pick"] = np.array(OUT)[pick]
    bets["ev"] = best_ev[keep]
    bets["p_model"] = p[keep][idx, pick]
    bets["bet_odds"] = odds[keep][idx, pick]
    close = bets[[f"close_{o}" for o in OUT]].to_numpy(dtype=float)[idx, pick]
    bets["close_odds"] = close
    bets["clv"] = bets["bet_odds"] / close - 1
    return bets


def kelly_fraction(p: np.ndarray, odds: np.ndarray, fraction: float = 0.25,
                   cap: float = 0.05) -> np.ndarray:
    """Fraction of bankroll to stake. Full Kelly is (p*odds - 1) / (odds - 1);
    we bet a fraction of it (model probabilities are uncertain) and cap it."""
    p, odds = np.asarray(p, dtype=float), np.asarray(odds, dtype=float)
    ok = odds > 1  # odds of 1 or less can't win anything: stake nothing
    with np.errstate(divide="ignore", invalid="ignore"):
        full = np.where(ok, (p * odds - 1) / (odds - 1), 0.0)
    return np.clip(full * fraction, 0, cap)


def settle(bets: pd.DataFrame, stake: float = 1.0) -> pd.DataFrame:
    """Flat stakes: `stake` units per bet."""
    bets = bets.sort_values("date", kind="stable").copy()
    won = bets["pick"].str.upper() == bets["result"]
    bets["stake"] = stake
    bets["profit"] = np.where(won, stake * (bets["bet_odds"] - 1), -stake)
    bets["cum_profit"] = bets["profit"].cumsum()
    bets["bankroll"] = np.nan
    return bets


def settle_kelly(bets: pd.DataFrame, bankroll: float = 100.0, fraction: float = 0.25,
                 cap: float = 0.05) -> pd.DataFrame:
    """Fractional Kelly. Stakes are sized from the bankroll at the start of each
    match day, so bets on the same day don't depend on each other's results."""
    bets = bets.sort_values("date", kind="stable").copy()
    won = (bets["pick"].str.upper() == bets["result"]).to_numpy()
    f = kelly_fraction(bets.p_model.to_numpy(), bets.bet_odds.to_numpy(), fraction, cap)
    stakes, profits, banks = np.zeros(len(bets)), np.zeros(len(bets)), np.zeros(len(bets))
    bank = bankroll
    dates = bets.date.to_numpy()
    i = 0
    while i < len(bets):
        j = i
        while j < len(bets) and dates[j] == dates[i]:
            j += 1
        day_bank = bank
        for k in range(i, j):
            stakes[k] = f[k] * day_bank
            profits[k] = stakes[k] * (bets.bet_odds.iat[k] - 1) if won[k] else -stakes[k]
            bank += profits[k]
            banks[k] = bank
        i = j
    bets["stake"] = stakes
    bets["profit"] = profits
    bets["cum_profit"] = profits.cumsum()
    bets["bankroll"] = banks
    return bets


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough fall, as a fraction of the peak."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def summary(bets: pd.DataFrame, edge: float, staking: str = "flat",
            start_bankroll: float = 100.0) -> dict:
    if bets.empty:
        return {"staking": staking, "min_edge": edge, "bets": 0}
    turnover = bets.stake.sum()
    out = {
        "staking": staking,
        "min_edge": edge,
        "bets": len(bets),
        "hit_rate": (bets.profit > 0).mean(),
        "avg_odds": bets.bet_odds.mean(),
        "staked": turnover,
        "profit_units": bets.profit.sum(),
        "roi": bets.profit.sum() / turnover if turnover else np.nan,
        "avg_clv": bets.clv.mean(),
        "pct_beat_close": (bets.clv.dropna() > 0).mean(),
    }
    if staking == "kelly":
        equity = np.concatenate([[start_bankroll], bets.bankroll.to_numpy()])
        out["final_bankroll"] = equity[-1]
        out["max_drawdown"] = max_drawdown(equity)
    else:
        equity = np.concatenate([[0.0], bets.cum_profit.to_numpy()])
        out["max_drawdown_units"] = float(np.max(np.maximum.accumulate(equity) - equity))
    return out


# ---------------------------------------------------------------- paper trades

def _season(date) -> int:
    """La Liga seasons start in August: Aug 2024 - May 2025 is season 2024."""
    return date.year if date.month >= 7 else date.year - 1


def settle_paper_trades(trades: pd.DataFrame, history: pd.DataFrame,
                        days_before: int = 3) -> pd.DataFrame:
    """Attach results and closing odds to logged paper trades.

    A trade matches the game with the same teams in the same season (each
    pairing is played once per season), played no more than `days_before`
    days before the fixture date it was logged with, nearest first. This
    settles postponed games however late they are played, and can never pick
    up last season's meeting. Closing odds are only used
    when the history row priced the game at the same bookmaker as the trade,
    so CLV stays like-for-like.
    """
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"])
    h = history[["date", "home", "away", "result", "book",
                 "close_h", "close_d", "close_a"]].rename(columns={"date": "played", "book": "hist_book"})
    m = t.reset_index().merge(h, on=["home", "away"], how="left")
    same_season = m.played.map(_season, na_action="ignore") == m.date.map(_season)
    m = m[m.played.isna() | (same_season & ((m.date - m.played).dt.days <= days_before))]
    m = (m.assign(_gap=(m.played - m.date).abs())
           .sort_values(["index", "_gap"]).drop_duplicates("index"))
    m = t.reset_index().merge(m[["index", "played", "result", "hist_book",
                                 "close_h", "close_d", "close_a"]], on="index", how="left")

    pick_i = m.pick.map({o: i for i, o in enumerate(OUT)}).to_numpy()
    closes = m[["close_h", "close_d", "close_a"]].to_numpy(dtype=float)
    close = closes[np.arange(len(m)), pick_i]
    same_book = m.hist_book.astype(object).eq(m.book).to_numpy()
    m["close_odds"] = np.where(same_book, close, np.nan)
    m["clv"] = m.bet_odds / m.close_odds - 1
    settled = m.result.notna()
    won = m.pick.str.upper() == m.result
    m["status"] = np.where(settled, np.where(won, "won", "lost"), "open")
    m["profit"] = np.where(settled, np.where(won, m.bet_odds - 1, -1.0), np.nan)
    return m.drop(columns=["index", "close_h", "close_d", "close_a", "hist_book"])
