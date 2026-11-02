"""Value-betting backtest: flat or fractional-Kelly stakes, with closing line value.

Two prices can be bet:
* "book": the paired bookmaker (Pinnacle, Bet365 after Jan 2026), whose own
  closing price gives a like-for-like CLV.
* "best": the best price across all bookmakers (line shopping). This is an
  optimistic upper bound: the "best" column includes stale prices and
  bookmakers that limit winning customers.

For both, `clv_fair` = bet_odds x fair closing probability - 1 is the
expected value of the bet judged by the sharp closing line (Pinnacle, else
Betfair Exchange). It is the professional yardstick: a bettor whose bets have
positive clv_fair on average is beating the market's final opinion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUT = ["h", "d", "a"]


PRICE_COLS = {"book": "odds", "best": "best"}


def find_value(pred: pd.DataFrame, min_edge: float, price: str = "book",
               max_edge: float | None = None) -> pd.DataFrame:
    """One bet per match at most: the outcome with the highest expected value,
    if that EV is above `min_edge`. EV = model_prob * odds - 1.

    `max_edge` skips bets whose EV looks too good to be true: when a model
    disagrees hugely with the market, the model is usually the one that is
    wrong (or the price is a stale outlier)."""
    p = pred[[f"p_{o}" for o in OUT]].to_numpy(dtype=float)
    prefix = PRICE_COLS[price]
    cols = [f"{prefix}_{o}" for o in OUT]
    if not set(cols).issubset(pred.columns):  # e.g. fixtures without a "best" price
        cols = [f"odds_{o}" for o in OUT]
    odds = pred[cols].to_numpy(dtype=float)
    ev = p * odds - 1
    ev = np.where(np.isnan(ev), -np.inf, ev)
    pick = ev.argmax(axis=1)
    best_ev = ev[np.arange(len(ev)), pick]
    keep = best_ev > min_edge
    if max_edge is not None:
        keep &= best_ev <= max_edge
    bets = pred.loc[keep].copy()
    pick = pick[keep]
    idx = np.arange(len(bets))
    bets["pick"] = np.array(OUT)[pick]
    bets["ev"] = best_ev[keep]
    bets["p_model"] = p[keep][idx, pick]
    bets["bet_odds"] = odds[keep][idx, pick]
    bets["price"] = price
    close = bets[[f"close_{o}" for o in OUT]].to_numpy(dtype=float)[idx, pick]
    bets["close_odds"] = close
    # like-for-like CLV only exists when we bet at the paired bookmaker
    bets["clv"] = bets["bet_odds"] / close - 1 if price == "book" else np.nan
    bench_cols = [f"bench_{o}" for o in OUT]
    if set(bench_cols).issubset(bets.columns) and len(bets):
        implied = 1 / bets[bench_cols].to_numpy(dtype=float)
        fair = implied / implied.sum(axis=1, keepdims=True)
        bets["clv_fair"] = bets["bet_odds"] * fair[idx, pick] - 1
    else:
        bets["clv_fair"] = np.nan
    return bets


def bootstrap_roi(profit: np.ndarray, stake: np.ndarray, n: int = 2000,
                  seed: int = 0) -> tuple[float, float]:
    """95% interval for ROI by resampling bets with replacement.

    If the interval contains 0, the backtest cannot tell the strategy apart
    from a break-even one: any profit or loss may be luck."""
    profit, stake = np.asarray(profit, dtype=float), np.asarray(stake, dtype=float)
    if len(profit) < 2 or stake.sum() <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(profit), size=(n, len(profit)))
    roi = profit[idx].sum(axis=1) / stake[idx].sum(axis=1)
    return float(np.quantile(roi, 0.025)), float(np.quantile(roi, 0.975))


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


def control_clv(pred: pd.DataFrame) -> dict:
    """Fair CLV of betting EVERY outcome of every match, with no model.

    This is the bar a model's bets must clear: with "best" prices it is
    close to zero (outlier prices help, the margin hurts), with one
    bookmaker it is about minus that bookmaker's margin."""
    bench = pred[[f"bench_{o}" for o in OUT]].to_numpy(dtype=float)
    ok = ~np.isnan(bench).any(axis=1)
    implied = 1 / bench[ok]
    fair = implied / implied.sum(axis=1, keepdims=True)
    out = {}
    for price, prefix in PRICE_COLS.items():
        cols = [f"{prefix}_{o}" for o in OUT]
        if set(cols).issubset(pred.columns):
            out[price] = float(np.nanmean(pred[cols].to_numpy(dtype=float)[ok] * fair - 1))
    return out


def by_season(bets: pd.DataFrame) -> pd.DataFrame:
    """Bets, ROI and fair CLV per season: is the edge stable or one lucky year?"""
    cols = ["season", "bets", "roi", "avg_clv_fair", "pct_clv_fair_pos"]
    if bets.empty:
        return pd.DataFrame(columns=cols)
    g = bets.groupby("season")
    out = pd.DataFrame({
        "bets": g.size(),
        "roi": g.profit.sum() / g.stake.sum(),
        "avg_clv_fair": g.clv_fair.mean(),
        "pct_clv_fair_pos": g.clv_fair.apply(lambda c: (c.dropna() > 0).mean()),
    }).reset_index()
    out["season"] = out.season.map(lambda y: f"{y}/{(y + 1) % 100:02d}")
    return out[cols]


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough fall, as a fraction of the peak."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def summary(bets: pd.DataFrame, edge: float, staking: str = "flat",
            start_bankroll: float = 100.0, price: str | None = None) -> dict:
    price = price or (bets.price.iat[0] if "price" in bets.columns and len(bets) else "book")
    if bets.empty:
        return {"price": price, "staking": staking, "min_edge": edge, "bets": 0}
    turnover = bets.stake.sum()
    lo, hi = bootstrap_roi(bets.profit.to_numpy(), bets.stake.to_numpy())
    clv_fair = bets.clv_fair.dropna() if "clv_fair" in bets.columns else pd.Series(dtype=float)
    out = {
        "price": price,
        "staking": staking,
        "min_edge": edge,
        "bets": len(bets),
        "hit_rate": (bets.profit > 0).mean(),
        "avg_odds": bets.bet_odds.mean(),
        "staked": turnover,
        "profit_units": bets.profit.sum(),
        "roi": bets.profit.sum() / turnover if turnover else np.nan,
        "roi_ci_low": lo,
        "roi_ci_high": hi,
        "avg_clv": bets.clv.mean(),
        "pct_beat_close": (bets.clv.dropna() > 0).mean() if bets.clv.notna().any() else np.nan,
        "avg_clv_fair": clv_fair.mean() if len(clv_fair) else np.nan,
        "pct_clv_fair_pos": (clv_fair > 0).mean() if len(clv_fair) else np.nan,
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
    bench_cols = [c for c in ("bench_h", "bench_d", "bench_a") if c in history.columns]
    h = history[["date", "home", "away", "result", "book",
                 "close_h", "close_d", "close_a"] + bench_cols].rename(
        columns={"date": "played", "book": "hist_book"})
    m = t.reset_index().merge(h, on=["home", "away"], how="left")
    same_season = m.played.map(_season, na_action="ignore") == m.date.map(_season)
    m = m[m.played.isna() | (same_season & ((m.date - m.played).dt.days <= days_before))]
    m = (m.assign(_gap=(m.played - m.date).abs())
           .sort_values(["index", "_gap"]).drop_duplicates("index"))
    m = t.reset_index().merge(m[["index", "played", "result", "hist_book",
                                 "close_h", "close_d", "close_a"] + bench_cols],
                              on="index", how="left")

    pick_i = m.pick.map({o: i for i, o in enumerate(OUT)}).to_numpy()
    closes = m[["close_h", "close_d", "close_a"]].to_numpy(dtype=float)
    close = closes[np.arange(len(m)), pick_i]
    same_book = m.hist_book.astype(object).eq(m.book).to_numpy()
    m["close_odds"] = np.where(same_book, close, np.nan)
    m["clv"] = m.bet_odds / m.close_odds - 1
    if len(bench_cols) == 3:
        implied = 1 / m[bench_cols].to_numpy(dtype=float)
        fair = implied / implied.sum(axis=1, keepdims=True)
        m["clv_fair"] = m.bet_odds * fair[np.arange(len(m)), pick_i] - 1
    else:
        m["clv_fair"] = np.nan
    settled = m.result.notna()
    won = m.pick.str.upper() == m.result
    m["status"] = np.where(settled, np.where(won, "won", "lost"), "open")
    m["profit"] = np.where(settled, np.where(won, m.bet_odds - 1, -1.0), np.nan)
    return m.drop(columns=["index", "close_h", "close_d", "close_a", "hist_book"] + bench_cols)
