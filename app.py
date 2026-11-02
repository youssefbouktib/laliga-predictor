"""Streamlit dashboard for the La Liga predictor.

    streamlit run app.py

Reads the files written by `run.py evaluate`, `predict` and `settle`;
it never trains a model itself, so it opens instantly.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).parent
REPORTS = ROOT / "reports"

st.set_page_config(page_title="La Liga Predictor", layout="wide")


@st.cache_data
def _read_csv(path: str, mtime: float, **kw) -> pd.DataFrame:
    # mtime is part of the cache key: a rewritten report is re-read at once
    return pd.read_csv(path, **kw)


def read(name: str, **kw) -> pd.DataFrame | None:
    f = REPORTS / name
    return _read_csv(str(f), f.stat().st_mtime, **kw) if f.exists() else None


def pct(x):
    return f"{x:+.1%}" if pd.notna(x) else "–"


st.title("La Liga Predictor")
st.caption("Can a model beat the bookmaker's closing probabilities? Walk-forward tested, "
           "paper-traded, no real money.")

comp = read("model_comparison.csv")
if comp is None:
    st.warning("No reports yet. Run `python run.py download` then `python run.py evaluate`.")
    st.stop()

tab_up, tab_paper, tab_models, tab_bt = st.tabs(
    ["Upcoming", "Paper trades", "Model vs market", "Backtest"])

# ---------------------------------------------------------------- upcoming
with tab_up:
    up = read("upcoming.csv", parse_dates=["date"])
    if up is None or up.empty:
        st.info("No upcoming fixtures. Run `python run.py download` and `python run.py predict`.")
    else:
        has_best = {"best_h", "best_d", "best_a"}.issubset(up.columns)
        c = st.columns(3)
        min_edge = c[0].slider("Minimum edge", 0.0, 0.20, 0.05, 0.01, format="%.2f")
        max_edge = c[1].slider("Maximum edge (skip too-good-to-be-true)", 0.05, 1.0, 0.15, 0.01,
                               format="%.2f")
        price = c[2].radio("Price", ["best", "book"] if has_best else ["book"], horizontal=True,
                           help="best = best price across all bookmakers (line shopping)")
        prefix = "best" if price == "best" else "odds"
        rows = []
        for r in up.itertuples():
            evs = {o: getattr(r, f"p_{o}") * getattr(r, f"{prefix}_{o}") - 1 for o in "hda"}
            evs = {o: v for o, v in evs.items() if pd.notna(v)}
            pick = max(evs, key=evs.get) if evs else None
            ev = evs[pick] if pick else float("nan")
            rows.append({
                "date": r.date.date(), "match": f"{r.home} v {r.away}",
                "P(H)": r.p_h, "P(D)": r.p_d, "P(A)": r.p_a,
                "odds H": getattr(r, f"{prefix}_h"), "odds D": getattr(r, f"{prefix}_d"),
                "odds A": getattr(r, f"{prefix}_a"),
                "best EV": ev,
                "value bet": ({"h": "Home", "d": "Draw", "a": "Away"}[pick]
                              if pick and min_edge < ev <= max_edge else ""),
            })
        t = pd.DataFrame(rows)
        st.caption(f"Model: {up.model.iloc[0]} · prices: "
                   + ("best available across bookmakers" if price == "best" else f"{up.book.iloc[0]}"))
        st.dataframe(
            t, hide_index=True, width="stretch",
            column_config={
                **{c: st.column_config.NumberColumn(format="%.2f") for c in
                   ["P(H)", "P(D)", "P(A)", "odds H", "odds D", "odds A"]},
                "best EV": st.column_config.NumberColumn(format="%+.3f"),
            })

# ---------------------------------------------------------------- paper trades
with tab_paper:
    raw = read("paper_trades.csv", parse_dates=["date"])
    settled = read("paper_trades_settled.csv", parse_dates=["date"])
    if raw is None and settled is None:
        st.info("No paper trades yet. `python run.py predict` logs value bets automatically.")
    else:
        # the settled file is a snapshot from the last `settle`: join it onto the
        # full log so trades logged since then still show up (as open)
        if settled is None:
            st.info("Run `python run.py settle` to attach results and closing odds.")
            pt = raw.assign(status="open", profit=float("nan"), clv=float("nan"))
        elif raw is None:
            pt = settled
        else:
            key = ["date", "home", "away"]
            extra = settled.drop(columns=[c for c in raw.columns if c not in key], errors="ignore")
            pt = raw.merge(extra.drop_duplicates(key), on=key, how="left")
            pt["status"] = pt.status.fillna("open")
        done = pt[pt.status != "open"]
        c = st.columns(4)
        c[0].metric("Bets logged", len(pt))
        c[1].metric("Settled", len(done))
        c[2].metric("Profit (units)", f"{done.profit.sum():+.2f}" if len(done) else "–")
        clv = (done.clv_fair if "clv_fair" in done.columns else done.clv).dropna()
        c[3].metric("Avg CLV vs sharp close", pct(clv.mean()) if len(clv) else "–",
                    help="Bet odds x fair closing probability - 1. Positive over many bets "
                         "is the best evidence of a real edge.")
        if len(done):
            st.line_chart(done.sort_values("date").set_index("date").profit.cumsum(),
                          y_label="cumulative profit (units)")
        st.dataframe(pt, hide_index=True, width="stretch")

# ---------------------------------------------------------------- models
with tab_models:
    best = comp.iloc[0]
    c = st.columns(3)
    c[0].metric("Best model", best.model)
    c[1].metric("Its log loss", f"{best.logloss:.4f}")
    c[2].metric("Gap to bookmaker", f"{best.gap_logloss:+.4f}", delta_color="inverse",
                delta=f"{best.gap_logloss:+.4f}")
    st.dataframe(comp, hide_index=True, width="stretch")
    metrics = read("probability_metrics.csv")
    if metrics is not None:
        model_name = st.selectbox("Per-season detail", comp.model.tolist())
        m = metrics[metrics.model == model_name].set_index("season")
        st.line_chart(m[["model_logloss", "bookie_logloss"]], y_label="log loss (lower is better)")
    cols = st.columns(2)
    for col, img in zip(cols, ["calibration.png", "model_comparison.png"]):
        if (REPORTS / img).exists():
            col.image(str(REPORTS / img))

# ---------------------------------------------------------------- backtest
with tab_bt:
    by_season = read("strategy_by_season.csv")
    control = read("control_clv.csv")
    if (REPORTS / "clv_by_season.png").exists():
        st.image(str(REPORTS / "clv_by_season.png"))
    if by_season is not None:
        st.dataframe(by_season, hide_index=True, width="stretch")
    if control is not None:
        st.caption("Control, no model (bet every outcome): average CLV vs sharp close "
                   + ", ".join(f"{r.price} price {r.avg_clv_fair:+.2%}" for r in control.itertuples()))
    bt = read("backtest_summary.csv")
    if bt is not None:
        st.subheader("All strategies")
        st.dataframe(bt, hide_index=True, width="stretch")
    if (REPORTS / "profit_curve.png").exists():
        st.image(str(REPORTS / "profit_curve.png"))
    st.caption("Flat stakes risk 1 unit per bet. Quarter Kelly stakes 25% of the Kelly fraction, "
               "capped at 5% of the bankroll. roi_ci_low/high: 95% bootstrap interval; if it "
               "contains 0, the profit or loss may be luck.")
