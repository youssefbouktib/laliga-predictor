"""La Liga match predictor.

    python run.py download   # results + odds (+ Understat xG) + upcoming fixtures
    python run.py evaluate   # walk-forward test of every model, writes reports/
    python run.py predict    # predictions + value bets for upcoming fixtures
    python run.py settle     # attach results + closing odds to paper trades
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import shutil

import pandas as pd

from src import backtest, data, dixon_coles, features, model, plots, xg

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
DOCS_IMG = ROOT / "docs" / "img"
EDGES = [0.02, 0.05, 0.10]
MAX_EDGE = 0.15     # skip "too good to be true" edges: usually model error or a stale price
PRICE = "best"      # bet at the best price across bookmakers (line shopping)
PAPER_LOG = "paper_trades.csv"


def today() -> pd.Timestamp:
    """Today's date (a function so tests can pretend it is another day)."""
    return pd.Timestamp.today().normalize()


def load_dataset(with_dc: bool = False) -> tuple[pd.DataFrame, features.TeamState]:
    """History with every model input. Dixon-Coles inputs take ~15 s on the
    full history, so they are only built when a model needs them."""
    hist = xg.merge(data.load_history(RAW), RAW)
    df, state = features.build_features(hist)
    df = model.add_market_features(df)
    if with_dc:
        df = model.add_dc_features(df)
    return df, state


def _edge_range(min_edge: float, max_edge: float | None) -> str:
    return f"edge > {min_edge:.0%}" if max_edge is None else f"edge {min_edge:.0%} to {max_edge:.0%}"


def _needs_dc(names) -> bool:
    return any("dc" in model.requirements(n) for n in names)


# ---------------------------------------------------------------- download

def cmd_download(args):
    data.download(RAW, first=args.first, last=args.last)
    if not args.no_xg:
        print("Downloading Understat xG ...")
        xg.download(RAW, first=args.first, last=args.last)


# ---------------------------------------------------------------- evaluate

def cmd_evaluate(args):
    REPORTS.mkdir(exist_ok=True)
    DOCS_IMG.mkdir(parents=True, exist_ok=True)
    price = getattr(args, "price", PRICE)
    max_edge = getattr(args, "max_edge", MAX_EDGE)
    names = list(args.models or model.ALL_MODELS)
    if args.model and args.model not in names:
        names.append(args.model)  # the main model must be tested too
    df, _ = load_dataset(with_dc=_needs_dc(names))
    available = model.available_models(df)
    if not model.has_xg(df):
        print("No Understat xG data found: xG models skipped (run download when Understat is up).")
    if args.model and args.model not in available:
        missing = model.requirements(args.model) - model.inputs_available(df)
        if "xg" in missing:
            raise SystemExit(f"Model {args.model!r} needs Understat xG data, which is not available.")
        raise SystemExit(f"Model {args.model!r} needs inputs that are not available: {sorted(missing)}")
    skipped = [n for n in names if n not in available]
    if skipped and args.models:
        print(f"  ! skipping {skipped} (missing inputs)")
    names = [n for n in names if n in available]
    if not names:
        raise SystemExit("No model left to evaluate.")

    preds, metrics = {}, []
    for name in names:
        print(f"Walk-forward: {name} ...")
        preds[name] = model.walk_forward(df, n_test_seasons=args.test_seasons, model=name)
        metrics.append(model.evaluate(preds[name]).assign(model=name))
    metrics = pd.concat(metrics, ignore_index=True)
    comp = model.compare(preds)
    main = args.model or (model.DEFAULT_MODEL if model.DEFAULT_MODEL in preds else comp.model.iloc[0])

    print("\nModel comparison, all test matches pooled (lower is better):")
    print(comp.to_string(index=False, float_format="%.4f"))
    print(f"\nPer season, {main}:")
    print(metrics[metrics.model == main].drop(columns="model").to_string(index=False, float_format="%.4f"))

    # backtests: main model at every edge, both prices, flat + Kelly;
    # every other model at 5%, both prices, flat
    rows, flat_curves, kelly_curves = [], {}, {}
    for name, pred in preds.items():
        for p in ("book", "best"):
            for e in EDGES if name == main else [0.05]:
                value = backtest.find_value(pred, e, p, max_edge)
                flat = backtest.settle(value)
                rows.append({"model": name, **backtest.summary(flat, e, "flat", price=p)})
                if name == main:
                    kelly = backtest.settle_kelly(value)
                    rows.append({"model": name, **backtest.summary(kelly, e, "kelly", price=p)})
                    if p == price and len(value):
                        flat_curves[e], kelly_curves[e] = flat, kelly
    res = pd.DataFrame(rows)
    control = backtest.control_clv(preds[main])
    cap = "no edge cap" if max_edge is None else f"edges capped at {max_edge:.0%}"
    print(f"\nValue-betting backtest ({cap}; "
          f"{main}: all edges, flat and quarter-Kelly; others: 5% flat):")
    print(res.round(3).to_string(index=False))
    print("\nControl, no model (bet every outcome): average fair CLV "
          + ", ".join(f"{p} price {v:+.2%}" for p, v in control.items()))

    strategy = backtest.settle(backtest.find_value(preds[main], 0.05, price, max_edge))
    by_season = backtest.by_season(strategy)
    print(f"\nMain strategy by season ({main}, {price} price, {_edge_range(0.05, max_edge)}, flat):")
    print(by_season.round(3).to_string(index=False))

    cal = model.calibration_table(preds[main])
    comp.to_csv(REPORTS / "model_comparison.csv", index=False)
    metrics.to_csv(REPORTS / "probability_metrics.csv", index=False)
    res.to_csv(REPORTS / "backtest_summary.csv", index=False)
    by_season.to_csv(REPORTS / "strategy_by_season.csv", index=False)
    pd.DataFrame([{"price": p, "avg_clv_fair": v} for p, v in control.items()]).to_csv(
        REPORTS / "control_clv.csv", index=False)
    cal.to_csv(REPORTS / "calibration.csv", index=False)
    pd.concat(preds.values()).to_csv(REPORTS / "predictions.csv", index=False)

    charts = {
        "model_comparison.png": lambda p: plots.model_comparison(comp, p),
        "season_logloss.png": lambda p: plots.season_logloss(metrics, main, p),
        "calibration.png": lambda p: plots.calibration(cal, main, p),
        "profit_curve.png": lambda p: plots.profit_curves(flat_curves, kelly_curves, main, p),
        "clv_by_season.png": lambda p: plots.clv_by_season(by_season, control.get(price), main, price, p),
    }
    for fname, draw in charts.items():
        draw(REPORTS / fname)
        shutil.copy(REPORTS / fname, DOCS_IMG / fname)  # docs/img is committed for the README
    print(f"\nReports written to {REPORTS}/ (charts also copied to docs/img/)")


# ---------------------------------------------------------------- predict

def dc_fixture_probs(df: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Dixon-Coles fitted on all history, P(H/D/A) for each fixture."""
    # the fixtures' season comes from their date: it may not be in the history yet
    season = features.season_of(fixtures.date.min())
    this = df[df.season == season]
    prev = df[df.season == season - 1]
    # promoted = not in last season's league (if the data has no previous
    # season, only teams never seen at all count as promoted)
    prev_teams = (set(prev.home) | set(prev.away)) or (set(df.home) | set(df.away))
    teams = sorted(set(this.home) | set(this.away) | set(fixtures.home) | set(fixtures.away))
    played = pd.concat([this.home, this.away]).value_counts()
    # the promoted prior only matters until they have played some games
    promoted = {t for t in teams if t not in prev_teams and played.get(t, 0) < 10}
    m = dixon_coles.DixonColes()
    m.fit(df, pd.Timestamp(fixtures.date.min()), promoted=promoted, teams=teams)
    return pd.DataFrame([m.predict_hda(h, a) for h, a in zip(fixtures.home, fixtures.away)],
                        index=fixtures.index, columns=model.PROB_COLS)


def fit_for_prediction(name: str, df: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Train `name` on all history and return P(H/D/A) for each fixture.
    `fixtures` must already carry the model's inputs."""
    if name == "dixon_coles":
        return dc_fixture_probs(df, fixtures)
    _, feats, _ = model.TABULAR_MODELS[name]
    m = model.fit_tabular(name, df[model.train_rows(df, name)])
    return pd.DataFrame(model.predict_hda(m, fixtures, feats), index=fixtures.index,
                        columns=model.PROB_COLS)


def cmd_predict(args):
    REPORTS.mkdir(exist_ok=True)
    name = args.model
    price = getattr(args, "price", PRICE)
    max_edge = getattr(args, "max_edge", MAX_EDGE)

    fx = data.load_fixtures(RAW)
    hist = data.load_history(RAW)
    if not fx.empty:
        # fixtures.csv also lists this matchweek's games that are already played
        played = set(zip(hist.date, hist.home, hist.away))
        fx = fx[[k not in played for k in zip(fx.date, fx.home, fx.away)]].reset_index(drop=True)
    if fx.empty:
        print("No upcoming La Liga fixtures in fixtures.csv. Run download again later.")
        (REPORTS / "upcoming.csv").unlink(missing_ok=True)  # don't keep showing last week's games
        return

    needs_dc = "dc" in model.requirements(name)
    df, state = load_dataset(with_dc=needs_dc)
    if name not in model.available_models(df):
        raise SystemExit(f"Model {name!r} is not available (xG models need Understat data).")
    unknown = (set(fx.home) | set(fx.away)) - (set(df.home) | set(df.away))
    if unknown:
        print(f"  ! fixture teams never seen in history (treated as promoted): {sorted(unknown)}")

    fx = pd.concat([fx, features.predict_snapshots(state, fx)], axis=1)
    fx = model.add_market_features(fx)
    if needs_dc:
        fx = pd.concat([fx, model.dc_features_from_probs(dc_fixture_probs(df, fx).to_numpy(),
                                                         index=fx.index)], axis=1)
    fx[model.PROB_COLS] = fit_for_prediction(name, df, fx)
    fx["result"] = None

    out_cols = ["date", "home", "away", "p_h", "p_d", "p_a", "book", "odds_h", "odds_d", "odds_a",
                "best_h", "best_d", "best_a"]
    out_cols = [c for c in out_cols if c in fx.columns]
    fx[out_cols].assign(model=name).to_csv(REPORTS / "upcoming.csv", index=False)
    print(f"\nUpcoming fixtures ({name}; odds = {fx.book.iloc[0]}, best = best price anywhere):")
    print(fx[out_cols].to_string(index=False, float_format="%.2f"))

    value = backtest.find_value(fx, args.min_edge, price, max_edge)
    print(f"\nValue bets ({price} price, {_edge_range(args.min_edge, max_edge)}):")
    if value.empty:
        print("  none")
        return
    print(value[["date", "home", "away", "pick", "p_model", "bet_odds", "ev"]]
          .to_string(index=False, float_format="%.3f"))
    # only paper-trade games still to come (fixtures.csv can list games already
    # kicked off whose results are not published yet)
    value = value[value.date >= today()]
    log = REPORTS / PAPER_LOG
    value = value.assign(logged_at=dt.datetime.now().isoformat(timespec="minutes"), model=name)
    keep = ["logged_at", "model", "date", "home", "away", "book", "price", "pick", "p_model",
            "bet_odds", "ev"]
    if log.exists():  # log each match once (first price seen), even on reruns
        # keyed on (season, home, away), not the date: each pairing is played
        # once per season, so a rescheduled game is still the same game
        old = pd.read_csv(log, parse_dates=["date"])
        seen = {(features.season_of(d), h, a) for d, h, a in zip(old.date, old.home, old.away)}
        value = value[[(features.season_of(d), h, a) not in seen
                       for d, h, a in zip(value.date, value.home, value.away)]]
    if value.empty:
        print("\nNo new value bets to log (already logged, or already kicked off).")
        return
    if log.exists():
        old = pd.read_csv(log)
        if "price" not in old.columns:  # log written by an older version
            old.insert(old.columns.get_loc("book") + 1, "price", "book")
            old.to_csv(log, index=False)
    value[keep].to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"\nLogged {len(value)} to {log} (paper trading: no real money).")
    if price == "best":
        print("  Price = best available across bookmakers: shop around before betting.")


# ---------------------------------------------------------------- settle

def cmd_settle(args):
    log = REPORTS / PAPER_LOG
    if not log.exists():
        print("No paper trades logged yet. Run: python run.py predict")
        return
    trades = pd.read_csv(log)
    settled = backtest.settle_paper_trades(trades, data.load_history(RAW))
    settled.to_csv(REPORTS / "paper_trades_settled.csv", index=False)
    done = settled[settled.status != "open"]
    print(f"Paper trades: {len(settled)} logged, {len(done)} settled, "
          f"{(settled.status == 'open').sum()} open.")
    if len(done):
        print(f"  profit {done.profit.sum():+.2f} units on {len(done)} bets "
              f"(ROI {done.profit.mean():+.1%})")
        clv = done.clv.dropna()
        if len(clv):
            print(f"  avg CLV {clv.mean():+.2%}, beat the close {(clv > 0).mean():.0%} of the time")
        fair = done.clv_fair.dropna() if "clv_fair" in done.columns else pd.Series(dtype=float)
        if len(fair):
            print(f"  avg CLV vs sharp close {fair.mean():+.2%} "
                  f"(positive on {(fair > 0).mean():.0%} of bets)")
    print(f"Written to {REPORTS / 'paper_trades_settled.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--first", type=int, default=2012)
    d.add_argument("--last", type=int, default=2026)
    d.add_argument("--no-xg", action="store_true", help="skip Understat")
    d.set_defaults(func=cmd_download)
    e = sub.add_parser("evaluate")
    e.add_argument("--test-seasons", type=int, default=5)
    e.add_argument("--models", nargs="+", choices=model.ALL_MODELS,
                   help="models to test (default: all available)")
    e.add_argument("--model", choices=model.ALL_MODELS,
                   help=f"model for the backtest and charts (default: {model.DEFAULT_MODEL})")
    e.add_argument("--price", choices=["book", "best"], default=PRICE,
                   help="price for the main strategy's charts (default: best)")
    e.add_argument("--max-edge", type=float, default=MAX_EDGE)
    e.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("predict")
    p.add_argument("--min-edge", type=float, default=0.05)
    p.add_argument("--max-edge", type=float, default=MAX_EDGE)
    p.add_argument("--price", choices=["book", "best"], default=PRICE)
    p.add_argument("--model", choices=model.ALL_MODELS, default=model.DEFAULT_MODEL)
    p.set_defaults(func=cmd_predict)
    s = sub.add_parser("settle")
    s.set_defaults(func=cmd_settle)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
