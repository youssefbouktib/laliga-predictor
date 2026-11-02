"""Report charts (matplotlib, PNG). One consistent, quiet style."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# categorical slots, fixed order: model, bookmaker, then extra series
C_MODEL, C_BOOKIE, C_3, C_4, C_5 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"
SERIES = [C_MODEL, C_3, C_4, C_5, "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, subtitle: str | None = None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: pathlib.Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def model_comparison(comp: pd.DataFrame, path: pathlib.Path):
    """Pooled log loss per model, with the bookmaker as a reference line."""
    comp = comp.sort_values("logloss", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(comp) + 1.6))
    y = range(len(comp))
    ax.hlines(y, comp.bookie_logloss, comp.logloss, color=GRID, lw=2)
    ax.scatter(comp.logloss, y, s=60, color=C_MODEL, zorder=3, label="Model", edgecolor="white", lw=1.5)
    ax.scatter(comp.bookie_logloss, y, s=60, color=C_BOOKIE, zorder=3, label="Bookmaker close",
               edgecolor="white", lw=1.5)
    for yi, (_, r) in zip(y, comp.iterrows()):
        ax.text(r.logloss + 0.0012, yi, f"{r.logloss:.4f}  ({r.gap_logloss:+.4f})", va="center",
                fontsize=8.5, color=MUTED)
    ax.set_yticks(list(y), comp.model)
    ax.set_xlabel("log loss over all test matches (lower is better)", color=MUTED, fontsize=9)
    lo = min(comp.bookie_logloss.min(), comp.logloss.min()) - 0.004
    hi = comp.logloss.max() + 0.02
    ax.set_xlim(lo, hi)
    _style(ax, "Every model vs the bookmaker", "Walk-forward test, same matches for every model")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    _save(fig, path)


def season_logloss(metrics: pd.DataFrame, model: str, path: pathlib.Path):
    m = metrics[metrics.model == model]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = range(len(m))
    ax.plot(x, m.model_logloss, "-o", color=C_MODEL, lw=2, ms=7, mec="white", mew=1.5, label=model)
    ax.plot(x, m.bookie_logloss, "-o", color=C_BOOKIE, lw=2, ms=7, mec="white", mew=1.5,
            label="Bookmaker close")
    ax.set_xticks(list(x), m.season)
    ax.set_ylabel("log loss", color=MUTED, fontsize=9)
    _style(ax, f"Log loss per season: {model} vs bookmaker", "Lower is better")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)


def calibration(cal: pd.DataFrame, model: str, path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    for source, color, label in (("bookie", C_BOOKIE, "Bookmaker close"), ("model", C_MODEL, model)):
        c = cal[cal.source == source]
        ax.plot(c.predicted, c.observed, "-o", color=color, lw=2, ms=6, mec="white", mew=1.2, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability", color=MUTED, fontsize=9)
    ax.set_ylabel("observed frequency", color=MUTED, fontsize=9)
    _style(ax, "Calibration", "H, D and A pooled, 10 bins")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _save(fig, path)


def profit_curves(flat: dict, kelly: dict, model: str, path: pathlib.Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for (edge, b), color in zip(sorted(flat.items()), SERIES):
        a1.plot(b.date, b.cum_profit, color=color, lw=1.8, label=f"edge > {edge:.0%} ({len(b)} bets)")
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_ylabel("profit (units)", color=MUTED, fontsize=9)
    _style(a1, "Flat 1-unit stakes", f"{model}, cumulative profit")
    a1.legend(frameon=False, fontsize=8.5)
    for (edge, b), color in zip(sorted(kelly.items()), SERIES):
        a2.plot(b.date, b.bankroll, color=color, lw=1.8, label=f"edge > {edge:.0%}")
    a2.axhline(100, color=MUTED, lw=0.8)
    a2.set_ylabel("bankroll (start = 100)", color=MUTED, fontsize=9)
    _style(a2, "Quarter Kelly, 5% cap", f"{model}, bankroll")
    a2.legend(frameon=False, fontsize=8.5)
    for ax in (a1, a2):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, path)


def clv_by_season(by_season: pd.DataFrame, control: float | None, model: str, price: str,
                  path: pathlib.Path):
    """Average fair CLV of the strategy's bets per season, against the
    no-model control (betting every outcome at the same prices)."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if by_season.empty:
        ax.text(0.5, 0.5, "no bets", ha="center", va="center", color=MUTED, transform=ax.transAxes)
    else:
        x = range(len(by_season))
        vals = by_season.avg_clv_fair * 100
        ax.bar(x, vals, width=0.55, color=C_MODEL, label=f"{model} value bets")
        for xi, v, n in zip(x, vals, by_season.bets):
            # labels always sit above the bar (or above zero for a negative bar)
            # so they never collide with the control line below
            ax.text(xi, max(v, 0) + 0.15, f"{v:+.1f}%\n{n} bets", ha="center",
                    va="bottom", fontsize=8.5, color=MUTED)
        ax.set_xticks(list(x), by_season.season)
        lo = min(0, vals.min(), (control or 0) * 100) - 1.2
        hi = max(0, vals.max(), (control or 0) * 100) + 1.4
        ax.set_ylim(lo, hi)
    ax.axhline(0, color=MUTED, lw=0.8)
    if control is not None:
        ax.axhline(control * 100, color=C_BOOKIE, lw=1.6, ls="--",
                   label=f"no model, every outcome ({control:+.1%})")
    ax.set_ylabel("average CLV vs sharp close (%)", color=MUTED, fontsize=9)
    _style(ax, "Do the bets beat the closing line?",
           f"{price} price, edge 5-15%. Above the dashed line = better than betting blindly")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    _save(fig, path)
