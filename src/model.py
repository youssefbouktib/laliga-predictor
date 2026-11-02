"""Models, walk-forward training and probability evaluation.

Every model is tested the same way: for each test season, train only on
earlier seasons and predict that season. Every model is judged against the same
benchmark: margin-free closing odds (`bench_*`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import dixon_coles
from .data import fair_probs
from .features import FEATURES, XG_FEATURES

OUTCOMES = ["H", "D", "A"]
PROB_COLS = ["p_h", "p_d", "p_a"]


# ---------------------------------------------------------------- estimators

def make_logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def make_lgbm():
    from lightgbm import LGBMClassifier
    # small, heavily regularised trees: football data is noisy and the
    # signal is mostly linear in Elo, so we only want gentle non-linearity
    return LGBMClassifier(
        objective="multiclass", n_estimators=250, learning_rate=0.02,
        num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0, random_state=42, verbose=-1,
    )


# name -> (estimator factory, feature list, needs xG)
TABULAR_MODELS = {
    "logistic": (make_logistic, FEATURES, False),
    "logistic_xg": (make_logistic, FEATURES + XG_FEATURES, True),
    "lgbm": (make_lgbm, FEATURES, False),
    "lgbm_xg": (make_lgbm, FEATURES + XG_FEATURES, True),
}
ALL_MODELS = list(TABULAR_MODELS) + ["dixon_coles"]
# Used for the main report, the profit curve and `predict` unless overridden.
# Chosen by mean walk-forward log loss on real data (see README).
DEFAULT_MODEL = "dixon_coles"


def has_xg(df: pd.DataFrame) -> bool:
    return "hxg" in df.columns and df["hxg"].notna().any()


def available_models(df: pd.DataFrame) -> list[str]:
    return [m for m in ALL_MODELS if m == "dixon_coles" or not TABULAR_MODELS[m][2] or has_xg(df)]


def predict_hda(model, X: pd.DataFrame, features: list[str] = FEATURES) -> np.ndarray:
    proba = model.predict_proba(X[features])
    order = [list(model.classes_).index(o) for o in OUTCOMES]
    return proba[:, order]


def train_rows(df: pd.DataFrame, needs_xg: bool) -> pd.Series:
    """Rows a model may train on: skip the Elo warm-up season, and for xG
    models skip seasons without xG plus the first xG season (rolling xG still
    filling up)."""
    ok = df.season > df.season.min()
    if needs_xg:
        xg_seasons = sorted(df.loc[df.hxg.notna(), "season"].unique())
        ok &= df.season > xg_seasons[0]
    return ok


def fit_tabular(name: str, train: pd.DataFrame):
    factory, feats, _ = TABULAR_MODELS[name]
    return factory().fit(train[feats], train.result)


def walk_forward(df: pd.DataFrame, n_test_seasons: int = 5, model: str = "logistic") -> pd.DataFrame:
    """Train on all earlier seasons, predict the next one. Never peek ahead."""
    seasons = sorted(df.season.unique())
    # at least the Elo warm-up season plus one training season must come first
    if not 1 <= n_test_seasons <= len(seasons) - 2:
        raise ValueError(f"n_test_seasons must be between 1 and {len(seasons) - 2} "
                         f"with {len(seasons)} seasons of data (got {n_test_seasons})")
    test_seasons = seasons[-n_test_seasons:]
    if model == "dixon_coles":
        pred = dixon_coles.walk_forward(df, test_seasons)
    else:
        _, feats, needs_xg = TABULAR_MODELS[model]
        allowed = train_rows(df, needs_xg)
        out = []
        for s in test_seasons:
            train = df[allowed & (df.season < s)]
            test = df[df.season == s].copy()
            m = fit_tabular(model, train)
            test[PROB_COLS] = predict_hda(m, test, feats)
            out.append(test)
        pred = pd.concat(out)
    pred = pred.copy()
    pred["model"] = model
    return pred


# ---------------------------------------------------------------- metrics

def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def _scored(g: pd.DataFrame):
    """Rows with a benchmark price, plus model probs, bookie probs, labels."""
    g = g[g[["bench_h", "bench_d", "bench_a"]].notna().all(axis=1)]
    y = g.result.map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    pm = g[PROB_COLS].to_numpy()
    pb = fair_probs(g[["bench_h", "bench_d", "bench_a"]].to_numpy())
    return g, y, pm, pb


def evaluate(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-season model vs bookmaker on the same matches (lower is better)."""
    rows = []
    for season, g in pred.groupby("season"):
        g, y, pm, pb = _scored(g)
        if not len(g):
            continue
        rows.append({
            "season": f"{season}/{(season + 1) % 100:02d}",
            "matches": len(g),
            "model_logloss": _log_loss(pm, y),
            "bookie_logloss": _log_loss(pb, y),
            "model_brier": _brier(pm, y),
            "bookie_brier": _brier(pb, y),
        })
    return pd.DataFrame(rows)


def compare(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per model, pooled over all test seasons, sorted best first."""
    rows = []
    for name, pred in preds.items():
        g, y, pm, pb = _scored(pred)
        rows.append({
            "model": name,
            "matches": len(g),
            "logloss": _log_loss(pm, y),
            "brier": _brier(pm, y),
            "bookie_logloss": _log_loss(pb, y),
            "bookie_brier": _brier(pb, y),
        })
    out = pd.DataFrame(rows)
    out["gap_logloss"] = out.logloss - out.bookie_logloss
    return out.sort_values("logloss").reset_index(drop=True)


def calibration_table(pred: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Reliability data, pooling all three outcomes: within each predicted-
    probability bin, how often did the event actually happen?"""
    g, y, pm, pb = _scored(pred)
    onehot = np.eye(3)[y]
    rows = []
    for source, p in (("model", pm), ("bookie", pb)):
        pf, of = p.ravel(), onehot.ravel()
        b = np.minimum((pf * bins).astype(int), bins - 1)
        for k in range(bins):
            sel = b == k
            if sel.sum() >= 20:
                rows.append({"source": source, "bin": k, "n": int(sel.sum()),
                             "predicted": pf[sel].mean(), "observed": of[sel].mean()})
    return pd.DataFrame(rows)
