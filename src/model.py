"""Models, walk-forward training and probability evaluation.

Every model is tested the same way: for each test season, train only on
earlier seasons and predict that season. Every model is judged against the same
benchmark: margin-free closing odds (`bench_*`).

Two families of models:

* Football models use only football information (Elo, form, shots, xG,
  Dixon-Coles). They answer "how good is my own forecast?".
* Market-aware models also see the sharp *pre-match* odds (published before
  kickoff, so no leakage). They answer the betting question: "does my
  football information improve on what the market already knows?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import dixon_coles
from .data import fair_probs
from .features import FEATURES, SHOT_FEATURES, XG_FEATURES

OUTCOMES = ["H", "D", "A"]
PROB_COLS = ["p_h", "p_d", "p_a"]
# log(P(H)/P(D)) and log(P(A)/P(D)): the natural inputs for a multinomial
# logistic model built on top of another model's probabilities
MARKET_FEATURES = ["mkt_lh", "mkt_la"]
DC_FEATURES = ["dc_lh", "dc_la"]


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


# name -> (estimator factory, feature list, required inputs)
TABULAR_MODELS = {
    # football models
    "logistic": (make_logistic, FEATURES, set()),
    "logistic_shots": (make_logistic, FEATURES + SHOT_FEATURES, {"shots"}),
    "logistic_xg": (make_logistic, FEATURES + XG_FEATURES, {"xg"}),
    "lgbm": (make_lgbm, FEATURES, set()),
    "lgbm_shots": (make_lgbm, FEATURES + SHOT_FEATURES, {"shots"}),
    "lgbm_xg": (make_lgbm, FEATURES + XG_FEATURES, {"xg"}),
    # stacked: Dixon-Coles probabilities + Elo + shots in one logistic model
    "ensemble": (make_logistic, DC_FEATURES + ["elo_diff"] + SHOT_FEATURES, {"dc", "shots"}),
    # market-aware
    "market": (make_logistic, MARKET_FEATURES, {"market"}),
    # market + football. Shots are left out on purpose: adding them to the
    # market made walk-forward log loss worse (0.9575 -> 0.9612).
    "market_plus": (make_logistic, MARKET_FEATURES + DC_FEATURES + ["elo_diff"], {"market", "dc"}),
}
ALL_MODELS = list(TABULAR_MODELS) + ["dixon_coles"]
MARKET_MODELS = {m for m, (_, _, req) in TABULAR_MODELS.items() if "market" in req}
# Used by `predict` unless overridden. Chosen by walk-forward log loss on real
# data (see README).
DEFAULT_MODEL = "market_plus"


def has_xg(df: pd.DataFrame) -> bool:
    return "hxg" in df.columns and df["hxg"].notna().any()


def inputs_available(df: pd.DataFrame) -> set[str]:
    have = set()
    if has_xg(df):
        have.add("xg")
    if set(SHOT_FEATURES).issubset(df.columns):
        have.add("shots")
    if set(DC_FEATURES).issubset(df.columns) and df[DC_FEATURES[0]].notna().any():
        have.add("dc")
    if set(MARKET_FEATURES).issubset(df.columns) and df[MARKET_FEATURES[0]].notna().any():
        have.add("market")
    return have


def available_models(df: pd.DataFrame) -> list[str]:
    have = inputs_available(df)
    return [m for m in ALL_MODELS if m == "dixon_coles" or TABULAR_MODELS[m][2] <= have]


def requirements(name: str) -> set[str]:
    return set() if name == "dixon_coles" else TABULAR_MODELS[name][2]


# ---------------------------------------------------------------- model inputs

def _log_ratios(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1)
    return np.log(p[:, 0] / p[:, 1]), np.log(p[:, 2] / p[:, 1])


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Margin-free sharp pre-match probabilities, as log-ratios."""
    out = df.copy()
    cols = ["sharp_h", "sharp_d", "sharp_a"]
    if not set(cols).issubset(df.columns) or df.empty:
        return out
    out["mkt_lh"], out["mkt_la"] = _log_ratios(fair_probs(df[cols].to_numpy(dtype=float)))
    return out


def add_dc_features(df: pd.DataFrame, cfg: dixon_coles.DCConfig | None = None) -> pd.DataFrame:
    """Out-of-sample Dixon-Coles probabilities for every match after the first
    season, each from a fit that only saw earlier matches (weekly refits).
    Because they never see the match itself, they are safe to use as inputs
    to another model."""
    out = df.copy()
    out["dc_lh"] = np.nan
    out["dc_la"] = np.nan
    seasons = sorted(df.season.unique())[1:]
    if not seasons:
        return out
    pred = dixon_coles.walk_forward(df, seasons, cfg)
    lh, la = _log_ratios(pred[PROB_COLS].to_numpy())
    out.loc[pred.index, "dc_lh"] = lh
    out.loc[pred.index, "dc_la"] = la
    return out


def dc_features_from_probs(p: np.ndarray, index=None) -> pd.DataFrame:
    lh, la = _log_ratios(p)
    return pd.DataFrame({"dc_lh": lh, "dc_la": la}, index=index)


def predict_hda(model, X: pd.DataFrame, features: list[str] = FEATURES) -> np.ndarray:
    proba = model.predict_proba(X[features])
    order = [list(model.classes_).index(o) for o in OUTCOMES]
    return proba[:, order]


def train_rows(df: pd.DataFrame, needs_xg: bool | str = False) -> pd.Series:
    """Rows a model may train on.

    Always skip the Elo warm-up season. xG models also skip seasons without
    xG plus the first xG season (rolling xG still filling up). Stacked models
    skip rows where one of their inputs is missing. `needs_xg` is either a
    bool or a model name.
    """
    name = needs_xg if isinstance(needs_xg, str) else None
    req = requirements(name) if name else ({"xg"} if needs_xg else set())
    ok = df.season > df.season.min()
    if "xg" in req:
        xg_seasons = sorted(df.loc[df.hxg.notna(), "season"].unique())
        ok &= df.season > xg_seasons[0]
    if name in TABULAR_MODELS:
        ok &= df[TABULAR_MODELS[name][1]].notna().all(axis=1)
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
        _, feats, _ = TABULAR_MODELS[model]
        allowed = train_rows(df, model)
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
    out["uses_market"] = out.model.isin(MARKET_MODELS)
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
