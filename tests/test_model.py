import numpy as np
import pandas as pd
import pytest

from src import model
from src.model import PROB_COLS


def test_walk_forward_trains_only_on_earlier_seasons_and_skips_warmup(featured, monkeypatch):
    df, _ = featured
    calls = []
    orig = model.fit_tabular

    def spy(name, train):
        calls.append(sorted(train.season.unique()))
        return orig(name, train)

    monkeypatch.setattr(model, "fit_tabular", spy)
    pred = model.walk_forward(df, n_test_seasons=3, model="logistic")
    test_seasons = sorted(df.season.unique())[-3:]
    assert sorted(pred.season.unique()) == test_seasons
    assert len(calls) == 3
    first = df.season.min()
    for s, train_seasons in zip(test_seasons, calls):
        assert first not in train_seasons, "Elo warm-up season must be skipped"
        assert max(train_seasons) == s - 1
        assert train_seasons == list(range(first + 1, s))
    assert len(pred) == df.season.isin(test_seasons).sum()
    assert (pred.model == "logistic").all()


@pytest.mark.parametrize("name", ["logistic", "lgbm", "dixon_coles"])
def test_walk_forward_probabilities(featured, name):
    df, _ = featured
    pred = model.walk_forward(df, n_test_seasons=2, model=name)
    p = pred[PROB_COLS].to_numpy()
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert ((p > 0) & (p < 1)).all()
    # features are carried through untouched
    pd.testing.assert_frame_equal(pred[["date", "home", "away", "result"]],
                                  df[df.season.isin(sorted(df.season.unique())[-2:])][
                                      ["date", "home", "away", "result"]])


def test_predict_hda_column_order():
    class Fake:
        classes_ = np.array(["A", "D", "H"])

        def predict_proba(self, X):
            return np.tile([0.2, 0.3, 0.5], (len(X), 1))

    X = pd.DataFrame({"f": [1.0, 2.0]})
    np.testing.assert_allclose(model.predict_hda(Fake(), X, ["f"]), [[0.5, 0.3, 0.2]] * 2)


def test_available_models_and_has_xg(featured):
    df, _ = featured
    # build_features alone: Elo/form + shots, but no market or Dixon-Coles inputs yet
    assert model.available_models(df) == ["logistic", "logistic_shots", "lgbm", "lgbm_shots",
                                          "dixon_coles"]
    d2 = df.assign(hxg=np.nan, axg=np.nan)
    assert not model.has_xg(d2)
    assert "logistic_xg" not in model.available_models(d2)
    d3 = d2.copy()
    d3.loc[d3.season >= 2019, ["hxg", "axg"]] = 1.2
    assert model.has_xg(d3)
    assert {"logistic_xg", "lgbm_xg"} <= set(model.available_models(d3))


def test_train_rows_for_xg_models(featured):
    df, _ = featured
    d = df.copy()
    d["hxg"], d["axg"] = np.nan, np.nan
    d.loc[d.season >= 2019, ["hxg", "axg"]] = 1.0
    ok = model.train_rows(d, needs_xg=True)
    assert sorted(d[ok].season.unique()) == [2020, 2021, 2022, 2023]
    assert sorted(d[model.train_rows(d, False)].season.unique()) == list(range(2018, 2024))


def _pred(n=300, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([4, 2.5, 3], n)
    y = np.array([rng.choice(3, p=pi) for pi in p])
    bench = 1 / (p * 1.02)
    return pd.DataFrame({
        "season": np.repeat([2021, 2022], n // 2),
        "result": np.array(["H", "D", "A"])[y],
        "p_h": p[:, 0], "p_d": p[:, 1], "p_a": p[:, 2],
        "bench_h": bench[:, 0], "bench_d": bench[:, 1], "bench_a": bench[:, 2],
    })


def test_evaluate_columns_and_values():
    pred = _pred()
    pred.loc[0, "bench_h"] = np.nan              # unpriced rows are excluded
    ev = model.evaluate(pred)
    assert list(ev.columns) == ["season", "matches", "model_logloss", "bookie_logloss",
                                "model_brier", "bookie_brier"]
    assert ev.season.tolist() == ["2021/22", "2022/23"]
    assert ev.matches.tolist() == [149, 150]
    # model probs == true probs == bookie fair probs here
    np.testing.assert_allclose(ev.model_logloss, ev.bookie_logloss, rtol=1e-9)
    y = pred.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()[150:]
    p = pred[PROB_COLS].to_numpy()[150:]
    assert ev.model_logloss[1] == pytest.approx(-np.mean(np.log(p[np.arange(150), y])))


def test_compare_sorted_with_gap():
    good = _pred()
    bad = good.copy()
    bad[PROB_COLS] = 1 / 3
    comp = model.compare({"bad": bad, "good": good})
    assert comp.model.tolist() == ["good", "bad"]
    assert list(comp.columns) == ["model", "matches", "logloss", "brier", "bookie_logloss",
                                  "bookie_brier", "gap_logloss", "uses_market"]
    assert not comp.uses_market.any()
    assert comp.loc[1, "logloss"] == pytest.approx(np.log(3))
    np.testing.assert_allclose(comp.gap_logloss, comp.logloss - comp.bookie_logloss)


def test_calibration_table():
    cal = model.calibration_table(_pred(3000))
    assert list(cal.columns) == ["source", "bin", "n", "predicted", "observed"]
    assert set(cal.source) == {"model", "bookie"}
    assert (cal.n >= 20).all()
    for _, r in cal.iterrows():
        assert r.bin / 10 <= r.predicted <= (r.bin + 1) / 10
    m = cal[cal.source == "model"]
    # well-calibrated synthetic probabilities
    assert np.average(np.abs(m.predicted - m.observed), weights=m.n) < 0.03


@pytest.mark.parametrize("n", [0, 6, 99])
def test_walk_forward_rejects_too_many_test_seasons(featured, n):
    df, _ = featured            # 7 seasons: warm-up + at least one training season first
    with pytest.raises(ValueError):
        model.walk_forward(df, n_test_seasons=n, model="logistic")
