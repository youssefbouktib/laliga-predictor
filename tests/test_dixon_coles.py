import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

import synth
from src import dixon_coles
from src.dixon_coles import DCConfig, DixonColes


@pytest.fixture(scope="module")
def matches():
    return synth.clean_matches(n_teams=10, seasons=(2019, 2020, 2021), home_adv=0.3)


@pytest.fixture(scope="module")
def fitted(matches):
    as_of = matches.date.max() + pd.Timedelta(days=1)
    return DixonColes().fit(matches, as_of)


def test_gradient_matches_finite_differences(matches):
    as_of = pd.Timestamp("2021-01-01")
    m = DixonColes().fit(matches, as_of, promoted={"T10"})
    rng = np.random.default_rng(0)
    n = len(m.teams)
    for _ in range(3):
        p = np.concatenate([rng.normal(0, 0.3, 2 * n), [0.2, 0.25, rng.uniform(-0.1, 0.1)]])
        val, grad = m._objective(p)
        assert val < 1e11
        num = approx_fprime(p, lambda q: m._objective(q)[0], 1e-6)
        np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-3)


def test_predict_hda_sums_to_one_and_score_matrix(fitted):
    for h, a in [("T01", "T10"), ("T05", "T06"), ("T10", "T01"), ("Unknown", "T01")]:
        p = fitted.predict_hda(h, a)
        assert p.shape == (3,)
        assert p.sum() == pytest.approx(1.0)
        assert (p > 0).all()
    mat = fitted.score_matrix("T01", "T02")
    assert mat.sum() == pytest.approx(1.0)
    # rows are home goals: P(H) is the lower triangle
    assert fitted.predict_hda("T01", "T02")[0] == pytest.approx(np.tril(mat, -1).sum())


def test_stronger_team_favoured_and_home_advantage(fitted):
    att, dfn, mu, home, rho = fitted._unpack(fitted.params)
    assert home > 0.1
    assert -0.3 <= rho <= 0.3
    i1, i10 = fitted.teams.index("T01"), fitted.teams.index("T10")
    assert att[i1] > att[i10] and dfn[i1] < dfn[i10]
    p = fitted.predict_hda("T01", "T10")
    q = fitted.predict_hda("T10", "T01")
    assert p[0] > p[2] and q[2] > q[0]
    # home advantage: T05 v T06 at home better than away
    assert fitted.predict_hda("T05", "T06")[0] > fitted.predict_hda("T06", "T05")[2]


def test_fit_ignores_matches_on_or_after_as_of(matches):
    as_of = pd.Timestamp("2020-12-01")
    a = DixonColes().fit(matches, as_of)
    future = matches.copy()
    late = future.date >= as_of
    future.loc[late, "hg"] = 9
    future.loc[late, "ag"] = 0
    b = DixonColes().fit(future, as_of)
    np.testing.assert_allclose(a.params, b.params)
    assert a.teams == b.teams


def test_promoted_prior_for_team_without_data(matches):
    cfg = DCConfig()
    as_of = pd.Timestamp("2021-06-01")
    teams = sorted(set(matches.home)) + ["NEW"]
    m = DixonColes().fit(matches, as_of, promoted={"NEW"}, teams=teams)
    att, dfn, *_ = m._unpack(m.params)
    i = m.teams.index("NEW")
    assert att[i] == pytest.approx(cfg.promoted_attack, abs=1e-4)
    assert dfn[i] == pytest.approx(cfg.promoted_defence, abs=1e-4)
    # an unknown team (not even in `teams`) falls back to the same prior
    np.testing.assert_allclose(m.predict_hda("T03", "NEW"), m.predict_hda("T03", "NEVER_SEEN"), atol=1e-4)


def test_promoted_in():
    df = pd.DataFrame({"season": [2019, 2019, 2020, 2020], "home": ["A", "B", "A", "C"],
                       "away": ["B", "A", "C", "A"]})
    assert dixon_coles.promoted_in(df, 2020) == {"C"}
    assert dixon_coles.promoted_in(df, 2019) == set()


def test_walk_forward_fits_only_on_past(matches, monkeypatch):
    seen = []
    orig = DixonColes.fit

    def spy(self, df, as_of, promoted=None, teams=None):
        seen.append(as_of)
        return orig(self, df, as_of, promoted=promoted, teams=teams)

    monkeypatch.setattr(DixonColes, "fit", spy)
    pred = dixon_coles.walk_forward(matches, [2021])
    assert set(pred.season) == {2021}
    assert len(pred) == (matches.season == 2021).sum()
    np.testing.assert_allclose(pred[["p_h", "p_d", "p_a"]].sum(axis=1), 1.0)
    weeks = pred.date.dt.to_period("W")
    # one fit per calendar week, as_of = first match day of that week
    assert len(seen) == weeks.nunique()
    for wk, as_of in zip(weeks.unique(), seen):
        assert as_of == pred.date[weeks == wk].min()
        assert (pred.date[weeks == wk] >= as_of).all()


def test_walk_forward_unchanged_when_future_results_change(matches):
    base = dixon_coles.walk_forward(matches, [2021])
    weeks = base.date.dt.to_period("W")
    cutoff = base.date[weeks == weeks.unique()[10]].min()
    alt = matches.copy()
    alt.loc[alt.date >= cutoff, ["hg", "ag"]] = [[7, 0]] * int((alt.date >= cutoff).sum())
    out = dixon_coles.walk_forward(alt, [2021])
    keep = (base.date < cutoff + pd.Timedelta(days=7)) & (weeks <= weeks.unique()[10])
    np.testing.assert_allclose(out.loc[keep, ["p_h", "p_d", "p_a"]].to_numpy(),
                               base.loc[keep, ["p_h", "p_d", "p_a"]].to_numpy(), atol=1e-12)
    assert not np.allclose(out.loc[~keep, "p_h"], base.loc[~keep, "p_h"])
