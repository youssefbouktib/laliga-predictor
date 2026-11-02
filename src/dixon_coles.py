"""Dixon-Coles (1997) Poisson model for football scores.

Each team has an attack and a defence strength. Expected goals are

    home: lambda = exp(mu + home_adv + attack[home] + defence[away])
    away: mu_a   = exp(mu            + attack[away] + defence[home])

Goals are Poisson, with the Dixon-Coles correction `tau` that adjusts the
four low scores (0-0, 1-0, 0-1, 1-1), where independent Poisson is known to be
off. Older matches are down-weighted with exp(-xi * days_ago).

Fitting maximises the weighted log-likelihood with exact gradients (L-BFGS),
plus a small ridge penalty pulling each team toward a prior. Promoted teams get
a weaker prior, because with no recent data they would otherwise be treated as
average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass
class DCConfig:
    xi: float = 0.0019            # time decay per day (half-life ~ 1 year)
    ridge: float = 2.0            # penalty strength toward the team prior
    promoted_attack: float = -0.15
    promoted_defence: float = 0.12
    history_days: int = 3 * 365   # older matches have weight < 0.13 anyway
    max_goals: int = 10


class DixonColes:
    def __init__(self, cfg: DCConfig | None = None):
        self.cfg = cfg or DCConfig()
        self.teams: list[str] = []
        self.params: np.ndarray | None = None
        self.converged = True

    # parameter layout: [attack(n), defence(n), mu, home, rho]
    def _unpack(self, p):
        n = len(self.teams)
        return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp,
            promoted: set[str] | None = None, teams: list[str] | None = None) -> "DixonColes":
        cfg = self.cfg
        m = matches[(matches.date < as_of) &
                    (matches.date >= as_of - pd.Timedelta(days=cfg.history_days))]
        new_teams = sorted(set(teams or []) | set(m.home) | set(m.away))
        old = dict(zip(self.teams, range(len(self.teams))))
        prev = self.params
        self.teams = new_teams
        n = len(new_teams)
        idx = {t: i for i, t in enumerate(new_teams)}
        promoted = promoted or set()
        prior_att = np.array([cfg.promoted_attack if t in promoted else 0.0 for t in new_teams])
        prior_def = np.array([cfg.promoted_defence if t in promoted else 0.0 for t in new_teams])

        # warm start from the previous fit (big speed-up in walk-forward)
        x0 = np.concatenate([prior_att, prior_def, [0.25, 0.25, -0.05]])
        if prev is not None:
            pa, pd_, pmu, ph, pr = (prev[:len(old)], prev[len(old):2 * len(old)],
                                    *prev[2 * len(old):])
            for t, i in idx.items():
                if t in old and t not in promoted:
                    x0[i], x0[n + i] = pa[old[t]], pd_[old[t]]
            x0[2 * n:] = [pmu, ph, pr]

        if m.empty:
            self.params = x0
            self.converged = True
            return self

        hi = m.home.map(idx).to_numpy()
        ai = m.away.map(idx).to_numpy()
        x = m.hg.to_numpy(dtype=float)
        y = m.ag.to_numpy(dtype=float)
        days = (as_of - m.date).dt.days.to_numpy(dtype=float)
        w = np.exp(-cfg.xi * days)
        s00, s01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
        s10, s11 = (x == 1) & (y == 0), (x == 1) & (y == 1)

        def nll(p):
            att, dfn, mu, home, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]
            ll_ = mu + home + att[hi] + dfn[ai]
            lm_ = mu + att[ai] + dfn[hi]
            lam, mua = np.exp(ll_), np.exp(lm_)
            tau = np.ones_like(lam)
            tau[s00] = 1 - lam[s00] * mua[s00] * rho
            tau[s01] = 1 + lam[s01] * rho
            tau[s10] = 1 + mua[s10] * rho
            tau[s11] = 1 - rho
            if np.any(tau <= 0):
                return 1e12, np.zeros_like(p)
            ll = np.log(tau) + x * ll_ - lam + y * lm_ - mua
            # d log-lik / d log(lambda), d log(mu_away), d rho
            g_l = x - lam
            g_m = y - mua
            g_r = np.zeros_like(lam)
            g_l[s00] += -lam[s00] * mua[s00] * rho / tau[s00]
            g_m[s00] += -lam[s00] * mua[s00] * rho / tau[s00]
            g_r[s00] = -lam[s00] * mua[s00] / tau[s00]
            g_l[s01] += lam[s01] * rho / tau[s01]
            g_r[s01] = lam[s01] / tau[s01]
            g_m[s10] += mua[s10] * rho / tau[s10]
            g_r[s10] = mua[s10] / tau[s10]
            g_r[s11] = -1 / tau[s11]
            g_l, g_m, g_r = w * g_l, w * g_m, w * g_r

            grad = np.zeros_like(p)
            np.add.at(grad, hi, g_l)              # attack home
            np.add.at(grad, ai, g_m)              # attack away
            np.add.at(grad, n + ai, g_l)          # defence of away team (vs home goals)
            np.add.at(grad, n + hi, g_m)          # defence of home team
            grad[2 * n] = g_l.sum() + g_m.sum()
            grad[2 * n + 1] = g_l.sum()
            grad[2 * n + 2] = g_r.sum()

            val = -(w * ll).sum()
            ra, rd = att - prior_att, dfn - prior_def
            val += cfg.ridge * (ra @ ra + rd @ rd)
            grad = -grad
            grad[:n] += 2 * cfg.ridge * ra
            grad[n:2 * n] += 2 * cfg.ridge * rd
            return val, grad

        self._objective = nll  # kept for gradient checks in tests
        bounds = [(None, None)] * (2 * n + 2) + [(-0.3, 0.3)]
        res = minimize(nll, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        self.params = res.x
        self.converged = bool(res.success)
        return self

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        att, dfn, mu, h, _ = self._unpack(self.params)
        idx = {t: i for i, t in enumerate(self.teams)}
        # an unknown team is treated as a promoted side
        ah = att[idx[home]] if home in idx else self.cfg.promoted_attack
        dh = dfn[idx[home]] if home in idx else self.cfg.promoted_defence
        aa = att[idx[away]] if away in idx else self.cfg.promoted_attack
        da = dfn[idx[away]] if away in idx else self.cfg.promoted_defence
        return float(np.exp(mu + h + ah + da)), float(np.exp(mu + aa + dh))

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lam, mua = self.expected_goals(home, away)
        rho = self.params[-1]
        g = np.arange(self.cfg.max_goals + 1)
        mat = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mua))
        mat[0, 0] *= 1 - lam * mua * rho
        mat[0, 1] *= 1 + lam * rho
        mat[1, 0] *= 1 + mua * rho
        mat[1, 1] *= 1 - rho
        return mat / mat.sum()

    def predict_hda(self, home: str, away: str) -> np.ndarray:
        mat = self.score_matrix(home, away)
        return np.array([np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()])


def season_teams(df: pd.DataFrame) -> dict[int, set[str]]:
    return {s: set(g.home) | set(g.away) for s, g in df.groupby("season")}


def promoted_in(df: pd.DataFrame, season: int) -> set[str]:
    teams = season_teams(df)
    prev = teams.get(season - 1, set())
    return teams.get(season, set()) - prev if prev else set()


def walk_forward(df: pd.DataFrame, test_seasons: list[int], cfg: DCConfig | None = None) -> pd.DataFrame:
    """Refit once per calendar week, predict that week's matches."""
    out = []
    model = DixonColes(cfg)
    n_fits = n_failed = 0
    for s in test_seasons:
        test = df[df.season == s].copy()
        promoted = promoted_in(df, s)
        teams = sorted(set(test.home) | set(test.away))
        week = test.date.dt.to_period("W")
        probs = np.zeros((len(test), 3))
        for wk in week.unique():
            mask = (week == wk).to_numpy()
            as_of = test.date[mask].min()
            # promoted prior only matters until they have played some games
            played = df[(df.season == s) & (df.date < as_of)]
            n_played = pd.concat([played.home, played.away]).value_counts()
            still_new = {t for t in promoted if n_played.get(t, 0) < 10}
            model.fit(df, as_of, promoted=still_new, teams=teams)
            n_fits += 1
            n_failed += not model.converged
            sub = test[mask]
            probs[mask] = [model.predict_hda(h, a) for h, a in zip(sub.home, sub.away)]
        test[["p_h", "p_d", "p_a"]] = probs
        out.append(test)
    if n_failed:  # one summary line, not one per fit
        print(f"  ! Dixon-Coles: {n_failed} of {n_fits} weekly fits did not converge")
    return pd.concat(out)
