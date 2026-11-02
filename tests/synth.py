"""Synthetic La Liga-like data in football-data.co.uk format (no network)."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from scipy.stats import poisson

OUT = "HDA"

# Season -> teams in the league. T08 is relegated after 2017 and returns in
# 2019 (a "returning" promoted team); T09 and T10 are brand-new promotions.
SEASON_TEAMS = {
    2017: [f"T{i:02d}" for i in range(1, 9)],
    2018: [f"T{i:02d}" for i in range(1, 8)] + ["T09"],
    2019: [f"T{i:02d}" for i in range(1, 7)] + ["T08", "T09"],
    2020: [f"T{i:02d}" for i in range(1, 7)] + ["T08", "T10"],
    2021: [f"T{i:02d}" for i in range(1, 7)] + ["T08", "T10"],
    2022: [f"T{i:02d}" for i in range(1, 7)] + ["T08", "T10"],
    2023: [f"T{i:02d}" for i in range(1, 7)] + ["T08", "T10"],
}
# true strengths: T01 strongest
STRENGTH = {f"T{i:02d}": 0.5 - 0.1 * (i - 1) for i in range(1, 11)}
HOME_ADV = 0.3
CURRENT_SEASON = 2023
CURRENT_PLAYED_ROUNDS = 7


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def rates(home: str, away: str, home_adv: float = HOME_ADV) -> tuple[float, float]:
    sh, sa = STRENGTH[home], STRENGTH[away]
    return float(np.exp(0.15 + home_adv + (sh - sa) / 2)), float(np.exp(0.15 + (sa - sh) / 2))


def hda_probs(lam: float, mu: float, max_goals: int = 12) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    p = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
    return p / p.sum()


def double_round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Circle method: 2*(n-1) rounds of n/2 games, each pairing once per venue."""
    t = list(teams)
    n = len(t)
    rounds = []
    for r in range(n - 1):
        games = []
        for i in range(n // 2):
            a, b = t[i], t[n - 1 - i]
            games.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(games)
        t = [t[0]] + [t[-1]] + t[1:-1]
    return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]


def schedule(season: int) -> pd.DataFrame:
    """Dates/pairings for a season: one round per week, two games Sat, two Sun."""
    start = pd.Timestamp(f"{season}-08-19")
    rows = []
    for r, games in enumerate(double_round_robin(SEASON_TEAMS[season])):
        for i, (h, a) in enumerate(games):
            rows.append({"round": r + 1, "date": start + pd.Timedelta(days=7 * r + (i >= 2)),
                         "home": h, "away": a})
    return pd.DataFrame(rows)


def _odds(p: np.ndarray, margin: float, rng, noise: float = 0.0) -> np.ndarray:
    q = p * (1 + margin)
    if noise:
        q = q * np.exp(rng.normal(0, noise, 3))
    return np.round(1 / q, 2)


def raw_season(season: int, rng, played_rounds: int | None = None) -> pd.DataFrame:
    """One SP1_xxyy.csv in football-data format."""
    sch = schedule(season)
    if played_rounds is not None:
        sch = sch[sch["round"] <= played_rounds]
    fmt = "%d/%m/%y" if season < 2019 else "%d/%m/%Y"   # 2- and 4-digit years like the real files
    rows = []
    for g in sch.itertuples():
        lam, mu = rates(g.home, g.away)
        hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu))
        p = hda_probs(lam, mu)
        row = {"Div": "SP1", "Date": g.date.strftime(fmt), "Time": "18:00",
               "HomeTeam": g.home, "AwayTeam": g.away, "FTHG": hg, "FTAG": ag,
               "FTR": "H" if hg > ag else "D" if hg == ag else "A"}
        books = {"B365": (_odds(p, 0.06, rng, 0.03), _odds(p, 0.05, rng, 0.01)),
                 "Avg": (_odds(p, 0.07, rng, 0.02), _odds(p, 0.06, rng, 0.01))}
        # Pinnacle stops half-way through 2022 (like the real data after Jan 2026)
        if season < 2022 or g.round <= 7:
            books["PS"] = (_odds(p, 0.025, rng, 0.03), _odds(p, 0.02, rng, 0.005))
        for name, (pre, close) in books.items():
            cprefix = {"PS": "PSC", "B365": "B365C", "Avg": "AvgC"}[name]
            for k, s in enumerate(OUT):
                row[f"{name}{s}"] = pre[k]
                row[f"{cprefix}{s}"] = close[k]
        if season >= 2021:
            bfe = _odds(p, 0.005, rng, 0.005)
            for k, s in enumerate(OUT):
                row[f"BFEC{s}"] = bfe[k]
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: pathlib.Path, bom: bool = False) -> None:
    text = df.to_csv(index=False, lineterminator="\n")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))


def write_league(raw_dir: pathlib.Path, seed: int = 0) -> dict[int, pd.DataFrame]:
    """All seasons to raw_dir/SP1_*.csv (current season partially played)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out = {}
    for s in SEASON_TEAMS:
        played = CURRENT_PLAYED_ROUNDS if s == CURRENT_SEASON else None
        df = raw_season(s, rng, played)
        write_csv(df, raw_dir / f"SP1_{season_code(s)}.csv", bom=s >= 2021)
        out[s] = df
    return out


def fixtures_frame(rounds: list[int], season: int = CURRENT_SEASON, edge: float = 0.15,
                   date_shift_days: int = 0) -> pd.DataFrame:
    """fixtures.csv rows (no results) with generous B365 + Avg odds."""
    sch = schedule(season)
    sch = sch[sch["round"].isin(rounds)]
    rows = []
    for g in sch.itertuples():
        p = hda_probs(*rates(g.home, g.away))
        o = np.round(1 / (p * (1 - edge)), 2)
        a = np.round(1 / (p * 1.05), 2)
        d = g.date + pd.Timedelta(days=date_shift_days)
        row = {"Div": "SP1", "Date": d.strftime("%d/%m/%Y"), "Time": "20:00",
               "HomeTeam": g.home, "AwayTeam": g.away}
        for k, s in enumerate(OUT):
            row[f"B365{s}"] = o[k]
        for k, s in enumerate(OUT):
            row[f"Avg{s}"] = a[k]
        rows.append(row)
    return pd.DataFrame(rows)


def clean_matches(n_teams: int = 10, seasons: tuple[int, ...] = (2019, 2020, 2021), seed: int = 1,
                  home_adv: float = 0.3) -> pd.DataFrame:
    """A cleaned (load_history-like) frame with only date/season/home/away/hg/ag/result."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(1, n_teams + 1)]
    strength = {t: 0.5 - 1.0 * i / (n_teams - 1) for i, t in enumerate(teams)}
    rows = []
    for s in seasons:
        start = pd.Timestamp(f"{s}-08-19")
        for r, games in enumerate(double_round_robin(teams)):
            for i, (h, a) in enumerate(games):
                lam = np.exp(0.15 + home_adv + (strength[h] - strength[a]) / 2)
                mu = np.exp(0.15 + (strength[a] - strength[h]) / 2)
                hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu))
                rows.append({"date": start + pd.Timedelta(days=7 * r + (i % 2)), "season": s,
                             "home": h, "away": a, "hg": hg, "ag": ag})
    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    return df
