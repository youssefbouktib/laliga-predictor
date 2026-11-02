"""Pre-match features: Elo ratings, recent form and (optionally) rolling xG.

Everything is computed in one chronological pass, and each match only sees
information from matches played *before* it. This is what prevents leakage:
for every match we first take a snapshot of the current state (that snapshot
is the feature row), and only then update the state with the result.
"""
from __future__ import annotations

import copy
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

FEATURES = [
    "elo_diff",
    "home_gf5", "home_ga5", "home_pts5",
    "away_gf5", "away_ga5", "away_pts5",
]
XG_FEATURES = ["home_xgf", "home_xga", "away_xgf", "away_xga"]
# rolling shots / shots on target for and against: a free xG substitute
SHOT_FEATURES = ["home_shf", "home_sha", "home_sotf", "home_sota",
                 "away_shf", "away_sha", "away_sotf", "away_sota"]
SNAPSHOT_COLS = ["elo_home", "elo_away"] + FEATURES + XG_FEATURES + SHOT_FEATURES


@dataclass
class EloConfig:
    k: float = 20.0
    home_adv: float = 60.0
    init: float = 1500.0
    promoted_init: float = 1420.0   # newly promoted teams are usually weaker
    season_regress: float = 0.2     # pull ratings toward the mean each summer
    form_window: int = 5
    xg_window: int = 8              # xG is noisy per match, so average more games
    xg_default: float = 1.25        # league-average xG per team per match
    shot_window: int = 8
    shots_default: float = 12.0     # league-average shots per team per match
    sot_default: float = 4.2        # league-average shots on target


@dataclass
class TeamState:
    ratings: dict = field(default_factory=dict)
    form: dict = field(default_factory=dict)
    xg: dict = field(default_factory=dict)
    shots: dict = field(default_factory=dict)
    season: int | None = None        # season of the last match seen
    first_season: int | None = None  # warm-up season: nobody counts as promoted
    prev_teams: set = field(default_factory=set)  # teams in last season's league
    cur_teams: set = field(default_factory=set)   # teams seen so far this season


def _goal_mult(gd: int) -> float:
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


def _form_feats(hist: deque) -> tuple[float, float, float]:
    if not hist:
        return 1.3, 1.3, 1.4  # league-average defaults
    n = len(hist)
    return (sum(h[0] for h in hist) / n,
            sum(h[1] for h in hist) / n,
            sum(h[2] for h in hist) / n)


def _xg_feats(hist: deque, default: float) -> tuple[float, float]:
    if not hist:
        return default, default
    n = len(hist)
    return sum(h[0] for h in hist) / n, sum(h[1] for h in hist) / n


def _shot_feats(hist: deque, cfg: "EloConfig") -> tuple[float, float, float, float]:
    """Average (shots for, shots against, on target for, on target against)."""
    if not hist:
        return cfg.shots_default, cfg.shots_default, cfg.sot_default, cfg.sot_default
    n = len(hist)
    return tuple(sum(float(h[i]) for h in hist) / n for i in range(4))


def new_state(cfg: EloConfig) -> TeamState:
    st = TeamState()
    st.form = defaultdict(lambda: deque(maxlen=cfg.form_window))
    st.xg = defaultdict(lambda: deque(maxlen=cfg.xg_window))
    st.shots = defaultdict(lambda: deque(maxlen=cfg.shot_window))
    return st


def season_of(date) -> int:
    """La Liga seasons start in August: Aug 2024 - May 2025 is season 2024."""
    d = pd.Timestamp(date)
    return d.year if d.month >= 7 else d.year - 1


def _enter_season(st: TeamState, season: int, cfg: EloConfig) -> None:
    """Called before the first match of a new season (summer break)."""
    if st.first_season is None:
        st.first_season = season
    # regress existing ratings toward the mean
    for t in st.ratings:
        st.ratings[t] += cfg.season_regress * (cfg.init - st.ratings[t])
    st.season = season
    st.prev_teams, st.cur_teams = st.cur_teams, set()


def _first_game(st: TeamState, team: str, cfg: EloConfig) -> None:
    """Called the first time `team` appears in the current season."""
    if team in st.cur_teams:
        return
    st.cur_teams.add(team)
    if st.season == st.first_season:
        st.ratings.setdefault(team, cfg.init)
    elif team not in st.prev_teams:
        # promoted (possibly returning after years away): reset rating
        # and drop stale form from a previous top-flight spell
        st.ratings[team] = cfg.promoted_init
        st.form[team].clear()
        st.xg[team].clear()
        st.shots[team].clear()


def build_features(df: pd.DataFrame, cfg: EloConfig | None = None):
    """Return (df with features, final TeamState for predicting new games).

    `df` must be sorted by date. If it has `hxg`/`axg` columns (Understat),
    rolling xG features are added; matches without xG simply don't update the
    xG history.
    """
    cfg = cfg or EloConfig()
    st = new_state(cfg)
    has_xg = {"hxg", "axg"}.issubset(df.columns)
    shot_cols = ["h_shots", "a_shots", "h_sot", "a_sot"]
    # clean() always creates these columns (NaN when the source has none), so
    # require real values, not just the columns
    has_shots = set(shot_cols).issubset(df.columns) and df[shot_cols].notna().any().all()
    rows = []

    for m in df.itertuples(index=False):
        if m.season != st.season:
            _enter_season(st, m.season, cfg)
        _first_game(st, m.home, cfg)
        _first_game(st, m.away, cfg)

        rows.append(snapshot(st, m.home, m.away, cfg))
        update(st, m.home, m.away, int(m.hg), int(m.ag), cfg)
        if has_xg and not (pd.isna(m.hxg) or pd.isna(m.axg)):
            st.xg[m.home].append((float(m.hxg), float(m.axg)))
            st.xg[m.away].append((float(m.axg), float(m.hxg)))
        if has_shots and not any(pd.isna(v) for v in (m.h_shots, m.a_shots, m.h_sot, m.a_sot)):
            st.shots[m.home].append((m.h_shots, m.a_shots, m.h_sot, m.a_sot))
            st.shots[m.away].append((m.a_shots, m.h_shots, m.a_sot, m.h_sot))

    # columns= keeps an empty input working (no rows -> no keys to infer from)
    feats = pd.DataFrame(rows, index=df.index, columns=SNAPSHOT_COLS)
    out = pd.concat([df, feats], axis=1)
    if not has_xg:
        out = out.drop(columns=XG_FEATURES)
    if not has_shots:
        out = out.drop(columns=SHOT_FEATURES)
    return out, st


def snapshot(st: TeamState, home: str, away: str, cfg: EloConfig) -> dict:
    rh = st.ratings.get(home, cfg.promoted_init)
    ra = st.ratings.get(away, cfg.promoted_init)
    hgf, hga, hpts = _form_feats(st.form[home])
    agf, aga, apts = _form_feats(st.form[away])
    hxf, hxa = _xg_feats(st.xg[home], cfg.xg_default)
    axf, axa = _xg_feats(st.xg[away], cfg.xg_default)
    hs, as_ = _shot_feats(st.shots[home], cfg), _shot_feats(st.shots[away], cfg)
    return {
        "elo_home": rh, "elo_away": ra,
        "elo_diff": rh + cfg.home_adv - ra,
        "home_gf5": hgf, "home_ga5": hga, "home_pts5": hpts,
        "away_gf5": agf, "away_ga5": aga, "away_pts5": apts,
        "home_xgf": hxf, "home_xga": hxa, "away_xgf": axf, "away_xga": axa,
        "home_shf": hs[0], "home_sha": hs[1], "home_sotf": hs[2], "home_sota": hs[3],
        "away_shf": as_[0], "away_sha": as_[1], "away_sotf": as_[2], "away_sota": as_[3],
    }


def update(st: TeamState, home: str, away: str, hg: int, ag: int, cfg: EloConfig) -> None:
    rh, ra = st.ratings[home], st.ratings[away]
    exp_h = 1.0 / (1.0 + 10 ** ((ra - rh - cfg.home_adv) / 400))
    score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    delta = cfg.k * _goal_mult(hg - ag) * (score - exp_h)
    st.ratings[home] = rh + delta
    st.ratings[away] = ra - delta
    hp = 3 if hg > ag else 1 if hg == ag else 0
    ap = 3 if ag > hg else 1 if hg == ag else 0
    st.form[home].append((hg, ag, hp))
    st.form[away].append((ag, hg, ap))


def predict_snapshots(st: TeamState, fixtures: pd.DataFrame, cfg: EloConfig | None = None) -> pd.DataFrame:
    """Feature rows for unplayed fixtures, from the state after the last game.

    A fixture of a season not in the history yet gets the same summer
    treatment build_features would give it (rating regression, promoted
    reset). This works on a copy, so `st` itself is never changed.
    """
    cfg = cfg or EloConfig()
    rows = []
    for d, h, a in zip(fixtures.date, fixtures.home, fixtures.away):
        s = copy.deepcopy(st)
        season = season_of(d)
        if s.season is not None and season > s.season:
            _enter_season(s, season, cfg)
        _first_game(s, h, cfg)
        _first_game(s, a, cfg)
        rows.append(snapshot(s, h, a, cfg))
    return pd.DataFrame(rows, index=fixtures.index, columns=SNAPSHOT_COLS)

