"""Expected goals (xG) from Understat, merged onto football-data.co.uk matches.

Understat has La Liga xG from 2014/15. Each season is cached as JSON in
data/raw/understat/, so the site is only hit once per season (the current
season is refreshed on every download).

If Understat is unreachable and nothing is cached, the project still runs:
the xG columns are simply absent and xG models are skipped.
"""
from __future__ import annotations

import json
import pathlib
import re
import unicodedata
import urllib.request

import pandas as pd

FIRST_SEASON = 2014
PAGE_URL = "https://understat.com/league/La_liga/{season}"
API_URL = "https://understat.com/getLeagueData/La_liga/{season}"
HEADERS = {"User-Agent": "Mozilla/5.0 (laliga-predictor)", "X-Requested-With": "XMLHttpRequest"}

# Understat team name -> football-data.co.uk team name. Names that are already
# identical after normalisation (Barcelona, Getafe, ...) don't need an entry.
TEAM_MAP = {
    "athletic club": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao",
    "atletico madrid": "Ath Madrid",
    "real betis": "Betis",
    "celta vigo": "Celta",
    "espanyol": "Espanol",
    "deportivo la coruna": "La Coruna",
    "deportivo": "La Coruna",
    "real sociedad": "Sociedad",
    "sporting gijon": "Sp Gijon",
    "rayo vallecano": "Vallecano",
    "sd huesca": "Huesca",
    "real valladolid": "Valladolid",
    "racing santander": "Santander",
    "real oviedo": "Oviedo",
    "real zaragoza": "Zaragoza",
    "las palmas": "Las Palmas",
    "ud las palmas": "Las Palmas",
    "cadiz": "Cadiz",
    "leganes": "Leganes",
    "malaga": "Malaga",
    "alaves": "Alaves",
    "almeria": "Almeria",
    "cordoba": "Cordoba",
    "real madrid": "Real Madrid",
}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


def to_fd_name(name: str, known: set[str] | None = None) -> str:
    """Map an Understat team name to the football-data.co.uk spelling."""
    key = _norm(name)
    if key in TEAM_MAP:
        return TEAM_MAP[key]
    if known:
        for k in known:
            if _norm(k) == key:
                return k
    return str(name).strip()


# ---------------------------------------------------------------- download

def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def parse_matches(payload: bytes | str) -> list[dict]:
    """Extract the match list from either Understat format.

    * JSON API (getLeagueData): {"dates": [...], "teams": ..., "players": ...}
    * Older HTML page: var datesData = JSON.parse('<hex-escaped json>')
    """
    text = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else payload
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        obj = json.loads(stripped)
        return obj["dates"] if isinstance(obj, dict) else obj
    m = re.search(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", text, re.S)
    if not m:
        raise ValueError("no match data found in Understat response")
    return json.loads(_unescape_js(m.group(1)))


_JS_ESCAPE = re.compile(r"\\(x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|.)", re.S)
_JS_SIMPLE = {"n": "\n", "t": "\t", "r": "\r"}


def _unescape_js(s: str) -> str:
    r"""Undo JavaScript string escapes (\xNN, \uNNNN, \n, \', \\ ...) in one pass.

    Unlike codecs' "unicode_escape", characters that are already non-ASCII
    (e.g. a literal "é") are left alone instead of being re-read as latin-1.
    """
    def sub(m):
        e = m.group(1)
        if len(e) > 1:
            return chr(int(e[1:], 16))
        return _JS_SIMPLE.get(e, e)
    return _JS_ESCAPE.sub(sub, s)


def download(raw_dir: pathlib.Path, first: int = FIRST_SEASON, last: int = 2026) -> None:
    out = raw_dir / "understat"
    out.mkdir(parents=True, exist_ok=True)
    for season in range(max(first, FIRST_SEASON), last + 1):
        dest = out / f"La_liga_{season}.json"
        if dest.exists() and season < last:
            continue  # finished seasons never change
        matches = None
        for url in (API_URL, PAGE_URL):
            try:
                matches = parse_matches(_get(url.format(season=season)))
                break
            except Exception as e:  # noqa: BLE001
                err = e
        if matches is None:
            print(f"  ! Understat {season}/{season + 1} unavailable: {err}")
            continue
        dest.write_text(json.dumps(matches), encoding="utf-8")
        print(f"  Understat {season}/{season + 1}: {len(matches)} matches")


# ---------------------------------------------------------------- load + merge

def load(raw_dir: pathlib.Path, known_teams: set[str] | None = None) -> pd.DataFrame:
    """All cached Understat matches with a result, in football-data team names."""
    rows = []
    for f in sorted((raw_dir / "understat").glob("La_liga_*.json")):
        season = int(f.stem.rsplit("_", 1)[1])
        for m in json.loads(f.read_text(encoding="utf-8")):
            if not m.get("isResult"):
                continue
            rows.append({
                "season": season,
                "home": to_fd_name(m["h"]["title"], known_teams),
                "away": to_fd_name(m["a"]["title"], known_teams),
                "hxg": float(m["xG"]["h"]),
                "axg": float(m["xG"]["a"]),
            })
    return pd.DataFrame(rows, columns=["season", "home", "away", "hxg", "axg"])


def merge(history: pd.DataFrame, raw_dir: pathlib.Path) -> pd.DataFrame:
    """Add hxg/axg columns to the match history.

    Matches are joined on (season, home, away): each pairing happens exactly
    once per season, so this is robust to kickoff-time/date differences.
    Returns the history unchanged when no Understat data is cached.
    """
    known = set(history.home) | set(history.away)
    xg = load(raw_dir, known)
    if xg.empty:
        return history
    unknown = (set(xg.home) | set(xg.away)) - known
    if unknown:
        print(f"  ! Understat teams with no football-data match: {sorted(unknown)} "
              "(add them to xg.TEAM_MAP)")
    xg = xg.drop_duplicates(["season", "home", "away"])
    out = history.merge(xg, on=["season", "home", "away"], how="left", validate="m:1")
    covered = out[out.season.isin(xg.season.unique())]
    missing = covered.hxg.isna().sum()
    if missing:
        print(f"  ! {missing} matches in Understat seasons have no xG")
    return out
