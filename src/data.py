"""Download and clean La Liga data from football-data.co.uk."""
from __future__ import annotations

import pathlib
import urllib.request

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/SP1.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (laliga-predictor)"}

# Bettable odds: (pre-match prefix, closing prefix) pairs in priority order.
# Bet odds and closing odds always come from the SAME bookmaker on a row, so
# closing line value compares like with like. Pinnacle disappears from
# football-data.co.uk after 12 Jan 2026, so later rows fall back to Bet365.
BOOK_PAIRS = [("PS", "PSC"), ("B365", "B365C"), ("Avg", "AvgC"), ("BbAv", None)]

# Benchmark closing odds: the sharpest closing market available, used only to
# judge probability quality. Betfair Exchange (BFEC, ~0.5% margin, from
# 2024/25) replaces Pinnacle once it is gone.
BENCH_CLOSE = ["PSC", "BFEC", "AvgC", "B365C"]


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _fetch(url: str, dest: pathlib.Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! failed {url}: {e}")
        return False


def download(raw_dir: pathlib.Path, first: int = 2012, last: int = 2026) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in range(first, last + 1):
        code = season_code(year)
        dest = raw_dir / f"SP1_{code}.csv"
        print(f"Downloading {year}/{year + 1} ...")
        _fetch(BASE_URL.format(code=code), dest)
    print("Downloading upcoming fixtures ...")
    _fetch(FIXTURES_URL, raw_dir / "fixtures.csv")


def _triplet(df: pd.DataFrame, prefix: str | None) -> np.ndarray:
    cols = [f"{prefix}{s}" for s in "HDA"] if prefix else []
    if not cols or not all(c in df.columns for c in cols):
        return np.full((len(df), 3), np.nan)
    # dtype=float: an empty (header-only) frame would otherwise come out as object
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _odds(df: pd.DataFrame, prefixes: list[str]) -> np.ndarray:
    """Take the first available bookmaker for each row (row-wise fallback)."""
    out = np.full((len(df), 3), np.nan)
    for p in prefixes:
        vals = _triplet(df, p)
        fill = np.isnan(out).any(axis=1) & ~np.isnan(vals).any(axis=1)
        out[fill] = vals[fill]
    return out


def _paired_odds(df: pd.DataFrame):
    """Per row, the first bookmaker with pre-match odds, plus ITS closing odds."""
    n = len(df)
    pre, close = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    book = np.full(n, None, dtype=object)
    for p_pre, p_close in BOOK_PAIRS:
        o, c = _triplet(df, p_pre), _triplet(df, p_close)
        fill = (book == None) & ~np.isnan(o).any(axis=1)  # noqa: E711
        pre[fill], close[fill], book[fill] = o[fill], c[fill], p_pre
    return pre, close, book


def read_csv(path: pathlib.Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    # fixtures.csv starts with a UTF-8 byte-order mark ("ï»¿Div" in latin-1)
    raw.columns = [c.replace("ï»¿", "").strip() for c in raw.columns]
    return raw


def clean(df: pd.DataFrame, season: int | None) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed"),
        "season": season,
        "home": df["HomeTeam"].str.strip(),
        "away": df["AwayTeam"].str.strip(),
    })
    if "FTHG" in df.columns:
        out["hg"] = pd.to_numeric(df["FTHG"], errors="coerce")
        out["ag"] = pd.to_numeric(df["FTAG"], errors="coerce")
    o, c, book = _paired_odds(df)
    bench = _odds(df, BENCH_CLOSE)
    out["book"] = book
    for i, s in enumerate("hda"):
        out[f"odds_{s}"] = o[:, i]
        out[f"close_{s}"] = c[:, i]
        out[f"bench_{s}"] = bench[:, i]
    return out.reset_index(drop=True)


def load_history(raw_dir: pathlib.Path) -> pd.DataFrame:
    frames = []
    for f in sorted(raw_dir.glob("SP1_*.csv")):
        code = f.stem.split("_")[1]
        start_year = 2000 + int(code[:2])
        frame = clean(read_csv(f), start_year)
        if len(frame):  # a new season's file can be header-only
            frames.append(frame)
    if not frames:
        raise SystemExit("No data found. Run: python run.py download")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["hg", "ag"]).sort_values("date", kind="stable").reset_index(drop=True)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    return df


def load_fixtures(raw_dir: pathlib.Path) -> pd.DataFrame:
    f = raw_dir / "fixtures.csv"
    if not f.exists():
        return pd.DataFrame()
    raw = read_csv(f)
    if "Div" not in raw.columns:
        return pd.DataFrame()
    raw = raw[raw["Div"] == "SP1"]
    return clean(raw, None) if len(raw) else pd.DataFrame()


def fair_probs(odds: np.ndarray) -> np.ndarray:
    """Remove the bookmaker margin by normalising implied probabilities."""
    implied = 1.0 / odds
    return implied / implied.sum(axis=1, keepdims=True)
