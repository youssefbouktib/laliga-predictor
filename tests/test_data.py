import numpy as np
import pandas as pd
import pytest

from src import data


def test_season_code():
    assert data.season_code(2012) == "1213"
    assert data.season_code(2009) == "0910"
    assert data.season_code(1999) == "9900"
    assert data.season_code(2026) == "2627"


def test_read_csv_strips_utf8_bom(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(b"\xef\xbb\xbfDiv,Date,HomeTeam,AwayTeam\nSP1,01/09/2024,A,B\n")
    raw = data.read_csv(f)
    assert list(raw.columns) == ["Div", "Date", "HomeTeam", "AwayTeam"]
    assert raw.loc[0, "Div"] == "SP1"


def _row(**kw):
    base = {"Date": "01/09/2024", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0}
    base.update(kw)
    return base


def _trip(prefix, vals):
    return {f"{prefix}{s}": v for s, v in zip("HDA", vals)}


def test_paired_odds_fall_back_to_same_book_close():
    raw = pd.DataFrame([
        # 0: Pinnacle pre + close
        _row(**_trip("PS", (2.0, 3.4, 4.0)), **_trip("PSC", (1.9, 3.5, 4.2)),
             **_trip("B365", (1.95, 3.3, 3.9)), **_trip("B365C", (1.85, 3.4, 4.1))),
        # 1: Pinnacle pre missing (but a PSC close exists): B365 pre AND B365C close
        _row(**_trip("PS", (np.nan,) * 3), **_trip("PSC", (1.7, 3.6, 5.0)),
             **_trip("B365", (2.1, 3.2, 3.7)), **_trip("B365C", (2.2, 3.1, 3.5))),
        # 2: partial Pinnacle triplet -> also falls back
        _row(**_trip("PS", (2.0, np.nan, 4.0)), **_trip("PSC", (1.9, 3.5, 4.2)),
             **_trip("B365", (2.05, 3.25, 3.8)), **_trip("B365C", (2.0, 3.3, 3.9))),
        # 3: only market averages
        _row(**_trip("PS", (np.nan,) * 3), **_trip("Avg", (2.2, 3.3, 3.4)),
             **_trip("AvgC", (2.3, 3.2, 3.3))),
        # 4: no odds at all
        _row(),
    ])
    df = data.clean(raw, 2024)
    assert list(df.book[:4]) == ["PS", "B365", "B365", "Avg"]
    assert pd.isna(df.book[4])
    np.testing.assert_allclose(df.loc[0, ["odds_h", "odds_d", "odds_a"]].to_numpy(float), [2.0, 3.4, 4.0])
    np.testing.assert_allclose(df.loc[0, ["close_h", "close_d", "close_a"]].to_numpy(float), [1.9, 3.5, 4.2])
    np.testing.assert_allclose(df.loc[1, ["odds_h", "odds_d", "odds_a"]].to_numpy(float), [2.1, 3.2, 3.7])
    np.testing.assert_allclose(df.loc[1, ["close_h", "close_d", "close_a"]].to_numpy(float), [2.2, 3.1, 3.5])
    np.testing.assert_allclose(df.loc[2, ["close_h", "close_d", "close_a"]].to_numpy(float), [2.0, 3.3, 3.9])
    np.testing.assert_allclose(df.loc[3, ["close_h", "close_d", "close_a"]].to_numpy(float), [2.3, 3.2, 3.3])
    assert df.loc[4, ["odds_h", "close_h"]].isna().all()


def test_bbav_book_has_no_close():
    raw = pd.DataFrame([_row(**_trip("BbAv", (2.0, 3.0, 4.0)))])
    df = data.clean(raw, 2012)
    assert df.book[0] == "BbAv"
    assert df[["close_h", "close_d", "close_a"]].isna().all(axis=None)


def test_bench_priority():
    raw = pd.DataFrame([
        _row(**_trip("PSC", (1.5, 4, 6)), **_trip("BFEC", (1.6, 4, 6)), **_trip("AvgC", (1.7, 4, 6)),
             **_trip("B365C", (1.8, 4, 6))),
        _row(**_trip("PSC", (np.nan,) * 3), **_trip("BFEC", (1.6, 4, 6)), **_trip("AvgC", (1.7, 4, 6)),
             **_trip("B365C", (1.8, 4, 6))),
        _row(**_trip("PSC", (np.nan,) * 3), **_trip("BFEC", (np.nan,) * 3), **_trip("AvgC", (1.7, 4, 6)),
             **_trip("B365C", (1.8, 4, 6))),
        _row(**_trip("PSC", (np.nan,) * 3), **_trip("BFEC", (np.nan,) * 3), **_trip("AvgC", (np.nan,) * 3),
             **_trip("B365C", (1.8, 4, 6))),
    ])
    df = data.clean(raw, 2024)
    assert df.bench_h.tolist() == [1.5, 1.6, 1.7, 1.8]


def test_fair_probs_sum_to_one():
    odds = np.array([[2.0, 3.4, 4.0], [1.2, 6.5, 13.0]])
    p = data.fair_probs(odds)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    implied = 1 / odds
    np.testing.assert_allclose(p, implied / implied.sum(axis=1, keepdims=True))
    assert (p[:, 0] > p[:, 2]).all()


def test_date_parsing_two_and_four_digit_years():
    raw = pd.DataFrame([_row(Date="18/08/12"), _row(Date="01/02/2013"), _row(Date="13/01/26")])
    df = data.clean(raw, 2012)
    assert df.date.tolist() == [pd.Timestamp("2012-08-18"), pd.Timestamp("2013-02-01"),
                                pd.Timestamp("2026-01-13")]


def test_load_fixtures_without_div_column(tmp_path):
    (tmp_path / "fixtures.csv").write_text("Date,HomeTeam,AwayTeam\n01/09/2024,A,B\n")
    assert data.load_fixtures(tmp_path).empty


def test_load_fixtures_missing_file(tmp_path):
    assert data.load_fixtures(tmp_path).empty


def test_load_fixtures_filters_league_and_strips_bom(tmp_path):
    (tmp_path / "fixtures.csv").write_bytes(
        b"\xef\xbb\xbfDiv,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
        b"E0,01/09/2024,15:00,Arsenal,Chelsea,2,3,4\n"
        b"SP1,01/09/2024,20:00,Barcelona,Getafe,1.3,5.5,9\n")
    fx = data.load_fixtures(tmp_path)
    assert len(fx) == 1
    assert fx.home[0] == "Barcelona" and fx.book[0] == "B365"
    assert fx.date[0] == pd.Timestamp("2024-09-01")
    assert "hg" not in fx.columns


def test_load_history_seasons_results_and_order(history):
    assert sorted(history.season.unique()) == list(range(2017, 2024))
    assert history.date.is_monotonic_increasing
    exp = np.where(history.hg > history.ag, "H", np.where(history.hg == history.ag, "D", "A"))
    assert (history.result.to_numpy() == exp).all()
    # 2-digit year files (2017/18) parsed into the right century
    assert history[history.season == 2017].date.dt.year.isin([2017, 2018]).all()
    # odds pairing: Pinnacle disappears mid-2022 -> B365
    assert set(history.book.dropna()) == {"PS", "B365"}


def test_load_history_no_files(tmp_path):
    with pytest.raises(SystemExit):
        data.load_history(tmp_path)


def test_load_history_tolerates_header_only_season_file(tmp_path, league_dir):
    """A season file with only the header row (new season, nothing played yet)
    must not break loading the rest of the history."""
    import shutil
    for f in league_dir.glob("SP1_*.csv"):
        shutil.copy(f, tmp_path / f.name)
    header = (league_dir / "SP1_2324.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    (tmp_path / "SP1_2425.csv").write_text(header + "\n", encoding="utf-8")
    h = data.load_history(tmp_path)
    assert h.season.max() == 2023 and len(h) == len(data.load_history(league_dir))
