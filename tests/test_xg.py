import json

import pandas as pd
import pytest

from src import xg


def _match(mid, home, away, hxg, axg, is_result=True, dt="2020-09-12 16:00:00"):
    return {"id": str(mid), "isResult": is_result, "h": {"id": "1", "title": home, "short_title": "H"},
            "a": {"id": "2", "title": away, "short_title": "A"},
            "goals": {"h": "1", "a": "0"}, "xG": {"h": str(hxg), "a": str(axg)}, "datetime": dt}


MATCHES = [
    _match(1, "Atlético Madrid", "Athletic Club", 1.75, 0.62),
    _match(2, "Deportivo La Coruña", "Real Betis", 0.9, 1.4),
    _match(3, "Barcelona", "Rayo Vallecano", 2.8, 0.3, is_result=False),
]


def _hex_escape(s: str) -> str:
    """Understat style: every non-alphanumeric ASCII char as \\xNN."""
    return "".join(c if (c.isalnum() and ord(c) < 128) or c == " " else
                   (f"\\x{ord(c):02X}" if ord(c) < 128 else c) for c in s)


def _html(matches, ensure_ascii=True) -> bytes:
    js = _hex_escape(json.dumps(matches, ensure_ascii=ensure_ascii))
    page = ("<html><script>\n var teamsData = JSON.parse('\\x7B\\x7D');\n"
            f"  var datesData\t= JSON.parse('{js}');\n</script></html>")
    return page.encode("utf-8")


def test_parse_json_api_format():
    payload = json.dumps({"dates": MATCHES, "teams": {}, "players": []}).encode()
    out = xg.parse_matches(payload)
    assert [m["id"] for m in out] == ["1", "2", "3"]
    assert out[0]["h"]["title"] == "Atlético Madrid"


def test_parse_json_list_and_str_payload():
    out = xg.parse_matches("  " + json.dumps(MATCHES))
    assert len(out) == 3


def test_parse_html_hex_escaped_format():
    out = xg.parse_matches(_html(MATCHES))
    assert out == json.loads(json.dumps(MATCHES))
    assert out[1]["h"]["title"] == "Deportivo La Coruña"


def test_parse_html_with_raw_utf8_team_names():
    """If the page carries non-ASCII characters literally (UTF-8) inside the
    escaped JSON string, names must not come out as mojibake ("AtlÃ©tico")."""
    out = xg.parse_matches(_html(MATCHES, ensure_ascii=False))
    assert out[0]["h"]["title"] == "Atlético Madrid"
    assert xg.to_fd_name(out[1]["h"]["title"]) == "La Coruna"


def test_parse_garbage_raises():
    with pytest.raises(ValueError):
        xg.parse_matches(b"<html>maintenance</html>")


@pytest.mark.parametrize("name,expected", [
    ("Atlético Madrid", "Ath Madrid"),
    ("Atletico Madrid", "Ath Madrid"),
    ("Deportivo La Coruña", "La Coruna"),
    ("Athletic Club", "Ath Bilbao"),
    ("Real Betis", "Betis"),
    ("Celta Vigo", "Celta"),
    ("Espanyol", "Espanol"),
    ("Real Sociedad", "Sociedad"),
    ("Rayo Vallecano", "Vallecano"),
    ("Sporting Gijón", "Sp Gijon"),
    ("Cádiz", "Cadiz"),
    ("Alavés", "Alaves"),
    ("Leganés", "Leganes"),
    ("Real Madrid", "Real Madrid"),
    ("  Barcelona ", "Barcelona"),
])
def test_to_fd_name(name, expected):
    assert xg.to_fd_name(name) == expected


def test_to_fd_name_uses_known_names():
    assert xg.to_fd_name("GIRONA", {"Girona", "Getafe"}) == "Girona"
    assert xg.to_fd_name("Unknown FC", {"Girona"}) == "Unknown FC"


def _write_cache(raw_dir, season, matches):
    d = raw_dir / "understat"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"La_liga_{season}.json").write_text(json.dumps(matches), encoding="utf-8")


def _history():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-09-12", "2020-09-13", "2021-05-01", "2021-09-12"]),
        "season": [2020, 2020, 2020, 2021],
        "home": ["Ath Madrid", "La Coruna", "Ath Bilbao", "Ath Madrid"],
        "away": ["Ath Bilbao", "Betis", "Ath Madrid", "Ath Bilbao"],
        "hg": [1, 0, 2, 0], "ag": [0, 1, 2, 0],
    })


def test_load_skips_unplayed(tmp_path):
    _write_cache(tmp_path, 2020, MATCHES)
    df = xg.load(tmp_path)
    assert len(df) == 2
    assert df.season.tolist() == [2020, 2020]
    assert df.home.tolist() == ["Ath Madrid", "La Coruna"]


def test_merge_joins_on_season_home_away(tmp_path):
    _write_cache(tmp_path, 2020, MATCHES)
    _write_cache(tmp_path, 2021, [_match(9, "Atlético Madrid", "Athletic Club", 0.4, 0.5)])
    hist = _history()
    out = xg.merge(hist, tmp_path)
    assert len(out) == len(hist)
    pd.testing.assert_frame_equal(out[hist.columns], hist)
    assert out.hxg.tolist()[:2] == [1.75, 0.9]
    assert pd.isna(out.hxg[2])                        # reverse fixture has no xG row
    assert out.loc[3, ["hxg", "axg"]].tolist() == [0.4, 0.5]   # same pairing, other season


def test_merge_without_cache_returns_history_unchanged(tmp_path):
    hist = _history()
    out = xg.merge(hist, tmp_path)
    pd.testing.assert_frame_equal(out, hist)
    assert "hxg" not in out.columns


def test_merge_warns_about_unknown_team(tmp_path, capsys):
    _write_cache(tmp_path, 2020, MATCHES + [_match(5, "Mystery CF", "Real Betis", 1, 1)])
    xg.merge(_history(), tmp_path)
    out = capsys.readouterr().out
    assert "Mystery CF" in out and "TEAM_MAP" in out


def test_download_offline_fallback_and_cache(tmp_path, monkeypatch):
    calls = []

    def fake_get(url):
        calls.append(url)
        if "getLeagueData" in url:
            raise OSError("API down")
        return _html(MATCHES)

    monkeypatch.setattr(xg, "_get", fake_get)
    xg.download(tmp_path, first=2014, last=2015)
    assert (tmp_path / "understat" / "La_liga_2014.json").exists()
    assert len(calls) == 4      # API then page, for two seasons
    calls.clear()
    xg.download(tmp_path, first=2014, last=2015)
    assert len(calls) == 2      # finished 2014 is cached; current 2015 refreshed
    assert all("2015" in c for c in calls)


def test_download_all_down_keeps_going(tmp_path, monkeypatch, capsys):
    def fake_get(url):
        raise OSError("getaddrinfo failed")

    monkeypatch.setattr(xg, "_get", fake_get)
    xg.download(tmp_path, first=2014, last=2015)
    assert "unavailable" in capsys.readouterr().out
    assert not list((tmp_path / "understat").glob("*.json"))
