"""End-to-end tests of run.py commands on a temporary synthetic data folder."""
import argparse

import numpy as np
import pandas as pd
import pytest

import run
import synth
from src import data, dixon_coles

ROUND_PLAYED = synth.CURRENT_PLAYED_ROUNDS        # last round already in SP1_2324.csv
ROUND_NEXT = ROUND_PLAYED + 1


@pytest.fixture
def env(tmp_path, raw_copy, monkeypatch):
    monkeypatch.setattr(run, "RAW", raw_copy)
    monkeypatch.setattr(run, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(run, "DOCS_IMG", tmp_path / "docs" / "img")
    monkeypatch.setattr(run, "today", lambda: pd.Timestamp("2000-01-01"))  # synthetic dates are past
    return tmp_path


def write_fixtures(raw_dir, rounds=(ROUND_PLAYED, ROUND_NEXT), shift=0):
    """fixtures.csv like football-data's: this matchweek's played games, the next
    round (optionally moved by `shift` days) and a game from another league."""
    frames = [synth.fixtures_frame([r], date_shift_days=shift if r == ROUND_NEXT else 0)
              for r in rounds]
    frames.append(pd.DataFrame([{"Div": "E0", "Date": "01/01/2024", "Time": "15:00",
                                 "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
                                 "B365H": 2.0, "B365D": 3.4, "B365A": 3.8}]))
    synth.write_csv(pd.concat(frames, ignore_index=True), raw_dir / "fixtures.csv", bom=True)


def predict(model="dixon_coles", min_edge=0.05):
    run.cmd_predict(argparse.Namespace(model=model, min_edge=min_edge))


def test_evaluate_writes_reports(env):
    run.cmd_evaluate(argparse.Namespace(models=["logistic", "dixon_coles"], test_seasons=2, model=None))
    rep = env / "reports"
    for f in ["model_comparison.csv", "probability_metrics.csv", "backtest_summary.csv",
              "calibration.csv", "predictions.csv", "model_comparison.png", "season_logloss.png",
              "calibration.png", "profit_curve.png"]:
        assert (rep / f).exists(), f
    for f in ["model_comparison.png", "profit_curve.png"]:
        assert (env / "docs" / "img" / f).exists()
    comp = pd.read_csv(rep / "model_comparison.csv")
    assert set(comp.model) == {"logistic", "dixon_coles"}
    assert comp.logloss.is_monotonic_increasing
    pred = pd.read_csv(rep / "predictions.csv")
    assert set(pred.season) == {2022, 2023}
    np.testing.assert_allclose(pred[["p_h", "p_d", "p_a"]].sum(axis=1), 1.0)
    metrics = pd.read_csv(rep / "probability_metrics.csv")
    assert set(metrics.season) == {"2022/23", "2023/24"}
    bt = pd.read_csv(rep / "backtest_summary.csv")
    main = comp.model.iloc[0]
    assert len(bt[bt.model == main]) == 2 * len(run.EDGES)


def test_evaluate_model_not_in_models_list(env):
    """`--models logistic --model dixon_coles` must not crash with a KeyError
    (either evaluate the requested main model too or exit with a clear message)."""
    try:
        run.cmd_evaluate(argparse.Namespace(models=["logistic"], test_seasons=2, model="dixon_coles"))
    except SystemExit:
        pass


def test_predict_excludes_played_fixtures_and_logs_once(env):
    write_fixtures(run.RAW)
    predict()
    up = pd.read_csv(env / "reports" / "upcoming.csv", parse_dates=["date"])
    nxt = synth.fixtures_frame([ROUND_NEXT])
    assert len(up) == len(nxt)
    assert set(zip(up.home, up.away)) == set(zip(nxt.HomeTeam, nxt.AwayTeam))
    assert (up.book == "B365").all()
    np.testing.assert_allclose(up[["p_h", "p_d", "p_a"]].sum(axis=1), 1.0)
    log = pd.read_csv(env / "reports" / run.PAPER_LOG)
    assert len(log) == len(nxt)      # odds are 15% generous -> every game is a value bet
    assert list(log.columns) == ["logged_at", "model", "date", "home", "away", "book", "pick",
                                 "p_model", "bet_odds", "ev"]
    assert (log.ev > 0.05).all()
    # rerun: nothing new is logged
    predict()
    assert len(pd.read_csv(env / "reports" / run.PAPER_LOG)) == len(nxt)


def test_predict_does_not_relog_rescheduled_fixture(env):
    """A game moved to another date (fixtures.csv updated) is still the same game:
    it must not be paper-traded twice (settle would then count it twice)."""
    write_fixtures(run.RAW)
    predict()
    n = len(pd.read_csv(env / "reports" / run.PAPER_LOG))
    write_fixtures(run.RAW, shift=2)
    predict()
    log = pd.read_csv(env / "reports" / run.PAPER_LOG)
    assert len(log) == n
    assert not log.duplicated(["home", "away"]).any()


def test_predict_tabular_model(env):
    write_fixtures(run.RAW)
    predict(model="logistic", min_edge=0.5)
    up = pd.read_csv(env / "reports" / "upcoming.csv")
    assert (up.model == "logistic").all() and len(up) == 4


def test_predict_unavailable_model(env):
    write_fixtures(run.RAW)
    with pytest.raises(SystemExit):
        predict(model="logistic_xg")


def test_predict_without_fixtures(env, capsys):
    (run.RAW / "fixtures.csv").unlink(missing_ok=True)
    predict()
    assert "No upcoming" in capsys.readouterr().out
    write_fixtures(run.RAW, rounds=(ROUND_PLAYED,))          # only already-played games
    predict()
    assert "No upcoming" in capsys.readouterr().out


def _season_teams_fit(df, fx, extra_teams=()):
    season = df.season.max()
    teams = sorted(set(df[df.season == season].home) | set(df[df.season == season].away) | set(extra_teams))
    m = dixon_coles.DixonColes()
    m.fit(df, pd.Timestamp(fx.date.min()), promoted=set(), teams=teams)
    return m


@pytest.mark.parametrize("team,date", [("NEWCO", "2023-10-08"),    # never seen, mid-season
                                       ("T09", "2024-08-24")])     # absent last season, next season
def test_fit_for_prediction_treats_new_team_as_promoted(env, team, date):
    """run.py warns that unseen fixture teams are 'treated as promoted', and
    DixonColes.expected_goals has a promoted fallback, but fit_for_prediction
    puts the team in `teams` with a 0 prior, so it is predicted as an average side."""
    df, _ = run.load_dataset()
    fx = pd.DataFrame({"date": pd.to_datetime([date]), "home": ["T01"], "away": [team]})
    got = run.fit_for_prediction("dixon_coles", df, fx).iloc[0].to_numpy()
    expected = _season_teams_fit(df, fx).predict_hda("T01", team)      # promoted fallback
    np.testing.assert_allclose(got, expected, atol=2e-3)


def _append_results(raw_dir):
    """Round ROUND_NEXT gets played: one game rescheduled +3 days, one priced only
    by the market average (no B365), one not played yet."""
    f = raw_dir / f"SP1_{synth.season_code(synth.CURRENT_SEASON)}.csv"
    hist = data.read_csv(f)
    fx = synth.fixtures_frame([ROUND_NEXT])
    rows = []
    for i, r in enumerate(fx.itertuples()):
        if i == 3:
            continue                                      # still to be played
        d = pd.to_datetime(r.Date, dayfirst=True) + pd.Timedelta(days=3 if i == 0 else 0)
        row = {"Div": "SP1", "Date": d.strftime("%d/%m/%Y"), "Time": "20:00",
               "HomeTeam": r.HomeTeam, "AwayTeam": r.AwayTeam, "FTHG": 1, "FTAG": 1, "FTR": "D",
               "B365H": r.B365H, "B365D": r.B365D, "B365A": r.B365A,
               "B365CH": r.B365H * 0.9, "B365CD": r.B365D * 0.9, "B365CA": r.B365A * 0.9,
               "AvgH": r.AvgH, "AvgD": r.AvgD, "AvgA": r.AvgA,
               "AvgCH": r.AvgH, "AvgCD": r.AvgD, "AvgCA": r.AvgA}
        if i == 2:
            row.update({"B365H": np.nan, "B365D": np.nan, "B365A": np.nan})
        rows.append(row)
    synth.write_csv(pd.concat([hist, pd.DataFrame(rows)], ignore_index=True), f, bom=True)
    return fx


def test_settle_end_to_end(env):
    write_fixtures(run.RAW)
    predict()
    log = pd.read_csv(env / "reports" / run.PAPER_LOG)
    fx = _append_results(run.RAW)
    run.cmd_settle(argparse.Namespace())
    out = pd.read_csv(env / "reports" / "paper_trades_settled.csv")
    assert len(out) == len(log)
    by_game = out.set_index("home")
    h = list(fx.HomeTeam)
    assert by_game.loc[h[3], "status"] == "open"
    for i in range(3):
        r = by_game.loc[h[i]]
        assert r.status == ("won" if r.pick == "d" else "lost")
    # B365 bet, B365 close of 0.9 x bet odds -> CLV = 1/0.9 - 1
    assert by_game.loc[h[0], "clv"] == pytest.approx(1 / 0.9 - 1, rel=1e-2)
    assert by_game.loc[h[1], "clv"] == pytest.approx(1 / 0.9 - 1, rel=1e-2)
    # history row priced by Avg only -> no like-for-like close
    assert np.isnan(by_game.loc[h[2], "clv"])


def test_settle_without_log(env, capsys):
    run.cmd_settle(argparse.Namespace())
    assert "No paper trades" in capsys.readouterr().out


def test_predict_clears_stale_upcoming(env):
    """When every fixture in fixtures.csv has been played, upcoming.csv must not
    keep showing last week's games as 'upcoming' in the dashboard."""
    write_fixtures(run.RAW)
    predict()
    assert len(pd.read_csv(env / "reports" / "upcoming.csv")) == 4
    write_fixtures(run.RAW, rounds=(ROUND_PLAYED,))     # only already-played games now
    predict()
    up = env / "reports" / "upcoming.csv"
    assert not up.exists() or pd.read_csv(up).empty


def test_evaluate_xg_model_without_xg_data(env):
    """`--models logistic_xg` with no Understat data must skip the model or exit
    with a clear message, not crash with an IndexError in train_rows."""
    try:
        run.cmd_evaluate(argparse.Namespace(models=["logistic", "logistic_xg"], test_seasons=2, model=None))
    except SystemExit:
        pass


def test_predict_does_not_log_games_already_kicked_off(env, monkeypatch):
    """Fixtures dated before today are shown but never paper-traded."""
    write_fixtures(run.RAW)
    monkeypatch.setattr(run, "today", lambda: pd.Timestamp("2030-01-01"))
    predict()
    assert len(pd.read_csv(env / "reports" / "upcoming.csv")) == 4
    assert not (env / "reports" / run.PAPER_LOG).exists()


def test_evaluate_main_model_is_added_to_models(env):
    run.cmd_evaluate(argparse.Namespace(models=["logistic"], test_seasons=2, model="dixon_coles"))
    comp = pd.read_csv(env / "reports" / "model_comparison.csv")
    assert set(comp.model) == {"logistic", "dixon_coles"}
