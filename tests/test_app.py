"""Smoke tests of the Streamlit dashboard (app.py) with streamlit's AppTest.

app.py reads reports/ next to itself, so each test copies it into a temporary
folder together with the reports it should see.
"""
import argparse
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import run  # noqa: E402
import synth  # noqa: E402
from conftest import ROOT  # noqa: E402

TIMEOUT = 60


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    # app.read() is st.cache_data'd on the file *name*; all AppTests share one
    # process, so clear it to stop one test seeing another test's reports.
    import streamlit as st
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture(scope="module")
def full_reports(tmp_path_factory):
    """reports/ produced by evaluate + predict + settle on the synthetic league."""
    base = tmp_path_factory.mktemp("pipeline")
    raw = base / "raw"
    synth.write_league(raw)
    fx = pd.concat([synth.fixtures_frame([synth.CURRENT_PLAYED_ROUNDS + 1])], ignore_index=True)
    synth.write_csv(fx, raw / "fixtures.csv", bom=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "RAW", raw)
        mp.setattr(run, "REPORTS", base / "reports")
        mp.setattr(run, "DOCS_IMG", base / "docs" / "img")
        mp.setattr(run, "today", lambda: pd.Timestamp("2000-01-01"))  # synthetic dates are past
        run.cmd_evaluate(argparse.Namespace(models=["logistic", "dixon_coles"], test_seasons=2, model=None))
        # book price, no edge cap: the synthetic odds are 15% generous on purpose
        run.cmd_predict(argparse.Namespace(model="dixon_coles", min_edge=0.05, price="book",
                                           max_edge=None))
        run.cmd_settle(argparse.Namespace())
    return base / "reports"


def _app(tmp_path, reports=None, keep=None):
    shutil.copy(ROOT / "app.py", tmp_path / "app.py")
    dst = tmp_path / "reports"
    dst.mkdir()
    if reports is not None:
        for f in reports.iterdir():
            if keep is None or f.name in keep:
                shutil.copy(f, dst / f.name)
    return dst


def _run(path):
    at = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    at.run()
    return at


def test_app_full_reports(tmp_path, full_reports):
    assert (full_reports / "paper_trades_settled.csv").exists()
    _app(tmp_path, full_reports)
    at = _run(tmp_path / "app.py")
    assert not at.exception
    assert len(at.tabs) == 4
    labels = {m.label: m.value for m in at.metric}
    assert labels["Bets logged"] == str(len(pd.read_csv(full_reports / "paper_trades.csv")))
    assert labels["Best model"] in {"logistic", "dixon_coles"}
    at.slider[0].set_value(0.0).run()
    assert not at.exception
    at.selectbox[0].select(at.selectbox[0].options[-1]).run()
    assert not at.exception


def test_app_no_reports(tmp_path):
    _app(tmp_path)
    at = _run(tmp_path / "app.py")
    assert not at.exception
    assert at.warning and "No reports yet" in at.warning[0].value


def test_app_only_model_comparison(tmp_path, full_reports):
    _app(tmp_path, full_reports, keep={"model_comparison.csv"})
    at = _run(tmp_path / "app.py")
    assert not at.exception


def test_app_unsettled_paper_trades(tmp_path, full_reports):
    _app(tmp_path, full_reports, keep={"model_comparison.csv", "paper_trades.csv", "upcoming.csv"})
    at = _run(tmp_path / "app.py")
    assert not at.exception
    assert any("settle" in i.value for i in at.info)


def test_app_upcoming_with_missing_odds(tmp_path, full_reports):
    rep = _app(tmp_path, full_reports, keep={"model_comparison.csv", "upcoming.csv"})
    up = pd.read_csv(rep / "upcoming.csv")
    up.loc[0, ["odds_h", "odds_d", "odds_a"]] = np.nan
    up.loc[1, "book"] = np.nan
    up.to_csv(rep / "upcoming.csv", index=False)
    at = _run(tmp_path / "app.py")
    assert not at.exception


def test_app_counts_trades_logged_after_last_settle(tmp_path, full_reports):
    """paper_trades_settled.csv is a snapshot from the last `settle`; trades
    logged by `predict` afterwards must still be counted/shown (as open)."""
    rep = _app(tmp_path, full_reports)
    log = pd.read_csv(rep / "paper_trades.csv")
    extra = log.iloc[[0]].assign(date="2030-01-01", home="T01", away="T02")
    pd.concat([log, extra]).to_csv(rep / "paper_trades.csv", index=False)
    at = _run(tmp_path / "app.py")
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Bets logged"] == str(len(log) + 1)


def test_app_against_real_reports():
    if not (ROOT / "reports" / "model_comparison.csv").exists():
        pytest.skip("no real reports/ (run `python run.py evaluate`)")
    at = _run(ROOT / "app.py")
    assert not at.exception
