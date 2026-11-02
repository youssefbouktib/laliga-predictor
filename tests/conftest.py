from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = pathlib.Path(__file__).resolve().parent
for p in (str(ROOT), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import synth  # noqa: E402
from src import data, features  # noqa: E402


@pytest.fixture(scope="session")
def league_dir(tmp_path_factory) -> pathlib.Path:
    """Raw football-data CSVs for a small synthetic league (read-only use)."""
    raw = tmp_path_factory.mktemp("raw")
    synth.write_league(raw)
    return raw


@pytest.fixture(scope="session")
def history(league_dir):
    return data.load_history(league_dir)


@pytest.fixture(scope="session")
def featured(history):
    df, st = features.build_features(history)
    return df, st


@pytest.fixture
def raw_copy(tmp_path, league_dir) -> pathlib.Path:
    """A writable copy of the synthetic raw folder."""
    import shutil
    dst = tmp_path / "raw"
    shutil.copytree(league_dir, dst)
    return dst
