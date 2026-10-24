"""Boundary checks of the pipeline: data problems fail with a message, not a KeyError."""

from __future__ import annotations

import logging
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from climate_risk.config import Config
from climate_risk.credit.cds import PX
from climate_risk.data import scenarios
from climate_risk.pipeline import Results, credit, portfolio, rates, run_all
from tests.conftest import small_config

pytestmark = pytest.mark.usefixtures("no_cds_override")


@pytest.fixture
def root(data_root: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "repo"
    shutil.copytree(data_root, copy)
    return copy


@pytest.fixture(scope="module")
def results(data_root: Path) -> Results:
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CLIMATE_RISK_CDS_PATH", raising=False)
        return run_all(small_config(data_root), with_backtest=False)


def _credit_config(cfg: Config, **changes: object) -> Config:
    return replace(cfg, credit=replace(cfg.credit, **changes))


def test_stale_issuers_are_dropped(root: Path, caplog: pytest.LogCaptureFixture) -> None:
    panel_path = root / "data" / "private" / "cds.csv"
    panel = pd.read_csv(panel_path)
    stale = panel["Ticker"] == "UTI1"
    panel.loc[stale, list(PX)] = panel.loc[stale, list(PX)].iloc[0].to_numpy()
    panel.to_csv(panel_path, index=False)
    with caplog.at_level(logging.WARNING):
        inputs = credit._market_inputs(small_config(root))
    assert "UTI1" not in inputs.issuer_quotes.index
    assert "UTI1" not in inputs.weekly_changes.columns
    assert "stale quotes: UTI1" in caplog.text


def test_issuer_with_arbitrageable_curve_is_left_out_and_named(
    results: Results, caplog: pytest.LogCaptureFixture
) -> None:
    quotes = results.credit.issuer_quotes.copy()
    quotes.loc["OIL2"] = [3000.0, *[100.0] * 9]  # steeply inverted: negative forward hazard
    inverted = replace(results.credit, issuer_quotes=quotes)
    with caplog.at_level(logging.WARNING):
        pds, _ = portfolio._issuer_pds(results.config, inverted)
    assert "OIL2" not in pds.index
    assert len(pds) == len(quotes) - 1
    assert "issuer OIL2 left out" in caplog.text


def test_missing_cds_override_is_an_error(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLIMATE_RISK_CDS_PATH", str(tmp_path / "nowhere.csv"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        credit._market_inputs(small_config(root))


def test_calibration_window_without_quotes_is_rejected(root: Path) -> None:
    cfg = _credit_config(small_config(root), window=("2021-01", "2021-12"))
    with pytest.raises(ValueError, match="no issuer is quoted"):
        credit._market_inputs(cfg)


def test_sector_without_issuers_is_rejected(root: Path) -> None:
    sectors = pd.read_csv(root / "data" / "sectors.csv")
    sectors.loc[sectors["sector"] == "Utilities", "sector"] = "Other"
    sectors.to_csv(root / "data" / "sectors.csv", index=False)
    with pytest.raises(ValueError, match="no eligible CDS issuers"):
        credit._market_inputs(small_config(root))


def test_demo_mode_needs_baa_data_in_the_window(root: Path) -> None:
    (root / "data" / "private" / "cds.csv").unlink()
    cfg = _credit_config(small_config(root), window=("2030-01", "2030-12"))
    with pytest.raises(ValueError, match="no Moody's Baa observations"):
        credit._market_inputs(cfg)


def test_jump_off_beyond_the_data_is_rejected(data_root: Path) -> None:
    cfg = small_config(data_root)
    late = replace(cfg, term_structure=replace(cfg.term_structure, as_of="2026-12"))
    frame = scenarios.load_scenarios(cfg.paths.scenarios_csv)
    with pytest.raises(ValueError, match="jump-off month 2026-12 is not covered"):
        rates.run(late, frame, with_backtest=False)


def test_credit_results_can_be_pickled(results: Results) -> None:
    restored = pickle.loads(pickle.dumps(results.credit))
    assert restored.discount(1.0) == pytest.approx(results.credit.discount(1.0))
