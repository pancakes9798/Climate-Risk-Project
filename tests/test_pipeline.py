"""End-to-end runs of the pipeline on the synthetic data root (see conftest.py)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_risk import report
from climate_risk.figures import render_all
from climate_risk.pipeline import Results, run_all, write_tables
from tests.conftest import small_config

pytestmark = pytest.mark.usefixtures("no_cds_override")


@pytest.fixture(scope="module")
def cds_results(data_root: Path) -> Results:
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CLIMATE_RISK_CDS_PATH", raising=False)
        return run_all(small_config(data_root))


def test_cds_mode_produces_every_table(cds_results: Results) -> None:
    expected = {
        "rates_summary",
        "ns_rmse_by_maturity",
        "ns_lambda_profile",
        "cycle_var",
        "cycle_var_modes",
        "endpoint_paths",
        "yield_projection_10y",
        "yield_curve_snapshots",
        "rates_sensitivity",
        "backtest_rmse_bp",
        "backtest_relative_rmse",
        "backtest_bias_bp",
        "backtest_diebold_mariano",
        "credit_calibration",
        "climate_spreads",
        "delayed_revelation",
        "bond_value_impact",
        "portfolio_correlation_blocks",
        "portfolio_sector_pds",
        "portfolio_risk",
        "portfolio_scenario_mixture",
    }
    assert expected <= set(cds_results.tables)
    assert cds_results.credit.mode == "cds"
    assert cds_results.portfolio.nu is not None


def test_recovers_synthetic_decay(cds_results: Results) -> None:
    assert cds_results.rates.fit.lam == pytest.approx(0.47, abs=0.03)
    assert cds_results.rates.model.var.is_stationary


def test_climate_spread_signs(cds_results: Results) -> None:
    spreads = cds_results.tables["climate_spreads"]
    central = spreads[np.isclose(spreads["chi"], 0.5) & np.isclose(spreads["tenor"], 10.0)]
    control = central[central["sector"] == "Low-carbon control"]
    # no transition shock for the control group: only the risk-free drift effect remains
    np.testing.assert_allclose(control["climate_spread_bp"], control["rate_effect_bp"], atol=1e-9)
    assert (np.sign(control["rate_effect_bp"]) == -np.sign(control["riskfree_shift_bp"])).all()
    oil = central[(central["sector"] == "Oil & gas") & (central["policy"] == "SSP2 - 1.9")]
    assert float(oil["climate_spread_bp"].iloc[0]) > 0


def test_priced_today_is_the_first_delayed_revelation(cds_results: Results) -> None:
    spreads = cds_results.tables["climate_spreads"]
    delayed = cds_results.tables["delayed_revelation"]
    first = delayed[delayed["revelation_month"] == delayed["revelation_month"].min()]
    merged = first.merge(
        spreads[np.isclose(spreads["chi"], 0.5)],
        left_on=["policy", "sector", "maturity"],
        right_on=["policy", "sector", "tenor"],
        suffixes=("_delayed", "_today"),
    )
    np.testing.assert_allclose(
        merged["climate_spread_bp_delayed"], merged["climate_spread_bp_today"], atol=1e-9
    )


def test_sensitivity_tables_cover_the_key_assumptions(cds_results: Results) -> None:
    credit = cds_results.tables["credit_sensitivity"]
    assert {"central", "kappa = 0 (look-through)", "convergence 2040"} <= set(credit["variant"])
    rates = cds_results.tables["rates_sensitivity"]
    central = rates[(rates["variant"] == "central") & (rates["scenario"] == "SSP2 - 1.9")]
    headline = cds_results.tables["bond_value_impact"]
    headline = headline[(headline["policy"] == "SSP2 - 1.9") & (headline["maturity"] == 10.0)]
    assert float(central["repricing_10y_bp"].iloc[0]) == pytest.approx(
        float(headline["riskfree_shift_bp"].iloc[0])
    )


def test_no_rate_drift_variant_keeps_the_treasury_repricing(cds_results: Results) -> None:
    sens = cds_results.tables["credit_sensitivity"].set_index(["variant", "sector"])
    central = sens.xs("central", level="variant")
    no_drift = sens.xs("no rate drift in PDs", level="variant")
    np.testing.assert_allclose(no_drift["riskfree_shift_bp"], central["riskfree_shift_bp"])


def test_bond_value_decomposition_adds_up(cds_results: Results) -> None:
    impact = cds_results.tables["bond_value_impact"]
    np.testing.assert_allclose(
        impact["rate"] + impact["credit"] + impact["interaction"], impact["total"], atol=1e-12
    )


def test_portfolio_risk_is_coherent(cds_results: Results) -> None:
    risk = cds_results.tables["portfolio_risk"]
    assert (risk["es"] >= risk["var"] - 1e-12).all()
    by_measure = risk.groupby("measure")["expected_loss"].mean()
    assert by_measure["Physical"] < by_measure["Risk-neutral"]


def test_tables_and_figures_are_written(cds_results: Results, tmp_path: Path) -> None:
    written = write_tables(cds_results.tables, tmp_path / "tables")
    assert len(written) == len(cds_results.tables)
    figures = render_all(cds_results, tmp_path / "figures")
    assert len(figures) == 24
    assert all(path.stat().st_size > 10_000 for path in figures)


def test_report_quotes_the_tables(cds_results: Results, tmp_path: Path) -> None:
    path = report.write(cds_results, tmp_path / "README.md")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Results")
    assert "Climate spreads if the transition is priced today" in text
    lam = cds_results.tables["rates_summary"].loc["lambda_per_year", "value"]
    assert f"{lam:.3f}" in text
    assert text.count("|---") >= 5


def test_markdown_table_formats_multiindex() -> None:
    frame = pd.DataFrame({"x": [1.234]}, index=pd.MultiIndex.from_tuples([("a", "b")]))
    assert "| a · b | 1.23 |" in report.markdown_table(frame)


def test_demo_mode_without_cds(data_root: Path, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    shutil.copytree(data_root, root)
    (root / "data" / "private" / "cds.csv").unlink()
    results = run_all(small_config(root), with_backtest=False)
    assert results.credit.mode == "demo"
    assert results.portfolio.nu is None
    calibration = results.tables["credit_calibration"]
    assert calibration["cds_PX5_bp"].nunique() == 1  # one public spread for every sector
