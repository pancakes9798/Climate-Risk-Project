from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from climate_risk.config import (
    CDS_PATH_ENV,
    BacktestConfig,
    Config,
    CreditConfig,
    Paths,
    PortfolioConfig,
    ScenarioConfig,
    SimulationConfig,
    TermStructureConfig,
)


def test_defaults_are_consistent() -> None:
    cfg = Config()
    assert cfg.scenarios.headline in cfg.scenarios.transitions
    assert cfg.credit.chi in cfg.credit.chi_sensitivity


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ScenarioConfig(headline=("SSP2 - Baseline", "SSP2 - 3.4")), "headline"),
        (lambda: ScenarioConfig(reference="SSP2 - 1.9"), "reference pathway"),
        (lambda: ScenarioConfig(headline=("SSP5 - Baseline", "SSP5 - 2.6")), "policy_family"),
        (lambda: ScenarioConfig(convergence_year=2120), "convergence_year"),
        (lambda: TermStructureConfig(output_maturities=(1, 5, 30)), "10-year"),
        (lambda: PortfolioConfig(risk_premium_ratio=0.8), "risk_premium_ratio"),
        (lambda: CreditConfig(chi=0.3), "chi_sensitivity"),
        (lambda: CreditConfig(recovery=1.0), "recovery"),
        (lambda: CreditConfig(window=("2019-12", "2015-01")), "window"),
        (lambda: SimulationConfig(quantiles=(0.1, 0.9)), "median"),
        (lambda: SimulationConfig(quantiles=(0.5, 0.25)), "increasing"),
        (lambda: PortfolioConfig(alpha=0.3), "alpha"),
        (lambda: TermStructureConfig(lambda_grid=(1.0, 0.5, 10)), "lambda_grid"),
    ],
)
def test_invalid_settings_fail_fast(factory: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_sections_are_checked_against_each_other() -> None:
    with pytest.raises(ValueError, match="convergence_year 2020"):
        Config(scenarios=ScenarioConfig(convergence_year=2020))
    with pytest.raises(ValueError, match="must precede 2100"):
        Config(term_structure=TermStructureConfig(as_of="2100-06"))
    with pytest.raises(ValueError, match="months of data before"):
        Config(backtest=BacktestConfig(first_origin="1988-01"))
    with pytest.raises(ValueError, match="estimation maturities"):
        Config(backtest=BacktestConfig(maturities=(2, 11)))


def test_replace_validates_again() -> None:
    with pytest.raises(ValueError, match="chi"):
        replace(CreditConfig(), chi=0.7)


def test_cds_location_is_read_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CDS_PATH_ENV, str(tmp_path / "panel.csv"))
    paths = Paths(root=tmp_path)
    monkeypatch.delenv(CDS_PATH_ENV)
    assert paths.cds_csv == tmp_path / "panel.csv"
    assert Paths(root=tmp_path).cds_csv == tmp_path / "data" / "private" / "cds.csv"
