from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_risk.config import EndpointConfig
from climate_risk.data import scenarios
from climate_risk.term_structure import endpoint

YEARS = [2005, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]


@pytest.fixture
def frame(tmp_path: Path) -> pd.DataFrame:
    """Reference grows at 2%/yr, 'fast' at 3%/yr; the policies price carbon from 2020 or 2030."""
    rows = []
    for scenario, growth, price, start in [
        ("REF - Baseline", 0.02, 0.0, 2020),
        ("FAST - Baseline", 0.03, 0.0, 2020),
        ("REF - 1.9", 0.02, 100.0, 2020),
        ("LATE - 1.9", 0.02, 100.0, 2030),
    ]:
        for year in YEARS:
            rows.append(
                {
                    "Scenario": scenario,
                    "Year": year,
                    "GDP per capita": 1e4 * np.exp(growth * (year - 2005)),
                    "Population": 1e9,
                    "Carbon price": price * max(year - start, 0) / 10,
                    "Fossil CO2 emissions": 1e10,
                    "CO2 emissions (BECCS)": 0.0,
                    "Primary energy (oil)": 1.0,
                    "Primary energy (gas)": 1.0,
                    "Secondary electricity": 2.0,
                    "Secondary electricity (coal)": 0.5,
                    "Secondary electricity (gas)": 0.5,
                    "Secondary electricity (oil)": 0.0,
                }
            )
    path = tmp_path / "ssp.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return scenarios.load_scenarios(path)


@pytest.fixture
def jump_off() -> endpoint.JumpOff:
    return endpoint.JumpOff(month=pd.Period("2026-09", "M"), rstar=1.0, inflation=2.5)


def _months() -> pd.PeriodIndex:
    return pd.period_range("2026-10", "2100-12", freq="M", name="month")


def test_decimal_year_is_mid_month() -> None:
    t = endpoint.decimal_year(pd.PeriodIndex(["2026-01", "2026-12"], freq="M"))
    np.testing.assert_allclose(t, [2026 + 0.5 / 12, 2026 + 11.5 / 12])


def test_jump_off_istar(jump_off: endpoint.JumpOff) -> None:
    assert jump_off.istar == pytest.approx(3.5)
    assert jump_off.t0 == pytest.approx(2026 + 8.5 / 12)


def test_reference_path_starts_at_jump_off(frame: pd.DataFrame, jump_off: endpoint.JumpOff) -> None:
    path = endpoint.scenario_path(
        frame, "REF - Baseline", _months(), jump_off, EndpointConfig(), "REF - Baseline", 2050
    )
    np.testing.assert_allclose(path.growth, 2.0, atol=1e-6)
    np.testing.assert_allclose(path.istar, jump_off.istar, atol=1e-6)
    np.testing.assert_allclose(path.shift[:, 1:], 0.0, atol=1e-9)


def test_faster_growth_raises_endpoint_after_convergence(
    frame: pd.DataFrame, jump_off: endpoint.JumpOff
) -> None:
    cfg = EndpointConfig(theta=1.0)
    path = endpoint.scenario_path(
        frame, "FAST - Baseline", _months(), jump_off, cfg, "REF - Baseline", 2050
    )
    # harmonised to the reference growth at the jump-off; the grid starts one month later
    weight = scenarios.convergence_weight(jump_off.t0 + 1 / 12, jump_off.t0, 2050)
    assert path.growth[0] == pytest.approx(3.0 - 1.0 * weight, abs=1e-6)
    late = path.frame().loc[pd.Period("2060-01", "M")]
    assert late["growth"] == pytest.approx(3.0, abs=1e-6)
    assert late["istar"] == pytest.approx(jump_off.istar + 1.0, abs=1e-6)
    assert np.all(np.diff(path.istar) >= -1e-12)  # monotone convergence


def test_reference_keeps_current_rstar_whatever_its_growth(
    frame: pd.DataFrame, jump_off: endpoint.JumpOff
) -> None:
    """r* is a martingale in HLW: only growth *differences* vs the reference move it."""
    cfg = EndpointConfig(theta=1.0)
    ref = endpoint.scenario_path(
        frame, "FAST - Baseline", _months(), jump_off, cfg, "FAST - Baseline", 2050
    )
    np.testing.assert_allclose(ref.rstar, jump_off.rstar, atol=1e-9)
    slower = endpoint.scenario_path(
        frame, "REF - Baseline", _months(), jump_off, cfg, "FAST - Baseline", 2050
    )
    late = slower.frame().loc[pd.Period("2070-06", "M")]
    assert late["rstar"] == pytest.approx(jump_off.rstar - 1.0, abs=1e-6)
    assert late["reference_growth"] == pytest.approx(3.0, abs=1e-6)


def test_carbon_pricing_raises_short_rates_while_ramping(
    frame: pd.DataFrame, jump_off: endpoint.JumpOff
) -> None:
    cfg = EndpointConfig(theta=0.0, kappa=1.5, psi=1.0, carbon_cost_share_t0=0.0)
    path = endpoint.scenario_path(
        frame, "REF - 1.9", _months(), jump_off, cfg, "REF - Baseline", 2050
    )
    frame_ = path.frame()
    ramp = frame_.loc[pd.Period("2035-01", "M"), "carbon_inflation"]
    assert ramp > 0
    assert frame_.loc[pd.Period("2035-01", "M"), "slope_shift"] == pytest.approx(1.5 * ramp)
    np.testing.assert_allclose(path.shift[:, 0], jump_off.istar, atol=1e-9)  # theta = 0
    no_policy = replace(cfg, kappa=0.0)
    flat = endpoint.scenario_path(
        frame, "REF - 1.9", _months(), jump_off, no_policy, "REF - Baseline", 2050
    )
    np.testing.assert_allclose(flat.shift[:, 1], 0.0)


def test_carbon_price_starting_after_the_jump_off_is_offset_harmonised(
    frame: pd.DataFrame, jump_off: endpoint.JumpOff
) -> None:
    cfg = EndpointConfig(theta=0.0)
    months = _months()
    path = endpoint.scenario_path(
        frame, "LATE - 1.9", months, jump_off, cfg, "REF - Baseline", 2050
    )
    native = scenarios.carbon_cost_share(frame, "LATE - 1.9", endpoint.decimal_year(months))
    assert path.carbon_share[0] == pytest.approx(cfg.carbon_cost_share_t0, rel=1e-3)
    after = np.asarray(months >= pd.Period("2050-01", "M"))
    np.testing.assert_allclose(path.carbon_share[after], native[after])
    assert (path.carbon_share > 0).all()
