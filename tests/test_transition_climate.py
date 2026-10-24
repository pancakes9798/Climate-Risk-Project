from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_risk.credit import climate, hazard, merton, transition
from climate_risk.data import scenarios

YEARS = [2005, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
TENORS = np.arange(1.0, 11.0)


@pytest.fixture
def frame(tmp_path: Path) -> pd.DataFrame:
    """Policy halves fossil supply and fossil power, doubles low-carbon power, from 2030 on."""
    rows = []
    for scenario in ("X - Baseline", "X - 1.9"):
        for year in YEARS:
            cut = scenario.endswith("1.9") and year >= 2030
            rows.append(
                {
                    "Scenario": scenario,
                    "Year": year,
                    "GDP per capita": 1e4,
                    "Population": 1e9,
                    "Carbon price": 0.0,
                    "Fossil CO2 emissions": 1e10,
                    "CO2 emissions (BECCS)": 0.0,
                    "Primary energy (oil)": 60.0 * (0.5 if cut else 1.0),
                    "Primary energy (gas)": 40.0 * (0.5 if cut else 1.0),
                    "Secondary electricity": 30.0 + 10.0 if cut else 30.0,
                    "Secondary electricity (coal)": 10.0 * (0.5 if cut else 1.0),
                    "Secondary electricity (gas)": 10.0 * (0.5 if cut else 1.0),
                    "Secondary electricity (oil)": 0.0,
                }
            )
    path = tmp_path / "ssp.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return scenarios.load_scenarios(path)


def test_power_mix_uses_baseline_shares(frame: pd.DataFrame) -> None:
    weights = transition.power_mix_weights(frame, "X - Baseline", 2026.0)
    assert weights == pytest.approx({"ElFos": 20.0 / 30.0, "ElLowC": 10.0 / 30.0})


def test_sector_exposures(frame: pd.DataFrame) -> None:
    exposures = transition.sector_exposures(frame, "X - Baseline", 2026.0)
    assert exposures["Oil & gas"] == {"PrOilGas": 1.0}
    assert sum(exposures["Utilities"].values()) == pytest.approx(1.0)
    assert exposures["Low-carbon control"] == {}


def test_asset_shock_is_chi_weighted_output_shock(frame: pd.DataFrame) -> None:
    t = np.array([2035.0, 2045.0])
    args = (frame, "X - Baseline", "X - 1.9")
    # jump-off before any divergence: harmonisation is inactive
    kw = {"t0": 2015.0, "t_conv": 2020.0}
    oil = transition.asset_shock(*args, {"PrOilGas": 1.0}, t, chi=0.5, **kw)
    np.testing.assert_allclose(oil, -0.25)
    utility = transition.asset_shock(*args, {"ElFos": 2 / 3, "ElLowC": 1 / 3}, t, chi=1.0, **kw)
    # fossil power -50 %, low-carbon power +200 % (10 → 30)
    np.testing.assert_allclose(utility, 2 / 3 * -0.5 + 1 / 3 * 2.0)
    none = transition.asset_shock(*args, {}, t, chi=1.0, **kw)
    np.testing.assert_allclose(none, 0.0)


def test_shock_is_zero_at_jump_off_and_native_after_convergence(frame: pd.DataFrame) -> None:
    args = (frame, "X - Baseline", "X - 1.9")
    t = np.array([2027.0, 2040.0, 2060.0])
    shocks = transition.harmonised_shocks(*args, t, t0=2027.0, t_conv=2050.0)
    native = scenarios.output_shocks(frame, "X - Baseline", "X - 1.9", t)
    assert shocks["PrOilGas"].iloc[0] == pytest.approx(0.0)
    assert shocks["PrOilGas"].iloc[2] == pytest.approx(native["PrOilGas"].iloc[2])
    weight = scenarios.convergence_weight(2040.0, 2027.0, 2050.0)
    start = scenarios.output_shocks(frame, "X - Baseline", "X - 1.9", [2027.0])["PrOilGas"].iloc[0]
    factor = 1.0 + weight * (1.0 / (1.0 + start) - 1.0)
    expected = (1.0 + native["PrOilGas"].iloc[1]) * factor - 1.0
    assert shocks["PrOilGas"].iloc[1] == pytest.approx(expected)
    no_catch_up = transition.harmonised_shocks(*args, t, t0=2027.0, t_conv=None)
    assert no_catch_up["PrOilGas"].iloc[2] == pytest.approx(
        (1.0 + native["PrOilGas"].iloc[2]) / (1.0 + start) - 1.0
    )


def test_asset_shock_floored_at_total_loss(frame: pd.DataFrame) -> None:
    shock = transition.asset_shock(
        frame,
        "X - Baseline",
        "X - 1.9",
        {"PrOilGas": 1.0},
        np.array([2050.0]),
        chi=3.0,
        t0=2015.0,
        t_conv=2020.0,
    )
    assert shock[0] == -1.0


def _credit() -> climate.SectorCredit:
    quotes = np.array([20, 32, 45, 58, 70, 82, 92, 101, 108, 114], dtype=float)

    def discount(t: np.ndarray) -> np.ndarray:
        return np.exp(-0.02 * np.asarray(t))

    return climate.calibrate_sector(
        "Oil & gas", quotes, TENORS, discount, zero_rates=np.full(10, 0.02), recovery=0.4
    )


def test_calibrate_sector_names_the_failing_sector() -> None:
    def discount(t: np.ndarray) -> np.ndarray:
        return np.exp(-0.02 * np.asarray(t))

    with pytest.raises(ValueError, match="Utilities"):
        climate.calibrate_sector(
            "Utilities",
            np.full(10, np.nan),
            TENORS,
            discount,
            zero_rates=np.full(10, 0.02),
            recovery=0.4,
        )


def test_calibrate_sector_uses_bootstrapped_pds() -> None:
    credit = _credit()
    discount = lambda t: np.exp(-0.02 * np.asarray(t))  # noqa: E731
    curve = hazard.bootstrap(credit.quotes_bp, TENORS, discount, 0.4)
    np.testing.assert_allclose(credit.baseline_pd(TENORS), curve.default_probability(TENORS))
    assert 0.05 < credit.sigma < 1.0


def test_climate_spreads_sign_and_zero_shock() -> None:
    credit = _credit()
    flat = climate.climate_spreads(credit, np.zeros(10), lgd=0.6)
    np.testing.assert_allclose(flat["climate_spread_bp"], 0.0, atol=1e-10)
    hit = climate.climate_spreads(credit, np.full(10, -0.2), lgd=0.6)
    assert (hit["climate_spread_bp"] > 0).all()
    assert (hit["pd_policy"] > hit["pd_baseline"]).all()
    expected = merton.zero_coupon_spread(hit["pd_policy"], TENORS, 0.6) * 1e4
    np.testing.assert_allclose(hit["spread_policy_bp"], expected)


def test_climate_spreads_beyond_last_quote_use_flat_hazard() -> None:
    credit = _credit()
    long = climate.climate_spreads(credit, np.zeros(2), lgd=0.6, maturities=np.array([10.0, 20.0]))
    lam_last = credit.curve.hazards[-1]
    q20 = 1 - (1 - long["pd_baseline"].iloc[0]) * np.exp(-lam_last * 10)
    assert long["pd_baseline"].iloc[1] == pytest.approx(q20)


def test_rate_shift_enters_the_merton_threshold() -> None:
    credit = _credit()
    higher_rates = climate.climate_spreads(credit, np.zeros(10), lgd=0.6, rate_shift=0.001)
    assert (higher_rates["climate_spread_bp"] < 0).all()  # higher risk-free drift, lower Q-PD
    np.testing.assert_allclose(higher_rates["rate_effect_bp"], higher_rates["climate_spread_bp"])
    np.testing.assert_allclose(higher_rates["riskfree_shift_bp"], 10.0)
    expected = merton.merton_pd(
        credit.merton_fit.a, credit.sigma, TENORS, np.full(10, 0.02 + 0.001)
    )
    direct = merton.merton_pd(credit.merton_fit.a, credit.sigma, TENORS, np.full(10, 0.02))
    shifted = merton.shifted_pd(direct, TENORS, credit.sigma, np.zeros(10), rate_shift=0.001)
    np.testing.assert_allclose(shifted, expected, rtol=1e-10)


def test_bond_value_decomposition_is_exact() -> None:
    impact = climate.bond_value_impact(
        rf_baseline=0.03, rf_policy=0.028, spread_baseline=0.01, spread_policy=0.015, maturity=10.0
    )
    assert impact["rate"] == pytest.approx(np.exp(0.02) - 1)
    assert impact["credit"] == pytest.approx(np.exp(-0.05) - 1)
    assert impact["total"] == pytest.approx((1 + impact["rate"]) * (1 + impact["credit"]) - 1)
    assert impact["rate"] + impact["credit"] + impact["interaction"] == pytest.approx(
        impact["total"]
    )


def test_rate_drift_flag_changes_pds_but_not_the_reported_repricing() -> None:
    def discount(t: np.ndarray) -> np.ndarray:
        return np.exp(-0.02 * np.asarray(t))

    quotes = np.linspace(20.0, 110.0, 10)
    credit = climate.calibrate_sector(
        "Oil & gas", quotes, TENORS, discount, zero_rates=np.full(10, 0.02), recovery=0.4
    )
    xi, dy = np.full(10, -0.1), np.full(10, 0.001)
    with_drift = climate.climate_spreads(credit, xi, 0.6, TENORS, dy)
    without = climate.climate_spreads(credit, xi, 0.6, TENORS, dy, rate_drift=False)
    np.testing.assert_allclose(without["riskfree_shift_bp"], 10.0)
    np.testing.assert_allclose(with_drift["riskfree_shift_bp"], without["riskfree_shift_bp"])
    np.testing.assert_allclose(without["rate_effect_bp"], 0.0, atol=1e-12)
    assert (with_drift["rate_effect_bp"] < 0).all()  # higher drift, lower risk-neutral PDs
