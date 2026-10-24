from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_risk.data import scenarios

YEARS = [2005, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]


def _frame(**overrides: dict[str, list[float]]) -> pd.DataFrame:
    """Minimal OWID-like table with two scenarios; growth of GDP per capita is exactly 2%/yr."""
    rows = []
    for scenario, scale in [("SSP2 - Baseline", 1.0), ("SSP2 - 1.9", 0.5)]:
        for i, year in enumerate(YEARS):
            row = {
                "Scenario": scenario,
                "Year": year,
                "GDP per capita": 10_000 * np.exp(0.02 * (year - 2005)),
                "Population": 8e9,
                "Carbon price": np.nan if scenario.endswith("Baseline") else 10.0 * i,
                "Fossil CO2 emissions": 3e10 * scale - 1e10,
                "CO2 emissions (BECCS)": 1e10,
                "Primary energy (oil)": 100.0 * scale,
                "Primary energy (gas)": 50.0 * scale,
                "Secondary electricity": 40.0,
                "Secondary electricity (coal)": 10.0 * scale,
                "Secondary electricity (gas)": 8.0 * scale,
                "Secondary electricity (oil)": 2.0 * scale,
            }
            for column, values in overrides.get(scenario, {}).items():
                row[column] = values[i]
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def scen(tmp_path: Path) -> pd.DataFrame:
    path = tmp_path / "ssp.csv"
    _frame().to_csv(path, index=False)
    return scenarios.load_scenarios(path)


def test_load_sets_index_and_zero_baseline_carbon_price(scen: pd.DataFrame) -> None:
    assert scen.index.names == ["scenario", "year"]
    assert scenarios.available(scen) == ("SSP2 - 1.9", "SSP2 - Baseline")
    assert (scenarios.series(scen, "SSP2 - Baseline", "Carbon price") == 0.0).all()


def test_load_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "ssp.csv"
    _frame().drop(columns=["Population"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Population"):
        scenarios.load_scenarios(path)


def test_load_rejects_inconsistent_power_mix(tmp_path: Path) -> None:
    path = tmp_path / "ssp.csv"
    frame = _frame()
    frame["Secondary electricity (gas)"] = 1e3  # e.g. secondary *energy* picked by mistake
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="fossil electricity"):
        scenarios.load_scenarios(path)


def test_unknown_scenario_raises(scen: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        scenarios.series(scen, "SSP9 - 9.9", "Population")


def test_gdp_growth_recovers_constant_rate(scen: pd.DataFrame) -> None:
    t = np.array([2026.5, 2050.0, 2099.9])
    growth = scenarios.gdp_per_capita_growth(scen, "SSP2 - Baseline", t)
    np.testing.assert_allclose(growth, 2.0, atol=1e-6)


def test_interpolation_is_clamped_outside_nodes(scen: pd.DataFrame) -> None:
    pop = scenarios.series(scen, "SSP2 - Baseline", "Population")
    values = scenarios.pchip(pop, np.array([1990.0, 2101.0]))
    np.testing.assert_allclose(values, 8e9)


def test_carbon_cost_share(scen: pd.DataFrame) -> None:
    t = np.array([2030.0])
    share = scenarios.carbon_cost_share(scen, "SSP2 - 1.9", t)
    gdp = 10_000 * np.exp(0.02 * 25) * 8e9
    assert share[0] == pytest.approx(30.0 * 1.5e10 / gdp, rel=1e-9)
    assert scenarios.carbon_cost_share(scen, "SSP2 - Baseline", t)[0] == 0.0


def test_harmonise_starts_at_anchor_and_rejoins_native_path() -> None:
    t = np.array([2026.0, 2038.0, 2050.0, 2070.0])
    native = np.array([3.0, 2.5, 2.0, 1.5])
    out = scenarios.harmonise(native, t, native_at_t0=3.0, anchor=2.0, t0=2026.0, t_conv=2050.0)
    np.testing.assert_allclose(out, [2.0, 2.0, 2.0, 1.5])


def test_harmonise_validates_years() -> None:
    with pytest.raises(ValueError, match="t_conv"):
        scenarios.harmonise(np.ones(1), np.ones(1), 1.0, 1.0, t0=2050.0, t_conv=2030.0)


def test_convergence_weight_is_smooth_at_both_ends() -> None:
    t = np.array([2026.0, 2026.001, 2049.999, 2050.0, 2060.0])
    w = scenarios.convergence_weight(t, 2026.0, 2050.0)
    np.testing.assert_allclose(w[[0, 3, 4]], [1.0, 0.0, 0.0])
    # zero slope at both ends: the first-order change over 0.001y is negligible
    assert 1.0 - w[1] < 1e-8
    assert w[2] < 1e-8


def test_values_extend_one_year_past_the_last_node(scen: pd.DataFrame) -> None:
    growth = scenarios.gdp_per_capita_growth(scen, "SSP2 - Baseline", np.array([2100.9, 2105]))
    np.testing.assert_allclose(growth, 2.0, atol=1e-6)


def test_sector_outputs_split_electricity(scen: pd.DataFrame) -> None:
    out = scenarios.sector_outputs(scen, "SSP2 - 1.9", np.array([2030.0]))
    assert list(out.columns) == list(scenarios.SECTOR_OUTPUTS)
    assert out["PrOilGas"].iloc[0] == pytest.approx(75.0)
    assert out["ElFos"].iloc[0] == pytest.approx(10.0)
    assert out["ElLowC"].iloc[0] == pytest.approx(30.0)


def test_output_shocks_relative_to_baseline(scen: pd.DataFrame) -> None:
    shocks = scenarios.output_shocks(scen, "SSP2 - Baseline", "SSP2 - 1.9", np.array([2040.0]))
    assert shocks["PrOilGas"].iloc[0] == pytest.approx(-0.5)
    assert shocks["ElFos"].iloc[0] == pytest.approx(-0.5)
    assert shocks["ElLowC"].iloc[0] == pytest.approx(30.0 / 20.0 - 1.0)


def test_ratio_harmonisation_keeps_sign_and_converges() -> None:
    t = np.array([2026.0, 2030.0, 2038.0, 2050.0, 2060.0])
    native = np.array([2.0, 1.0, 0.5, 1.5, 3.0])
    out = scenarios.harmonise_ratio(
        native, t, native_at_t0=2.0, anchor=0.1, t0=2026.0, t_conv=2050.0
    )
    assert out[0] == pytest.approx(0.1)
    assert (out > 0).all()
    np.testing.assert_allclose(out[3:], native[3:])
    with pytest.raises(ValueError, match="positive"):
        scenarios.harmonise_ratio(native, t, 0.0, 0.1, 2026.0, 2050.0)
