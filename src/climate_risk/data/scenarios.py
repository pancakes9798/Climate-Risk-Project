"""Shared Socioeconomic Pathway (SSP) scenarios.

The source is the SSP Public Database (Riahi et al., 2017) as compiled by Our World in Data:
one row per (scenario, year) at the IAM reporting years 2005, 2010, 2020, …, 2100. Each SSP
narrative is produced by a single marker IAM, so a baseline and the policy variants of the same
SSP are internally consistent (Battiston & Monasterolo, 2020, compare B → P within a model).

Decadal levels are interpolated with a shape-preserving monotone cubic (PCHIP), which never
overshoots the data; the growth rate of GDP per capita is the derivative of a natural cubic
spline through ln GDP per capita, which is smooth across the decadal nodes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline, PchipInterpolator

REQUIRED_COLUMNS = (
    "Scenario",
    "Year",
    "GDP per capita",
    "Population",
    "Carbon price",
    "Fossil CO2 emissions",
    "CO2 emissions (BECCS)",
    "Primary energy (oil)",
    "Primary energy (gas)",
    "Secondary electricity",
    "Secondary electricity (coal)",
    "Secondary electricity (gas)",
    "Secondary electricity (oil)",
)

#: Sector outputs used for the climate-policy shock (Climate Policy Relevant Sectors):
#: primary oil & gas supply, fossil electricity (coal, gas, oil; CCS included), low-carbon
#: electricity (all other sources). OWID's "(coal)"/"(gas)" columns already include CCS units.
SECTOR_OUTPUTS = ("PrOilGas", "ElFos", "ElLowC")

FloatArray = NDArray[np.float64]

#: Paths may be evaluated up to one year past the last IAM node (projections end in December
#: 2100); within that year interpolants are continued along their last segment.
EXTRAPOLATION_YEARS = 1.0


def _clip(t: ArrayLike, years: FloatArray) -> FloatArray:
    return np.clip(np.asarray(t, dtype=float), years[0], years[-1] + EXTRAPOLATION_YEARS)


def load_scenarios(path: Path) -> pd.DataFrame:
    """Read the OWID SSP table, indexed by ``(scenario, year)``.

    Baseline scenarios carry no climate policy, so a missing carbon price is set to zero there.
    """
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    frame = raw.rename(columns={"Scenario": "scenario", "Year": "year"})
    frame["scenario"] = frame["scenario"].astype(str)
    frame["year"] = frame["year"].astype(int)
    numeric = [c for c in frame.columns if c not in ("scenario", "year")]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    is_baseline = frame["scenario"].str.endswith("Baseline")
    frame.loc[is_baseline, "Carbon price"] = frame.loc[is_baseline, "Carbon price"].fillna(0.0)
    _check_power_mix(frame, path)
    return frame.set_index(["scenario", "year"]).sort_index()


def _check_power_mix(frame: pd.DataFrame, path: Path) -> None:
    """Fossil generation cannot exceed total generation.

    The OWID file repeats the headers "Secondary electricity (gas)" and "(oil)" for secondary
    *energy* (gases, liquids); pandas keeps the first copy, the electricity one. This check fails
    loudly if a future file changes that order.
    """
    fossil = frame[
        [
            "Secondary electricity (coal)",
            "Secondary electricity (gas)",
            "Secondary electricity (oil)",
        ]
    ].sum(axis=1, min_count=1)
    excess = fossil - frame["Secondary electricity"]
    if (excess > 1e-6 * frame["Secondary electricity"].abs().clip(lower=1.0)).any():
        raise ValueError(f"{path.name}: fossil electricity exceeds total electricity")


def available(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(frame.index.get_level_values("scenario").unique()))


def series(frame: pd.DataFrame, scenario: str, variable: str) -> pd.Series:
    """One variable of one scenario, indexed by year, without missing years."""
    if scenario not in frame.index.get_level_values("scenario"):
        raise KeyError(f"unknown scenario {scenario!r}")
    return frame.loc[scenario, variable].dropna().astype(float)


def pchip(values: pd.Series, t: ArrayLike) -> FloatArray:
    """Interpolate a by-year series at decimal years ``t`` (flat before the first node)."""
    years = values.index.to_numpy(dtype=float)
    interpolator = PchipInterpolator(years, values.to_numpy(dtype=float), extrapolate=True)
    return np.asarray(interpolator(_clip(t, years)), dtype=float)


def gdp_per_capita_growth(frame: pd.DataFrame, scenario: str, t: ArrayLike) -> FloatArray:
    """Instantaneous growth of GDP per capita, % per year: 100·d ln(GDPpc)/dt."""
    gdp_pc = series(frame, scenario, "GDP per capita")
    years = gdp_pc.index.to_numpy(dtype=float)
    log_path = CubicSpline(years, np.log(gdp_pc.to_numpy()), bc_type="natural")
    return 100.0 * log_path.derivative()(_clip(t, years))


def carbon_cost_share(frame: pd.DataFrame, scenario: str, t: ArrayLike) -> FloatArray:
    """Carbon-tax bill on gross fossil CO2 as a share of GDP.

    OWID reports energy-and-industry CO2 *net* of BECCS removals, which turn negative late in
    the stringent pathways. The tax base relevant for consumer prices is gross emissions
    (net + BECCS removals, floored at zero); payments for removals are fiscal transfers, not
    a cost pushed onto the consumption basket.
    """
    nodes = frame.loc[scenario]
    gross = (nodes["Fossil CO2 emissions"] + nodes["CO2 emissions (BECCS)"].fillna(0.0)).clip(
        lower=0.0
    )
    gdp = nodes["GDP per capita"] * nodes["Population"]
    share = (nodes["Carbon price"] * gross / gdp).dropna()
    return pchip(share, t)


def convergence_weight(t: ArrayLike, t0: float, t_conv: float) -> FloatArray:
    """Weight of the harmonisation offset: 1 at ``t0``, 0 from ``t_conv``, cosine taper between.

    A smooth (C¹) version of the linear convergence of Gidden et al. (2018), so that derived
    rates of change (growth, inflation impulses) have no jump at the convergence year.
    """
    if t_conv <= t0:
        raise ValueError("t_conv must be later than t0")
    x = np.clip((np.asarray(t, dtype=float) - t0) / (t_conv - t0), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def harmonise(
    native: FloatArray,
    t: FloatArray,
    native_at_t0: float,
    anchor: float,
    t0: float,
    t_conv: float,
) -> FloatArray:
    """Offset harmonisation with convergence (Gidden et al., 2018).

    The path is shifted so that it equals ``anchor`` at ``t0``; the shift fades out and the
    native IAM path is recovered from ``t_conv`` onwards. This removes the jump between the
    observed jump-off point and the (older) scenario vintage without altering long-run dynamics.
    """
    weight = convergence_weight(t, t0, t_conv)
    return np.asarray(native, dtype=float) + (anchor - native_at_t0) * weight


def harmonise_ratio(
    native: FloatArray,
    t: FloatArray,
    native_at_t0: float,
    anchor: float,
    t0: float,
    t_conv: float,
) -> FloatArray:
    """Ratio harmonisation with convergence (Gidden et al., 2018), for positive quantities.

    ṽ(t) = v(t) · [1 + (anchor / v(t₀) − 1) · w(t)]: the path is rescaled to start at
    ``anchor`` and the scaling fades out by ``t_conv``. It preserves signs, and it scales the
    native rate of change at t₀ by anchor / v(t₀) instead of carrying it over unchanged.
    """
    if native_at_t0 <= 0:
        raise ValueError("ratio harmonisation needs a positive value at t0")
    weight = convergence_weight(t, t0, t_conv)
    return np.asarray(native, dtype=float) * (1.0 + (anchor / native_at_t0 - 1.0) * weight)


def sector_outputs(frame: pd.DataFrame, scenario: str, t: ArrayLike) -> pd.DataFrame:
    """Physical output of the climate-policy-relevant sectors at decimal years ``t``."""

    def path(variable: str) -> FloatArray:
        return pchip(series(frame, scenario, variable), t)

    fossil_power = (
        path("Secondary electricity (coal)")
        + path("Secondary electricity (gas)")
        + path("Secondary electricity (oil)")
    )
    out = pd.DataFrame(
        {
            "PrOilGas": path("Primary energy (oil)") + path("Primary energy (gas)"),
            "ElFos": fossil_power,
            "ElLowC": path("Secondary electricity") - fossil_power,
        },
        index=pd.Index(np.asarray(t, dtype=float), name="t"),
    )
    return out[list(SECTOR_OUTPUTS)]


def output_shocks(frame: pd.DataFrame, baseline: str, policy: str, t: ArrayLike) -> pd.DataFrame:
    """Battiston–Monasterolo sector shock u_S(P, t) = Y_S(P, t) / Y_S(B, t) − 1."""
    return sector_outputs(frame, policy, t) / sector_outputs(frame, baseline, t) - 1.0
