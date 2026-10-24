"""Shifting endpoint of the yield curve and its scenario-conditional path.

Historically the endpoint is the nominal natural rate (Bauer & Rudebusch, 2020)

    i*ₜ = r*ₜ + π*ₜ,

with r* the one-sided Holston–Laubach–Williams estimate and π* the Cleveland Fed 10-year
expected inflation. In the projections the endpoint moves with the SSP scenario through two
transparent channels:

1. *Growth (Ramsey/HLW) channel* – r*(s, t) = r*_T0 + θ · (g̃(s, t) − g_ref(t)), where g̃ is the
   scenario's world GDP-per-capita growth harmonised to the reference growth at the jump-off,
   g_ref the growth of the reference scenario and θ the HLW elasticity of r* to trend growth.
   In HLW r* is a random walk, so absent scenario information the best forecast of r* is its
   current value: the reference scenario keeps r*_T0, and the IAMs are used only for what they
   are designed for, the *relative* comparison of pathways. (Using absolute world growth would
   import the end of emerging-market convergence into US r*.)
2. *Carbon-price ("greenflation") channel* – the carbon bill as a share of GDP, c̃(s, t), ratio-
   harmonised from today's observed level to the pathway's level, raises the price level one for
   one (pass-through ψ); while it rises, inflation is higher by π_c = ψ · dc̃/dt and the short
   rate responds by κ · π_c (Taylor principle), i.e. the slope factor shifts by κ · π_c.
   Long-run inflation expectations stay anchored at π*_T0.

The curvature factor has no scenario channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from climate_risk.config import EndpointConfig
from climate_risk.data import scenarios

FloatArray = NDArray[np.float64]


def decimal_year(months: pd.PeriodIndex) -> FloatArray:
    """Mid-month decimal year, e.g. January 2026 → 2026 + 0.5/12."""
    return months.year.to_numpy(dtype=float) + (months.month.to_numpy(dtype=float) - 0.5) / 12.0


@dataclass(frozen=True)
class JumpOff:
    """Observed state of the endpoint drivers in the last month of data."""

    month: pd.Period
    rstar: float
    inflation: float

    @property
    def istar(self) -> float:
        return self.rstar + self.inflation

    @property
    def t0(self) -> float:
        return float(decimal_year(pd.PeriodIndex([self.month]))[0])


@dataclass(frozen=True)
class ScenarioPath:
    """Deterministic, scenario-specific component of the factors on a monthly grid."""

    scenario: str
    months: pd.PeriodIndex
    growth: FloatArray
    reference_growth: FloatArray
    rstar: FloatArray
    istar: FloatArray
    carbon_share: FloatArray
    carbon_inflation: FloatArray
    slope_shift: FloatArray

    @property
    def shift(self) -> FloatArray:
        """d(s, t) added to the stationary cycle: [i*, κ·π_c, 0]."""
        return np.column_stack([self.istar, self.slope_shift, np.zeros_like(self.istar)])

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "growth": self.growth,
                "reference_growth": self.reference_growth,
                "rstar": self.rstar,
                "istar": self.istar,
                "carbon_share": self.carbon_share,
                "carbon_inflation": self.carbon_inflation,
                "slope_shift": self.slope_shift,
            },
            index=self.months,
        )


def scenario_path(
    frame: pd.DataFrame,
    scenario: str,
    months: pd.PeriodIndex,
    jump_off: JumpOff,
    cfg: EndpointConfig,
    reference: str,
    convergence_year: float,
) -> ScenarioPath:
    """Endpoint drivers of ``scenario`` for every month in ``months`` (after the jump-off)."""
    t = decimal_year(months)
    t0 = jump_off.t0

    g_ref = scenarios.gdp_per_capita_growth(frame, reference, t)
    g_ref_t0 = float(scenarios.gdp_per_capita_growth(frame, reference, [t0])[0])
    g_native = scenarios.gdp_per_capita_growth(frame, scenario, t)
    g_native_t0 = float(scenarios.gdp_per_capita_growth(frame, scenario, [t0])[0])
    growth = scenarios.harmonise(g_native, t, g_native_t0, g_ref_t0, t0, convergence_year)
    rstar = jump_off.rstar + cfg.theta * (growth - g_ref)
    istar = rstar + jump_off.inflation

    c_native = scenarios.carbon_cost_share(frame, scenario, t)
    c_native_t0 = float(scenarios.carbon_cost_share(frame, scenario, [t0])[0])
    if np.allclose(c_native, 0.0):
        # No carbon price in the scenario: today's (small) carbon pricing simply persists.
        share = np.full_like(t, cfg.carbon_cost_share_t0)
    elif c_native_t0 > 0.0:
        # The carbon bill is a positive quantity: ratio harmonisation keeps it positive and
        # does not carry the IAM's 2020s ramp over to today's (much lower) level.
        share = scenarios.harmonise_ratio(
            c_native, t, c_native_t0, cfg.carbon_cost_share_t0, t0, convergence_year
        )
    else:
        share = scenarios.harmonise(
            c_native, t, c_native_t0, cfg.carbon_cost_share_t0, t0, convergence_year
        )
    carbon_inflation = cfg.psi * 100.0 * np.gradient(share, t) if t.size > 1 else np.zeros(1)
    return ScenarioPath(
        scenario=scenario,
        months=months,
        growth=growth,
        reference_growth=g_ref,
        rstar=rstar,
        istar=istar,
        carbon_share=share,
        carbon_inflation=carbon_inflation,
        slope_shift=cfg.kappa * carbon_inflation,
    )
