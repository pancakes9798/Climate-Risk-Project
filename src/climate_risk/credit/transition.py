"""Climate-policy shock on issuers' assets (Battiston & Monasterolo, 2020, Section 3.3).

For a transition from the baseline B to the policy scenario P of the same IAM, the output of
each climate-policy-relevant sector S changes by u_S(t) = Y_S(P, t)/Y_S(B, t) − 1. The SSP
vintage lets the policy pathways diverge from 2020, a divergence that has not materialised, so
the policy-to-baseline output ratio is harmonised to one at the jump-off date t₀ (ratio
harmonisation of Gidden et al., 2018, suited to positive quantities) and converges back to the
IAM ratio by the convergence year, with the same weight w(t) as every other scenario variable:

    1 + ũ_S(t) = (1 + u_S(t)) · [1 + ((1 + u_S(t₀))⁻¹ − 1) · w(t)],   ξ(t) = χ · Σ_S w_S ũ_S(t).

The pathway starts from today's state and catches up with the IAM's divergence by the
convergence year, as the carbon price does in the rates stage. With ``t_conv=None`` there is no
catch-up: the IAM's divergence after t₀ is applied to today's state for ever.

An issuer with business-line weights w_S (pre-shock, i.e. measured in B) receives the asset
shock ξ; χ is the elasticity of asset value to output (BM: χ = 1 in their application;
empirical market-share/profitability elasticities are 0.2–0.6). The shock relevant for a bond
maturing at t is the one realised by t.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from climate_risk.data import scenarios

FloatArray = NDArray[np.float64]

OIL_GAS = "Oil & gas"
UTILITIES = "Utilities"
CONTROL = "Low-carbon control"
SECTORS = (OIL_GAS, UTILITIES, CONTROL)


def power_mix_weights(frame: pd.DataFrame, baseline: str, t0: float) -> dict[str, float]:
    """Fossil vs low-carbon share of electricity generation in the baseline at ``t0``.

    A representative utility is assumed to own the generation mix of the market it serves.
    """
    outputs = scenarios.sector_outputs(frame, baseline, [t0]).iloc[0]
    total = outputs["ElFos"] + outputs["ElLowC"]
    return {"ElFos": float(outputs["ElFos"] / total), "ElLowC": float(outputs["ElLowC"] / total)}


def sector_exposures(frame: pd.DataFrame, baseline: str, t0: float) -> dict[str, dict[str, float]]:
    """Business-line weights of the three credit sectors analysed."""
    return {
        OIL_GAS: {"PrOilGas": 1.0},
        UTILITIES: power_mix_weights(frame, baseline, t0),
        CONTROL: {},
    }


def harmonised_shocks(
    frame: pd.DataFrame,
    baseline: str,
    policy: str,
    t: ArrayLike,
    t0: float,
    t_conv: float | None,
) -> pd.DataFrame:
    """ũ_S(t): output shock harmonised to zero at ``t0``, back to the IAM path from ``t_conv``."""
    tt = np.asarray(t, dtype=float)
    now = scenarios.output_shocks(frame, baseline, policy, [t0]).iloc[0].to_numpy()
    later = scenarios.output_shocks(frame, baseline, policy, tt)
    weight = np.ones_like(tt) if t_conv is None else scenarios.convergence_weight(tt, t0, t_conv)
    factor = 1.0 + np.outer(weight, 1.0 / (1.0 + now) - 1.0)
    return (1.0 + later) * factor - 1.0


def asset_shock(
    frame: pd.DataFrame,
    baseline: str,
    policy: str,
    exposure: Mapping[str, float],
    t: ArrayLike,
    chi: float,
    t0: float,
    t_conv: float | None,
) -> FloatArray:
    """ξ(t) for an issuer with the given business-line ``exposure`` (floored at −1)."""
    tt = np.asarray(t, dtype=float)
    if not exposure:
        return np.zeros_like(tt)
    shocks = harmonised_shocks(frame, baseline, policy, tt, t0, t_conv)
    weighted = sum(weight * shocks[line].to_numpy() for line, weight in exposure.items())
    return np.maximum(chi * np.asarray(weighted, dtype=float), -1.0)
