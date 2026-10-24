"""Shifting-endpoint dynamic Nelson–Siegel model.

    βₜ = xₜ + dₜ,     dₜ = [i*ₜ, 0, 0]',     xₜ = c + Φ xₜ₋₁ + ηₜ

The level factor is decomposed into the slow-moving nominal natural rate i* and a stationary
cycle; slope and curvature are stationary around constant means. Removing the trend from the
level before estimating the VAR avoids the near-unit-root persistence (and the downward-biased
long-horizon forecasts) of constant-mean models (Bauer & Rudebusch, 2020).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from climate_risk.term_structure.nelson_siegel import FACTORS
from climate_risk.term_structure.var import VAR1, fit_var1


@dataclass(frozen=True)
class ShiftingEndpointDNS:
    lam: float
    var: VAR1


def require_contiguous(index: pd.Index, what: str) -> None:
    """Fail unless ``index`` is a gap-free monthly PeriodIndex (lags are positional)."""
    if not isinstance(index, pd.PeriodIndex) or index.freqstr != "M":
        raise TypeError(f"{what} must be indexed by monthly periods")
    if len(index) and not index.equals(pd.period_range(index[0], index[-1], freq="M")):
        raise ValueError(f"{what} has missing months between {index[0]} and {index[-1]}")


def endpoint_cycle(betas: pd.DataFrame, istar: pd.Series) -> pd.DataFrame:
    """Cycle xₜ = βₜ − [i*ₜ, 0, 0] on the months where i* is observed."""
    aligned = istar.reindex(betas.index)
    cycle = betas[list(FACTORS)].copy()
    cycle["level"] = cycle["level"] - aligned
    return cycle.dropna()


def fit_model(betas: pd.DataFrame, istar: pd.Series, lam: float) -> ShiftingEndpointDNS:
    """Estimate the cycle VAR; a non-stationary cycle cannot be projected to 2100."""
    cycle = endpoint_cycle(betas, istar)
    require_contiguous(cycle.index, "the endpoint cycle")
    var = fit_var1(cycle)
    if not var.is_stationary:
        modulus = float(np.max(np.abs(var.eigenvalues)))
        raise ValueError(f"the cycle VAR is not stationary (largest |eigenvalue| = {modulus:.4f})")
    return ShiftingEndpointDNS(lam=lam, var=var)
