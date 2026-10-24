"""Nelson–Siegel cross-section with a single, time-invariant decay parameter.

Following Diebold & Li (2006), the decay λ is held fixed over time, which makes the factor
loadings constant: the betas are then obtained month by month by ordinary least squares and form
a well-defined time series (level, slope, curvature) that can be modelled dynamically. Here λ is
not imposed but estimated once, by profiling the pooled sum of squared fitting errors over all
months and maturities.

    y_t(τ) = β₁ₜ + β₂ₜ · (1 − e^{−λτ})/(λτ) + β₃ₜ · [(1 − e^{−λτ})/(λτ) − e^{−λτ}] + εₜ(τ)

Maturities τ are in years, λ in 1/years, yields in percent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

FACTORS = ("level", "slope", "curvature")
# The curvature loading (1 − e^{−x})/x − e^{−x} peaks at x = λτ ≈ 1.79328.
CURVATURE_ARGMAX = 1.7932821329007602

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


def loadings(tau: ArrayLike, lam: float) -> FloatArray:
    """Factor loadings, one row per maturity: [1, slope loading, curvature loading]."""
    x = lam * np.asarray(tau, dtype=float)
    safe = np.where(x < 1e-8, 1.0, x)
    slope = np.where(x < 1e-8, 1.0 - x / 2.0, -np.expm1(-safe) / safe)
    curvature = slope - np.exp(-x)
    return np.column_stack([np.ones_like(x), slope, curvature])


def _maturities(yields: pd.DataFrame) -> FloatArray:
    return np.asarray(yields.columns, dtype=float)


def fit_betas(yields: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Cross-sectional OLS betas for every date (rows) given the decay ``lam``."""
    if yields.isna().to_numpy().any():
        raise ValueError("yield panel contains missing values; drop or fill them first")
    design = loadings(_maturities(yields), lam)
    coef, *_ = np.linalg.lstsq(design, yields.to_numpy(dtype=float).T, rcond=None)
    return pd.DataFrame(coef.T, index=yields.index, columns=list(FACTORS))


def zero_rates(betas: pd.Series | ArrayLike, maturities: ArrayLike, lam: float) -> FloatArray:
    """Zero-coupon yields (percent) implied by one vector of betas."""
    beta = betas.reindex(list(FACTORS)) if isinstance(betas, pd.Series) else betas
    return loadings(maturities, lam) @ np.asarray(beta, dtype=float)


def fitted_yields(betas: pd.DataFrame, maturities: ArrayLike, lam: float) -> pd.DataFrame:
    """Model yields for every date in ``betas`` at the requested maturities."""
    taus = np.asarray(maturities, dtype=float)
    values = betas[list(FACTORS)].to_numpy() @ loadings(taus, lam).T
    return pd.DataFrame(values, index=betas.index, columns=taus)


def _pooled_sse(panel: FloatArray, maturities: FloatArray, lam: float) -> float:
    design = loadings(maturities, lam)
    coef, *_ = np.linalg.lstsq(design, panel.T, rcond=None)
    resid = panel.T - design @ coef
    return float(np.sum(resid**2))


def pooled_rmse_profile(yields: pd.DataFrame, lams: ArrayLike) -> pd.Series:
    """Pooled RMSE (basis points) of the cross-sectional fit for each decay in ``lams``."""
    panel = yields.to_numpy(dtype=float)
    taus = _maturities(yields)
    grid = np.asarray(lams, dtype=float)
    rmse = [100.0 * np.sqrt(_pooled_sse(panel, taus, lam) / panel.size) for lam in grid]
    return pd.Series(rmse, index=pd.Index(grid, name="lambda"), name="pooled_rmse_bp")


@dataclass(frozen=True)
class NelsonSiegelFit:
    """Estimated decay, factor time series and fitting errors."""

    lam: float
    betas: pd.DataFrame
    residuals: pd.DataFrame
    sse_profile: pd.Series

    @property
    def rmse_bp(self) -> pd.Series:
        """Root-mean-squared fitting error by maturity, in basis points."""
        return 100.0 * np.sqrt((self.residuals**2).mean())

    @property
    def curvature_peak(self) -> float:
        """Maturity (years) at which the curvature loading is maximal."""
        return CURVATURE_ARGMAX / self.lam


def fit(yields: pd.DataFrame, grid: tuple[float, float, int] = (0.05, 1.5, 291)) -> NelsonSiegelFit:
    """Profile least squares for λ on a grid, refined by a bounded scalar search."""
    if yields.isna().to_numpy().any():
        raise ValueError("yield panel contains missing values; drop or fill them first")
    panel = yields.to_numpy(dtype=float)
    taus = _maturities(yields)
    lo, hi, n = grid
    lams = np.linspace(lo, hi, n)
    sse = np.array([_pooled_sse(panel, taus, lam) for lam in lams])
    best = int(np.argmin(sse))
    bracket = (lams[max(best - 1, 0)], lams[min(best + 1, n - 1)])
    refined = minimize_scalar(
        lambda lam: _pooled_sse(panel, taus, lam),
        bounds=bracket,
        method="bounded",
        options={"xatol": 1e-6},
    )
    lam_hat = float(refined.x)
    step = (hi - lo) / (n - 1)
    if min(lam_hat - lo, hi - lam_hat) < step:
        LOGGER.warning(
            "Nelson-Siegel decay %.4f lies at the edge of the search grid [%g, %g]", lam_hat, lo, hi
        )
    betas = fit_betas(yields, lam_hat)
    residuals = yields - fitted_yields(betas, taus, lam_hat).set_axis(yields.columns, axis=1)
    profile = pd.Series(sse, index=pd.Index(lams, name="lambda"), name="sse")
    return NelsonSiegelFit(lam=lam_hat, betas=betas, residuals=residuals, sse_profile=profile)
