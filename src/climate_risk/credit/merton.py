"""Merton (1974) default threshold and the Battiston–Monasterolo climate-policy shock.

Under the risk-neutral measure, with a deterministic short rate, log assets at horizon T are
normal and the probability of ending below the debt barrier D is

    q(T) = Φ( −[a + R(T) − σ²T/2] / (σ√T) ),   a = ln(V₀/D),   R(T) = ∫₀ᵀ r = T · y(T),

with y the zero-coupon yield. Battiston & Monasterolo (2020, eqs. 11–14) model a climate-policy
shock ξ as a jump of assets, A_T → A_T (1 + ξ). In the lognormal model this shifts the default
threshold by ln(1 + ξ); a change Δy(T) of the zero yield moves the drift by T·Δy(T). Hence

    q_P(T) = Φ( Φ⁻¹(q_B(T)) − [ln(1 + ξ) + T·Δy(T)] / (σ√T) ).

Only the asset volatility σ is needed to map a shock into default probabilities; the baseline
term structure q_B is taken directly from the market (CDS), not from the fitted model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.stats import norm

FloatArray = NDArray[np.float64]
_MIN_ASSET_FACTOR = 1e-12


_BOUNDS = ([1e-6, 0.01], [10.0, 2.0])  # (a, σ)
LOGGER = logging.getLogger(__name__)


def _probit(a: float, sigma: float, T: FloatArray, zero_rates: FloatArray) -> FloatArray:
    """Φ⁻¹ of the Merton default probability."""
    return -(a + zero_rates * T - 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))


def merton_pd(a: float, sigma: float, T: ArrayLike, zero_rates: ArrayLike) -> FloatArray:
    """Risk-neutral probability that assets end below the barrier at each horizon."""
    tt = np.asarray(T, dtype=float)
    return norm.cdf(_probit(a, sigma, tt, np.asarray(zero_rates, dtype=float)))


@dataclass(frozen=True)
class MertonFit:
    a: float
    sigma: float
    rmse_probit: float

    @property
    def leverage(self) -> float:
        """Implied debt-to-assets ratio D/V₀."""
        return float(np.exp(-self.a))


def calibrate(q: ArrayLike, T: ArrayLike, zero_rates: ArrayLike) -> MertonFit:
    """Fit (a, σ) to a default-probability term structure by least squares in probit space."""
    pds = np.asarray(q, dtype=float)
    if not np.all(np.isfinite(pds)) or np.any((pds <= 0) | (pds >= 1)):
        raise ValueError("default probabilities must be finite and strictly between 0 and 1")
    target = norm.ppf(pds)
    tt = np.asarray(T, dtype=float)
    rr = np.asarray(zero_rates, dtype=float)

    def residuals(params: FloatArray) -> FloatArray:
        return _probit(params[0], params[1], tt, rr) - target

    solution = least_squares(
        residuals, x0=np.array([0.5, 0.2]), bounds=_BOUNDS, xtol=1e-14, ftol=1e-14
    )
    if not solution.success:
        raise RuntimeError(f"Merton calibration did not converge: {solution.message}")
    a_hat, sigma_hat = solution.x
    if np.isclose(sigma_hat, _BOUNDS[0][1]) or np.isclose(sigma_hat, _BOUNDS[1][1]):
        LOGGER.warning("Merton asset volatility at its bound (%.3f)", sigma_hat)
    rmse = float(np.sqrt(np.mean(solution.fun**2)))
    return MertonFit(a=float(a_hat), sigma=float(sigma_hat), rmse_probit=rmse)


def shifted_pd(
    q: ArrayLike, T: ArrayLike, sigma: float, xi: ArrayLike, rate_shift: ArrayLike = 0.0
) -> FloatArray:
    """Default probabilities after an asset shock ξ and a change Δy of the T-year zero yield.

    With face value fixed, as in the calibration, Φ⁻¹(q) moves by −[ln(1 + ξ) + T·Δy]/(σ√T):
    lower assets raise the PD, a higher risk-free drift lowers it. ξ = −1 wipes the assets out.
    """
    tt = np.asarray(T, dtype=float)
    factor = np.maximum(1.0 + np.asarray(xi, dtype=float), _MIN_ASSET_FACTOR)
    shift = (np.log(factor) + tt * np.asarray(rate_shift, dtype=float)) / (sigma * np.sqrt(tt))
    return norm.cdf(norm.ppf(np.asarray(q, dtype=float)) - shift)


def zero_coupon_spread(q: ArrayLike, T: ArrayLike, lgd: float) -> FloatArray:
    """Battiston–Monasterolo (2020, eqs. 7–8): e^{−sT} = 1 − q·LGD (decimal per year)."""
    return -np.log1p(-np.asarray(q, dtype=float) * lgd) / np.asarray(T, dtype=float)
