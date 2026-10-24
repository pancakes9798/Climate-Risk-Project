"""Risk-neutral default intensities implied by CDS par spreads.

Standard reduced-form CDS valuation (O'Kane & Turnbull, 2003) with a piecewise-constant hazard
rate between quoted tenors, quarterly premium payments, accrued premium on default and protection
paid at mid-period:

    premium leg   = s · Σⱼ Δⱼ [P(tⱼ) Q(tⱼ) + ½ P(t̄ⱼ) (Q(tⱼ₋₁) − Q(tⱼ))]
    protection    = (1 − R) · Σⱼ P(t̄ⱼ) (Q(tⱼ₋₁) − Q(tⱼ))

where Q is the survival probability, P the risk-free discount factor and t̄ⱼ the mid-point of
period j. The bootstrap solves tenor by tenor for the hazard that reprices each quote.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]
Discount = Callable[[FloatArray], FloatArray]
PAYMENTS_PER_YEAR = 4


@dataclass(frozen=True)
class HazardCurve:
    """Hazard λₖ on (Tₖ₋₁, Tₖ]; the last hazard is extrapolated flat beyond the last tenor."""

    tenors: FloatArray
    hazards: FloatArray

    def cumulative_hazard(self, t: ArrayLike) -> FloatArray:
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        edges = np.concatenate([[0.0], self.tenors])
        widths = np.diff(edges)
        overlap = np.clip(tt[:, None] - edges[:-1], 0.0, widths)
        beyond = np.maximum(tt - self.tenors[-1], 0.0) * self.hazards[-1]
        return overlap @ self.hazards + beyond

    def survival(self, t: ArrayLike) -> FloatArray:
        return np.exp(-self.cumulative_hazard(t))

    def default_probability(self, t: ArrayLike) -> FloatArray:
        return -np.expm1(-self.cumulative_hazard(t))


def to_physical(q_risk_neutral: ArrayLike, ratio: float) -> FloatArray:
    """Physical default probability when risk-neutral intensities are ``ratio`` × actual ones.

    CDS-implied (risk-neutral) intensities embed default-risk premia: Driessen (2005) estimates
    risk-neutral/actual intensity ratios of 1.8 (AA), 2.6 (A) and 2.4 (BBB). Scaling the
    cumulative hazard gives q_P = 1 − (1 − q_Q)^(1/ratio).
    """
    if ratio < 1.0:
        raise ValueError("risk-neutral intensities cannot be below actual ones (ratio >= 1)")
    q = np.asarray(q_risk_neutral, dtype=float)
    return -np.expm1(np.log1p(-q) / ratio)


def _legs(
    curve: HazardCurve, maturity: float, discount: Discount, recovery: float
) -> tuple[float, float]:
    """Risky annuity (RPV01) and protection-leg value of a CDS maturing at ``maturity``."""
    n = max(int(np.ceil(maturity * PAYMENTS_PER_YEAR - 1e-9)), 1)
    times = np.minimum(np.arange(1, n + 1) / PAYMENTS_PER_YEAR, maturity)
    starts = np.concatenate([[0.0], times[:-1]])
    accrual = times - starts
    survival = curve.survival(times)
    survival_prev = np.concatenate([[1.0], survival[:-1]])
    default_mass = survival_prev - survival
    p_pay = discount(times)
    p_mid = discount(0.5 * (starts + times))
    rpv01 = float(np.sum(accrual * (p_pay * survival + 0.5 * p_mid * default_mass)))
    protection = float((1.0 - recovery) * np.sum(p_mid * default_mass))
    return rpv01, protection


def par_spreads(
    curve: HazardCurve, tenors: ArrayLike, discount: Discount, recovery: float
) -> FloatArray:
    """Model par spreads (basis points) for the given maturities."""
    out = []
    for maturity in np.asarray(tenors, dtype=float):
        rpv01, protection = _legs(curve, float(maturity), discount, recovery)
        out.append(1e4 * protection / rpv01)
    return np.asarray(out)


def bootstrap(
    quotes_bp: ArrayLike, tenors: ArrayLike, discount: Discount, recovery: float
) -> HazardCurve:
    """Piecewise-constant hazards that reprice the CDS par spreads ``quotes_bp`` exactly."""
    quotes = np.asarray(quotes_bp, dtype=float)
    mats = np.asarray(tenors, dtype=float)
    if quotes.shape != mats.shape:
        raise ValueError("quotes and tenors must have the same length")
    if np.any(quotes < 0) or np.any(~np.isfinite(quotes)):
        raise ValueError("CDS quotes must be finite and non-negative")
    if not 0.0 <= recovery < 1.0:
        raise ValueError("recovery must lie in [0, 1)")
    if np.any(np.diff(mats) <= 0) or mats[0] <= 0:
        raise ValueError("tenors must be positive and strictly increasing")

    hazards: list[float] = []
    for k, (maturity, quote) in enumerate(zip(mats, quotes, strict=True)):

        def pricing_error(
            lam: float, k: int = k, maturity: float = maturity, quote: float = quote
        ) -> float:
            trial = HazardCurve(tenors=mats[: k + 1], hazards=np.array([*hazards, lam]))
            rpv01, protection = _legs(trial, float(maturity), discount, recovery)
            return 1e4 * protection / rpv01 - quote

        if pricing_error(0.0) > 1e-10:
            raise ValueError(
                f"quote at {maturity:g}y requires a negative forward hazard (static arbitrage)"
            )
        upper = 1.0
        while pricing_error(upper) < 0:
            upper *= 2.0
            if upper > 1e3:
                raise ValueError(f"cannot reprice the {maturity:g}y quote of {quote:g}bp")
        hazards.append(float(brentq(pricing_error, 0.0, upper, xtol=1e-14, rtol=1e-12)))
    return HazardCurve(tenors=mats, hazards=np.asarray(hazards))
