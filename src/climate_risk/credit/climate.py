"""Climate spreads and the joint rate/credit impact on corporate bond values.

* :func:`calibrate_sector` – bootstrap the reference CDS curve into risk-neutral default
  probabilities q_B(T) and back out the Merton asset volatility from their term structure.
* :func:`climate_spreads` – Battiston–Monasterolo climate spread Δs(T) = s(q_P) − s(q_B).
* :func:`bond_value_impact` – exact split of the change in a zero-coupon corporate bond value
  between the risk-free (term-structure) channel and the credit channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from climate_risk.credit import hazard, merton
from climate_risk.credit.hazard import Discount

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SectorCredit:
    sector: str
    tenors: FloatArray
    quotes_bp: FloatArray
    curve: hazard.HazardCurve
    merton_fit: merton.MertonFit

    @property
    def sigma(self) -> float:
        return self.merton_fit.sigma

    def baseline_pd(self, maturities: ArrayLike) -> FloatArray:
        """Baseline (market-implied) PDs at any maturity; beyond the last quote the hazard is
        held flat."""
        return self.curve.default_probability(maturities)


def calibrate_sector(
    sector: str,
    quotes_bp: ArrayLike,
    tenors: ArrayLike,
    discount: Discount,
    zero_rates: ArrayLike,
    recovery: float,
) -> SectorCredit:
    """Market-implied baseline PDs and the Merton σ consistent with their term structure."""
    quotes = np.array(quotes_bp, dtype=float)  # copies: callers' arrays are never aliased
    mats = np.array(tenors, dtype=float)
    try:
        curve = hazard.bootstrap(quotes, mats, discount, recovery)
        q = curve.default_probability(mats)
        fit = merton.calibrate(q, mats, zero_rates)
    except (ValueError, RuntimeError) as err:
        raise ValueError(f"credit calibration failed for sector {sector!r}: {err}") from err
    return SectorCredit(
        sector=sector,
        tenors=mats,
        quotes_bp=quotes,
        curve=curve,
        merton_fit=fit,
    )


def climate_spreads(
    credit: SectorCredit,
    xi: ArrayLike,
    lgd: float,
    maturities: ArrayLike | None = None,
    rate_shift: ArrayLike = 0.0,
    *,
    rate_drift: bool = True,
) -> pd.DataFrame:
    """Baseline and policy PDs and zero-coupon spreads (bp) for a shock path ξ(T).

    ``rate_shift`` is the repricing Δy(T) of the risk-free zero curve (decimal) that comes with
    the policy, reported as ``riskfree_shift_bp``. With ``rate_drift`` it also moves the Merton
    drift, hence the PDs; ``rate_effect_bp`` isolates that contribution to the climate spread.
    ``maturities`` defaults to the quoted tenors; longer maturities use the flat-hazard
    extrapolation of the baseline curve.
    """
    mats = credit.tenors if maturities is None else np.asarray(maturities, dtype=float)
    q_b = credit.baseline_pd(mats)
    q_shock = merton.shifted_pd(q_b, mats, credit.sigma, xi)
    q_p = merton.shifted_pd(q_b, mats, credit.sigma, xi, rate_shift) if rate_drift else q_shock
    s_b = 1e4 * merton.zero_coupon_spread(q_b, mats, lgd)
    s_p = 1e4 * merton.zero_coupon_spread(q_p, mats, lgd)
    s_shock = 1e4 * merton.zero_coupon_spread(q_shock, mats, lgd)
    return pd.DataFrame(
        {
            "xi": np.asarray(xi, dtype=float),
            "riskfree_shift_bp": 1e4 * np.broadcast_to(np.asarray(rate_shift, float), mats.shape),
            "pd_baseline": q_b,
            "pd_policy": q_p,
            "spread_baseline_bp": s_b,
            "spread_policy_bp": s_p,
            "climate_spread_bp": s_p - s_b,
            "rate_effect_bp": s_p - s_shock,
        },
        index=pd.Index(mats, name="tenor"),
    )


def bond_value_impact(
    rf_baseline: float,
    rf_policy: float,
    spread_baseline: float,
    spread_policy: float,
    maturity: float,
) -> dict[str, float]:
    """Relative change of v = exp(−(r_f + s)·T) from baseline to policy, split exactly into
    rate, credit and interaction terms (inputs in decimal per year)."""
    rate = float(np.expm1(-(rf_policy - rf_baseline) * maturity))
    credit = float(np.expm1(-(spread_policy - spread_baseline) * maturity))
    total = (1.0 + rate) * (1.0 + credit) - 1.0
    return {"rate": rate, "credit": credit, "interaction": total - rate - credit, "total": total}
