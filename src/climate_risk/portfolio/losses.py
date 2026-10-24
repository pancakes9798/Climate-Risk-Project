"""Default-loss distributions of a bond portfolio and their tail risk measures.

Losses are hold-to-maturity default losses (Battiston & Monasterolo, 2020, Section 3.7):
issuer i defaults before the horizon when its copula uniform falls below its default
probability q_i, losing LGD on its weight. VaR and Expected Shortfall use the Acerbi–Tasche
definition, which remains coherent for discrete loss distributions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def default_losses(u: FloatArray, pds: FloatArray, weights: FloatArray, lgd: float) -> FloatArray:
    """Portfolio loss (fraction of notional) for each simulated row of uniforms."""
    return (np.asarray(u) <= np.asarray(pds)) @ (lgd * np.asarray(weights, dtype=float))


def _weighted_var_es(values: FloatArray, weights: FloatArray, alpha: float) -> tuple[float, float]:
    order = np.argsort(values, kind="stable")
    x, w = values[order], weights[order] / weights.sum()
    cdf = np.cumsum(w)
    k = int(np.searchsorted(cdf, alpha - 1e-12, side="left"))
    var = float(x[k])
    tail = x > var
    es = (np.sum(w[tail] * x[tail]) + var * (np.sum(w[~tail]) - alpha)) / (1.0 - alpha)
    return var, float(es)


def var_es(losses: FloatArray, alpha: float) -> tuple[float, float]:
    """Value-at-Risk and Expected Shortfall at level ``alpha``."""
    sample = np.asarray(losses, dtype=float)
    return _weighted_var_es(sample, np.ones_like(sample), alpha)


def mixture_var_es(
    samples: Sequence[FloatArray], probabilities: Sequence[float], alpha: float
) -> tuple[float, float]:
    """VaR and ES when the scenario itself is uncertain (BM 2020, Section 3.9)."""
    probs = np.asarray(probabilities, dtype=float)
    if not np.isclose(probs.sum(), 1.0) or np.any(probs < 0):
        raise ValueError("scenario probabilities must be non-negative and sum to one")
    values = np.concatenate([np.asarray(s, dtype=float) for s in samples])
    weights = np.concatenate(
        [np.full(len(s), p / len(s)) for s, p in zip(samples, probs, strict=True)]
    )
    keep = weights > 0
    return _weighted_var_es(values[keep], weights[keep], alpha)


def vasicek_quantile(pd_: float, rho: float, alpha: float, lgd: float) -> float:
    """Loss quantile of an infinitely granular homogeneous portfolio (Vasicek, 2002)."""
    return float(
        lgd * norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1.0 - rho))
    )


def basel_correlation(pd_: FloatArray) -> FloatArray:
    """IRB asset correlation for corporate exposures (BCBS, 2006, §272)."""
    weight = (1.0 - np.exp(-50.0 * np.asarray(pd_, dtype=float))) / (1.0 - np.exp(-50.0))
    return 0.12 * weight + 0.24 * (1.0 - weight)
