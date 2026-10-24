"""Monte Carlo projection of the yield curve under SSP scenarios.

Yields are the sum of a stochastic cycle, simulated once and shared by all scenarios (common
random numbers), and a deterministic scenario overlay:

    y_s(t, τ) = Λ(τ)·xₜ + o_s(t, τ),
    o_s(t, τ) = (1/τ) ∫ₜ^{t+τ} [d_level(s, u) + d_slope(s, u)] du.

The overlay is the expectations-hypothesis effect (term premia unchanged) of the scenario's
deterministic short-rate path: a τ-year yield anticipates the whole path over the life of the
bond. For the reference pathway the endpoint is constant and the overlay is i*_T0 at every
maturity, exactly as in estimation, where the endpoint is a martingale. The same overlay prices
the repricing of bonds when a pathway is revealed (:func:`repricing`), so projections and
repricing are mutually consistent. Every quantile of y_s is the corresponding quantile of the
cycle yield shifted by the overlay, so scenario differences are exact and free of simulation
noise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from climate_risk.term_structure.endpoint import ScenarioPath
from climate_risk.term_structure.model import ShiftingEndpointDNS
from climate_risk.term_structure.nelson_siegel import FACTORS, loadings

FloatArray = NDArray[np.float64]
DEFAULT_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class YieldProjection:
    """Annual (December) yield distribution and mean factor paths for each scenario.

    ``yields`` is indexed by (scenario, year, maturity) with columns ``mean`` and ``qXX``;
    ``factors`` by (scenario, month) with the expected level, slope and curvature.
    """

    yields: pd.DataFrame
    factors: pd.DataFrame


def short_rate(factors: pd.DataFrame) -> pd.Series:
    """Instantaneous short rate of the Nelson–Siegel curve, y(τ → 0) = level + slope."""
    return factors["level"] + factors["slope"]


def overlay(path: ScenarioPath, maturities: ArrayLike) -> FloatArray:
    """Scenario component of yields, months × maturities (percent).

    The deterministic shifts move the expected short rate by d_level + d_slope; the τ-year yield
    at month t moves by its average over months t … t + 12τ − 1. Beyond the last projected month
    the shift is held at its final value.
    """
    taus = np.asarray(maturities, dtype=float)
    windows = np.rint(12 * taus).astype(int)
    if np.any(windows < 1):
        raise ValueError("maturities must be at least one month")
    short = path.shift[:, 0] + path.shift[:, 1]
    extended = np.concatenate([short, np.full(int(windows.max()), short[-1])])
    cumulative = np.concatenate([[0.0], np.cumsum(extended)])
    start = np.arange(short.size)
    return np.column_stack([(cumulative[start + w] - cumulative[start]) / w for w in windows])


def repricing(
    policy: ScenarioPath, baseline: ScenarioPath, maturities: ArrayLike, month: int = 0
) -> FloatArray:
    """Change of the τ-year yields (percent) when the market switches from baseline to policy
    at the ``month``-th projected month (0 = immediately after the jump-off)."""
    if not policy.months.equals(baseline.months):
        raise ValueError("scenario paths are defined on different months")
    return overlay(policy, maturities)[month] - overlay(baseline, maturities)[month]


def quantile_label(q: float) -> str:
    """Column label of a quantile: 0.05 → ``q05``, 0.5 → ``q50``, 0.025 → ``q2_5``."""
    pct = 100.0 * q
    if abs(pct - round(pct)) < 1e-9:
        return f"q{round(pct):02d}"
    return "q" + f"{pct:g}".replace(".", "_")


def project(
    model: ShiftingEndpointDNS,
    x0: FloatArray,
    paths: Mapping[str, ScenarioPath],
    maturities: Sequence[float],
    *,
    n_paths: int = 2000,
    seed: int = 0,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> YieldProjection:
    """Simulate the cycle from ``x0`` and attach every scenario's endpoint path."""
    months = next(iter(paths.values())).months
    for name, path in paths.items():
        if not path.months.equals(months):
            raise ValueError(f"scenario {name!r} is defined on different months")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((len(months), n_paths, model.var.k))
    cycle = model.var.simulate(np.asarray(x0, dtype=float), shocks)  # (months, paths, k)
    mean_cycle = model.var.forecast(np.asarray(x0, dtype=float), len(months))

    december = np.flatnonzero(months.month == 12)
    taus = np.asarray(maturities, dtype=float)
    load = loadings(taus, model.lam)  # (m, k)
    cycle_yields = np.einsum("tpk,mk->tpm", cycle[december], load)  # (years, paths, m)
    labels = [quantile_label(q) for q in quantiles]
    if len(set(labels)) != len(labels):
        raise ValueError(f"quantiles {tuple(quantiles)} do not have distinct labels")
    stats = {"mean": mean_cycle[december] @ load.T}
    for q, label in zip(quantiles, labels, strict=True):
        stats[label] = np.quantile(cycle_yields, q, axis=1)

    years = months[december].year.to_numpy()
    yield_frames, factor_frames = [], []
    for name, path in paths.items():
        shift_yields = overlay(path, taus)[december]  # (years, m)
        index = pd.MultiIndex.from_product(
            [[name], years, taus], names=["scenario", "year", "maturity"]
        )
        columns = {label: (values + shift_yields).ravel() for label, values in stats.items()}
        yield_frames.append(pd.DataFrame(columns, index=index))
        factor_index = pd.MultiIndex.from_product([[name], months], names=["scenario", "month"])
        factor_frames.append(
            pd.DataFrame(mean_cycle + path.shift, index=factor_index, columns=list(FACTORS))
        )
    return YieldProjection(yields=pd.concat(yield_frames), factors=pd.concat(factor_frames))
