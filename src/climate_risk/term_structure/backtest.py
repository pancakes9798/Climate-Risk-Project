"""Pseudo-out-of-sample evaluation of yield-curve forecasts (Diebold & Li, 2006, Section 4).

At every origin the dynamic models are re-estimated on an expanding window and used to forecast
yields h months ahead. Each month's betas depend on that month's curve only. Two inputs are not
real-time: the decay λ is the full-sample estimate (a cross-sectional shape parameter, which
Diebold & Li fix ex ante; estimated on 1986–1999 it reaches the lower bound 0.25, which moves
the relative RMSEs of the shifting-endpoint model by at most 0.02 at 24–60 months and by up to
0.1 at one month, through the cross-sectional fit), and r* is the current-vintage one-sided HLW
estimate, lagged one quarter. The
shifting-endpoint model forecasts the endpoint with its current value (r* and long-run
inflation expectations are martingales in the models that produce them).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from climate_risk.term_structure import nelson_siegel as ns
from climate_risk.term_structure.model import require_contiguous
from climate_risk.term_structure.var import fit_ar1_diagonal, fit_var1

MODELS = ("Random walk", "DNS AR(1)", "DNS VAR(1)", "Shifting-endpoint VAR(1)")

FloatArray = NDArray[np.float64]


def _forecast_paths(
    betas: pd.DataFrame, cycle: pd.DataFrame, istar_now: float, max_h: int, load: FloatArray
) -> dict[str, FloatArray]:
    """Yield forecasts for horizons 1..max_h (shape ``max_h × maturities``) per dynamic model."""
    paths: dict[str, FloatArray] = {}
    for name, estimator in (("DNS AR(1)", fit_ar1_diagonal), ("DNS VAR(1)", fit_var1)):
        model = estimator(betas)
        paths[name] = model.forecast(betas.iloc[-1].to_numpy(), max_h) @ load.T
    se_model = fit_var1(cycle)
    endpoint_now = np.array([istar_now, 0.0, 0.0])
    x_path = se_model.forecast(cycle.iloc[-1].to_numpy(), max_h)
    paths["Shifting-endpoint VAR(1)"] = (x_path + endpoint_now) @ load.T
    return paths


def forecast_errors(
    yields: pd.DataFrame,
    istar: pd.Series,
    lam: float,
    horizons: Sequence[int],
    maturities: Sequence[float],
    first_origin: pd.Period,
) -> pd.DataFrame:
    """Forecast errors (forecast − realised, percentage points) for every model and origin."""
    require_contiguous(yields.index, "the yield panel")
    taus = np.asarray(maturities, dtype=float)
    load = ns.loadings(taus, lam)
    betas = ns.fit_betas(yields, lam)
    aligned_istar = istar.reindex(betas.index)
    cycle_all = betas.assign(level=betas["level"] - aligned_istar)
    require_contiguous(cycle_all.dropna().index, "the endpoint cycle")
    observed = yields[list(taus)].to_numpy()
    months = yields.index
    records: list[dict[str, object]] = []
    for i in np.flatnonzero(months >= first_origin):
        feasible = [h for h in horizons if i + h < len(months)]
        if not feasible or np.isnan(aligned_istar.iloc[i]):
            continue
        history = betas.iloc[: i + 1]
        cycle = cycle_all.iloc[: i + 1].dropna()
        paths = _forecast_paths(history, cycle, float(aligned_istar.iloc[i]), max(feasible), load)
        paths["Random walk"] = np.broadcast_to(observed[i], (max(feasible), taus.size))
        for h in feasible:
            actual = observed[i + h]
            for model in MODELS:
                for tau, err in zip(taus, paths[model][h - 1] - actual, strict=True):
                    records.append(
                        {
                            "origin": months[i],
                            "horizon": h,
                            "model": model,
                            "maturity": tau,
                            "error": float(err),
                        }
                    )
    return pd.DataFrame.from_records(records)


def rmse_table(errors: pd.DataFrame) -> pd.DataFrame:
    """RMSE in basis points, rows (model, horizon), columns maturities."""
    squared = errors.assign(sq=np.square(errors["error"]))
    rmse = 100.0 * np.sqrt(squared.groupby(["model", "horizon", "maturity"])["sq"].mean())
    return rmse.unstack("maturity").reindex(list(MODELS), level="model")


def relative_rmse(table: pd.DataFrame, benchmark: str = "Random walk") -> pd.DataFrame:
    """RMSE of each model divided by the benchmark's at the same horizon and maturity."""
    bench = table.xs(benchmark, level="model")
    return table.div(bench, level="horizon")


def _long_run_variance(d: FloatArray, horizon: int, *, bartlett: bool) -> float:
    centred = d - d.mean()
    n = d.size
    variance = float(centred @ centred) / n
    for lag in range(1, horizon):
        weight = 1.0 - lag / horizon if bartlett else 1.0
        variance += 2.0 * weight * float(centred[lag:] @ centred[:-lag]) / n
    return variance


def diebold_mariano(
    errors: pd.DataFrame,
    model: str,
    horizon: int,
    maturity: float,
    benchmark: str = "Random walk",
) -> tuple[float, float]:
    """Diebold–Mariano test of equal squared-error loss with the Harvey–Leybourne–Newbold
    small-sample correction.

    The long-run variance of the loss differential uses its first h − 1 autocovariances with
    equal weights (the original DM estimator, for which the HLN correction is derived); when that
    estimate is not positive, Bartlett (Newey–West) weights are used instead, which guarantee a
    positive variance. A positive statistic means ``model`` is more accurate than
    ``benchmark``; the p-value is two-sided, from a Student t with T − 1 degrees of freedom.
    """
    sel = errors[(errors["horizon"] == horizon) & (errors["maturity"] == maturity)]
    pivot = sel.pivot_table(index="origin", columns="model", values="error")
    d = (pivot[benchmark] ** 2 - pivot[model] ** 2).dropna().to_numpy()
    n = d.size
    if np.allclose(d, 0.0, atol=1e-15):
        return 0.0, 1.0
    variance = _long_run_variance(d, horizon, bartlett=False)
    if variance <= 0.0:
        variance = _long_run_variance(d, horizon, bartlett=True)
    if variance <= 0.0:  # a constant, non-zero loss differential
        return float(np.sign(d.mean()) * np.inf), 0.0
    statistic = d.mean() / np.sqrt(variance / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic *= correction
    pvalue = 2.0 * stats.t.sf(abs(statistic), df=n - 1)
    return float(statistic), float(pvalue)
