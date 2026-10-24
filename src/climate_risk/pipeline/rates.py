"""Stage 1 – the scenario-conditional term structure."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from climate_risk.config import SCENARIO_LAST_YEAR, SUMMARY_MATURITY, Config
from climate_risk.data import store
from climate_risk.term_structure import backtest
from climate_risk.term_structure import nelson_siegel as ns
from climate_risk.term_structure.endpoint import JumpOff, ScenarioPath, scenario_path
from climate_risk.term_structure.model import ShiftingEndpointDNS, endpoint_cycle, fit_model
from climate_risk.term_structure.projection import (
    YieldProjection,
    overlay,
    project,
    repricing,
)
from climate_risk.term_structure.var import fit_var1

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]
SNAPSHOT_YEARS = (2030, 2050, 2075, 2100)


@dataclass(frozen=True)
class RatesResults:
    sample: pd.DataFrame
    fit: ns.NelsonSiegelFit
    drivers: pd.DataFrame
    model: ShiftingEndpointDNS
    jump_off: JumpOff
    x0: FloatArray
    paths: dict[str, ScenarioPath]
    projection: YieldProjection
    errors: pd.DataFrame


def scenario_set(cfg: Config) -> list[str]:
    """Every scenario needed downstream, in a stable order."""
    names = [*cfg.scenarios.narratives, *cfg.scenarios.policy_family]
    names += [s for pair in cfg.scenarios.transitions for s in pair]
    return list(dict.fromkeys(names))


def scenario_paths(
    cfg: Config,
    frame: pd.DataFrame,
    jump: JumpOff,
    months: pd.PeriodIndex,
    names: Sequence[str] | None = None,
) -> dict[str, ScenarioPath]:
    """Endpoint paths of ``names`` (default: every scenario used downstream)."""
    convergence = cfg.scenarios.convergence_year
    if convergence is None:
        raise ValueError("the rates stage needs a harmonisation convergence year")
    return {
        name: scenario_path(
            frame,
            name,
            months,
            jump,
            cfg.endpoint,
            cfg.scenarios.reference,
            convergence,
        )
        for name in (scenario_set(cfg) if names is None else names)
    }


def _require_jump_off(data: pd.DataFrame, month: pd.Period, what: str) -> None:
    if month not in data.index or data.loc[month].isna().any():
        raise ValueError(
            f"the jump-off month {month} is not covered by the {what} "
            f"({data.index[0]} to {data.index[-1]}): change term_structure.as_of or refresh "
            "the data with `climate-risk download`"
        )


def run(cfg: Config, frame: pd.DataFrame, *, with_backtest: bool = True) -> RatesResults:
    ts = cfg.term_structure
    as_of = pd.Period(ts.as_of, "M")
    curves = store.load_curves(cfg.paths)
    _require_jump_off(curves[list(ts.maturities)], as_of, "Treasury curves")
    sample = curves.loc[pd.Period(ts.sample_start, "M") : as_of, list(ts.maturities)]
    fit = ns.fit(sample, ts.lambda_grid)
    LOGGER.info("Nelson-Siegel decay %.4f/yr (curvature peak %.2fy)", fit.lam, fit.curvature_peak)

    drivers = store.load_endpoint_drivers(cfg.paths)
    _require_jump_off(drivers, as_of, "endpoint drivers (r*, π*)")
    model = fit_model(fit.betas, drivers["istar"], fit.lam)
    jump = JumpOff(
        month=as_of,
        rstar=float(drivers.loc[as_of, "rstar"]),
        inflation=float(drivers.loc[as_of, "inflation"]),
    )
    x0 = endpoint_cycle(fit.betas, drivers["istar"]).loc[as_of].to_numpy()
    months = pd.period_range(as_of + 1, f"{SCENARIO_LAST_YEAR}-12", freq="M", name="month")
    paths = scenario_paths(cfg, frame, jump, months)
    sim = cfg.simulation
    projection = project(
        model,
        x0,
        paths,
        ts.output_maturities,
        n_paths=sim.n_paths,
        seed=sim.seed,
        quantiles=sim.quantiles,
    )

    errors = pd.DataFrame()
    if with_backtest:
        bt = cfg.backtest
        errors = backtest.forecast_errors(
            sample,
            drivers["istar"],
            fit.lam,
            bt.horizons,
            bt.maturities,
            pd.Period(bt.first_origin, "M"),
        )
    return RatesResults(
        sample=sample,
        fit=fit,
        drivers=drivers,
        model=model,
        jump_off=jump,
        x0=x0,
        paths=paths,
        projection=projection,
        errors=errors,
    )


def _var_table(model: ShiftingEndpointDNS) -> pd.DataFrame:
    """Coefficients of the cycle VAR, one row per factor equation."""
    var = model.var
    table = pd.DataFrame(var.phi, index=list(var.names), columns=[f"phi_{n}" for n in var.names])
    table["intercept"] = var.intercept
    table["mean"] = var.mean
    table["shock_sd"] = np.sqrt(np.diag(var.sigma))
    table["unconditional_sd"] = np.sqrt(np.diag(var.unconditional_covariance()))
    table.index.name = "equation"
    return table


def _modes_table(res: RatesResults) -> pd.DataFrame:
    """Eigen-modes of the shifting-endpoint cycle VAR and of a constant-mean VAR on the betas."""
    rows = []
    for label, var in (
        ("Shifting-endpoint VAR(1) (cycle)", res.model.var),
        ("Constant-mean VAR(1) (betas)", fit_var1(res.fit.betas)),
    ):
        modulus = np.abs(var.eigenvalues)
        order = np.argsort(-modulus)
        for rank, k in enumerate(order, start=1):
            rows.append(
                {
                    "model": label,
                    "mode": rank,
                    "eigenvalue_modulus": modulus[k],
                    "half_life_months": var.half_lives[k],
                }
            )
    return pd.DataFrame(rows).set_index(["model", "mode"])


def _dm_table(errors: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for model in backtest.MODELS[1:]:
        for h in cfg.backtest.horizons:
            for maturity in cfg.backtest.maturities:
                stat, pvalue = backtest.diebold_mariano(errors, model, h, maturity)
                rows.append(
                    {
                        "model": model,
                        "horizon": h,
                        "maturity": maturity,
                        "dm_stat": stat,
                        "p_value": pvalue,
                    }
                )
    return pd.DataFrame(rows)


def sensitivity(cfg: Config, frame: pd.DataFrame, res: RatesResults) -> pd.DataFrame:
    """One-at-a-time sensitivity of the scenario effect on the 10-year yield.

    For every scenario: the repricing at revelation (EH overlay gap at the jump-off) and the gap
    in the expected 10y yield vs the reference in selected years, all in basis points.
    """
    months = next(iter(res.paths.values())).months
    variants: list[tuple[str, Config]] = [("central", cfg)]
    variants += [
        (f"theta = {theta:g}", replace(cfg, endpoint=replace(cfg.endpoint, theta=theta)))
        for theta in (0.5, 1.5)
    ]
    variants.append(
        ("kappa = 0 (look-through)", replace(cfg, endpoint=replace(cfg.endpoint, kappa=0.0)))
    )
    variants += [
        (
            f"convergence {year}",
            replace(cfg, scenarios=replace(cfg.scenarios, convergence_year=year)),
        )
        for year in (2040, 2060, 2070)
    ]
    years = (2030, 2050, 2100)
    index = {year: int(np.flatnonzero(months == pd.Period(f"{year}-12", "M"))[0]) for year in years}
    rows = []
    for label, alt in variants:
        paths = scenario_paths(alt, frame, res.jump_off, months)
        ref = overlay(paths[cfg.scenarios.reference], [10.0])[:, 0]
        baselines = {pair[1]: pair[0] for pair in cfg.scenarios.transitions}
        for name, path in paths.items():
            gap = 100.0 * (overlay(path, [10.0])[:, 0] - ref)
            row: dict[str, object] = {"variant": label, "scenario": name}
            if name in baselines:
                row["repricing_10y_bp"] = 100.0 * float(
                    repricing(path, paths[baselines[name]], [10.0])[0]
                )
            row |= {f"gap_10y_{year}_bp": gap[i] for year, i in index.items()}
            rows.append(row)
    return pd.DataFrame(rows)


def tables(cfg: Config, frame: pd.DataFrame, res: RatesResults) -> dict[str, pd.DataFrame]:
    fit = res.fit
    lo, hi, _ = cfg.term_structure.lambda_grid
    wide = ns.pooled_rmse_profile(res.sample, np.round(np.arange(0.05, 1.5001, 0.005), 3))
    profile = wide.to_frame().assign(admissible=lambda f: (f.index >= lo) & (f.index <= hi))
    summary = pd.DataFrame(
        {
            "value": [
                fit.lam,
                fit.curvature_peak,
                len(res.sample),
                res.jump_off.rstar,
                res.jump_off.inflation,
                res.jump_off.istar,
            ]
        },
        index=[
            "lambda_per_year",
            "curvature_peak_years",
            "months",
            "rstar_jump_off",
            "expected_inflation_jump_off",
            "istar_jump_off",
        ],
    )
    december = {
        name: path.frame()
        .loc[path.months.month == 12]
        .assign(year=lambda f: f.index.year)
        .set_index("year")
        for name, path in res.paths.items()
    }
    endpoint = pd.concat(december, names=["scenario"])
    yields = res.projection.yields
    out = {
        "rates_summary": summary,
        "ns_rmse_by_maturity": fit.rmse_bp.rename("rmse_bp").to_frame(),
        "ns_lambda_profile": profile,
        "cycle_var": _var_table(res.model),
        "cycle_var_modes": _modes_table(res),
        "endpoint_paths": endpoint,
        "yield_projection_10y": yields.xs(SUMMARY_MATURITY, level="maturity"),
        "yield_curve_snapshots": yields[yields.index.get_level_values("year").isin(SNAPSHOT_YEARS)],
        "rates_sensitivity": sensitivity(cfg, frame, res),
    }
    if not res.errors.empty:
        rmse = backtest.rmse_table(res.errors)
        out["backtest_rmse_bp"] = rmse
        out["backtest_relative_rmse"] = backtest.relative_rmse(rmse)
        out["backtest_bias_bp"] = (
            100 * res.errors.groupby(["model", "horizon", "maturity"])["error"].mean()
        ).unstack("maturity")
        out["backtest_diebold_mariano"] = _dm_table(res.errors, cfg)
    return out
