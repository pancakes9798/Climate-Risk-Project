"""Stage 3 – default-loss distribution of a bond portfolio under each transition scenario.

Default probabilities come from the CDS curves, i.e. they are risk-neutral. The loss
distribution is reported under both measures: the *physical* one (CDS-implied intensities
divided by the risk-premium ratio of Driessen, 2005) is the risk-management view; the
risk-neutral one is the market-implied view. The climate shock moves the Merton default
threshold, which is the same under both measures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from climate_risk.config import Config
from climate_risk.credit import cds, hazard, merton, transition
from climate_risk.pipeline.credit import CreditResults, revelation_now
from climate_risk.pipeline.rates import RatesResults
from climate_risk.portfolio import copula, losses
from climate_risk.term_structure.projection import repricing

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]
BASELINE = "Baseline (market-implied)"
PHYSICAL, RISK_NEUTRAL = "Physical", "Risk-neutral"
DEMO_ISSUERS_PER_SECTOR = 20
BATCH = 25_000


@dataclass(frozen=True)
class PortfolioResults:
    mode: str
    labels: pd.Series  # issuer → sector
    blocks: pd.DataFrame  # average asset correlation within/between sectors
    nu: float | None
    pds: pd.DataFrame  # issuers × (measure, scenario): default probability over the horizon
    risk: pd.DataFrame
    mixture: pd.DataFrame


def _issuer_pds(cfg: Config, credit: CreditResults) -> tuple[pd.Series, pd.Series]:
    """Risk-neutral baseline horizon PDs per issuer and their sector labels."""
    horizon = np.array([float(cfg.portfolio.horizon)])
    pds, labels = {}, {}
    if credit.mode == "cds" and credit.issuer_quotes is not None:
        left_out = []
        for ticker, quotes in credit.issuer_quotes.iterrows():
            try:
                curve = hazard.bootstrap(
                    quotes.to_numpy(), cds.TENOR_YEARS, credit.discount, cfg.credit.recovery
                )
            except ValueError as err:
                LOGGER.warning("issuer %s left out of the portfolio: %s", ticker, err)
                left_out.append(ticker)
                continue
            pds[ticker] = float(curve.default_probability(horizon)[0])
            labels[ticker] = credit.sectors[ticker]
        if left_out:
            LOGGER.warning(
                "%d of %d issuers left out of the portfolio: their CDS curves imply negative "
                "forward hazards, which biases the portfolio PD downwards",
                len(left_out),
                len(credit.issuer_quotes),
            )
        return pd.Series(pds), pd.Series(labels)
    for sector, sector_credit in credit.sector_credit.items():
        for k in range(DEMO_ISSUERS_PER_SECTOR):
            name = f"{sector} #{k + 1}"
            pds[name] = float(sector_credit.baseline_pd(horizon)[0])
            labels[name] = sector
    return pd.Series(pds), pd.Series(labels)


def _basel_matrix(cfg: Config, labels: pd.Series, base_pds: pd.Series) -> pd.DataFrame:
    """Equicorrelation at the Basel IRB asset correlation of the average 1y physical PD."""
    physical = hazard.to_physical(base_pds.to_numpy(), cfg.portfolio.risk_premium_ratio)
    one_year = 1.0 - (1.0 - physical.mean()) ** (1.0 / cfg.portfolio.horizon)
    rho = float(losses.basel_correlation(np.array([one_year]))[0])
    n = len(labels)
    return pd.DataFrame(
        np.full((n, n), rho) + (1 - rho) * np.eye(n), index=labels.index, columns=labels.index
    )


def _dependence(
    cfg: Config, credit: CreditResults, labels: pd.Series, base_pds: pd.Series
) -> tuple[dict[str, tuple[pd.DataFrame, float | None]], pd.DataFrame, float | None]:
    """Copula specifications to compare, the sector correlation table and the fitted ν.

    With the CDS panel: Student t and Gaussian copulas with the CDS-implied block correlation,
    plus the Basel IRB benchmark. CDS-spread co-movements also reflect common shocks to risk
    premia, so they tend to overstate asset (default) correlation; the regulatory benchmark
    brackets the result from below.
    """
    basel = _basel_matrix(cfg, labels, base_pds)
    benchmark: dict[str, tuple[pd.DataFrame, float | None]] = {
        "Gaussian (Basel IRB ρ)": (basel, None)
    }
    sectors = list(dict.fromkeys(labels))
    if credit.mode == "cds" and credit.weekly_changes is not None:
        changes = credit.weekly_changes[list(labels.index)]
        tau = copula.kendall_tau_matrix(changes)
        rho = pd.DataFrame(copula.tau_to_rho(tau.to_numpy()), index=tau.index, columns=tau.columns)
        full, blocks = copula.block_correlation(rho, labels)
        nu = copula.fit_nu(copula.pseudo_observations(changes), full.to_numpy())
        LOGGER.info("t-copula: nu = %.2f on %d weekly observations", nu, len(changes))
        specs = {"Student t (CDS)": (full, nu), "Gaussian (CDS)": (full, None), **benchmark}
        return specs, blocks, nu
    rho_basel = float(basel.iat[0, 1])
    return benchmark, pd.DataFrame(rho_basel, index=sectors, columns=sectors), None


def _scenario_pds(
    cfg: Config,
    frame: pd.DataFrame,
    credit: CreditResults,
    rates: RatesResults,
    base: pd.Series,
    labels: pd.Series,
) -> pd.DataFrame:
    """Horizon PDs per issuer for every (measure, scenario).

    The asset shock and the risk-free repricing move the Merton threshold by the same amount
    under both measures (constant asset risk premium).
    """
    horizon = np.array([float(cfg.portfolio.horizon)])
    t0 = rates.jump_off.t0
    baselines = {
        RISK_NEUTRAL: base,
        PHYSICAL: pd.Series(
            hazard.to_physical(base.to_numpy(), cfg.portfolio.risk_premium_ratio), index=base.index
        ),
    }
    columns: dict[tuple[str, str], pd.Series] = {}
    for measure, q_base in baselines.items():
        columns[(measure, BASELINE)] = q_base
        for baseline, policy in cfg.scenarios.transitions:
            exposures = transition.sector_exposures(frame, baseline, t0)
            shocks = {
                sector: transition.asset_shock(
                    frame,
                    baseline,
                    policy,
                    exposures[sector],
                    revelation_now(rates.paths) + horizon,
                    cfg.credit.chi,
                    t0,
                    cfg.scenarios.convergence_year,
                )
                for sector in credit.sector_credit
            }
            dy = (
                repricing(rates.paths[policy], rates.paths[baseline], horizon) / 100.0
                if cfg.credit.include_rate_drift
                else np.zeros(1)
            )
            shocked = {
                issuer: float(
                    merton.shifted_pd(
                        np.array([q]),
                        horizon,
                        credit.sector_credit[labels[issuer]].sigma,
                        shocks[labels[issuer]],
                        dy,
                    )[0]
                )
                for issuer, q in q_base.items()
            }
            columns[(measure, policy)] = pd.Series(shocked)
    out = pd.DataFrame(columns)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["measure", "scenario"])
    return out


def _simulate(
    cfg: Config, R: FloatArray, nu: float | None, pds: pd.DataFrame, weights: FloatArray, seed: int
) -> dict[tuple[str, str], FloatArray]:
    """Loss samples per (measure, scenario), all driven by the same copula draws."""
    rng = np.random.default_rng(seed)
    lgd = 1.0 - cfg.credit.recovery
    chunks: dict[tuple[str, str], list[FloatArray]] = {key: [] for key in pds.columns}
    remaining = cfg.portfolio.n_sims
    while remaining > 0:
        n = min(BATCH, remaining)
        u = copula.simulate_gaussian(R, n, rng) if nu is None else copula.simulate_t(R, nu, n, rng)
        for key in pds.columns:
            chunks[key].append(losses.default_losses(u, pds[key].to_numpy(), weights, lgd))
        remaining -= n
    return {key: np.concatenate(parts) for key, parts in chunks.items()}


def run(
    cfg: Config, frame: pd.DataFrame, credit: CreditResults, rates: RatesResults
) -> PortfolioResults:
    base, labels = _issuer_pds(cfg, credit)
    specs, blocks, nu = _dependence(cfg, credit, labels, base)
    pds = _scenario_pds(cfg, frame, credit, rates, base, labels)
    weights = np.full(len(base), 1.0 / len(base))
    alpha = cfg.portfolio.alpha
    rows, samples_by_copula = [], {}
    for name, (matrix, dof) in specs.items():
        samples = _simulate(cfg, matrix.to_numpy(), dof, pds, weights, cfg.portfolio.seed)
        samples_by_copula[name] = samples
        for (measure, scenario), sample in samples.items():
            var, es = losses.var_es(sample, alpha)
            rows.append(
                {
                    "nu": dof,
                    "copula": name,
                    "measure": measure,
                    "scenario": scenario,
                    "expected_loss": float(sample.mean()),
                    "var": var,
                    "es": es,
                }
            )
    main = next(iter(specs))
    adverse = cfg.scenarios.headline[1]
    mix_rows = []
    for p in np.linspace(0.0, 1.0, 21):
        main_samples = samples_by_copula[main]
        var, es = losses.mixture_var_es(
            [main_samples[(PHYSICAL, BASELINE)], main_samples[(PHYSICAL, adverse)]],
            [1 - p, p],
            alpha,
        )
        mix_rows.append(
            {
                "copula": main,
                "measure": PHYSICAL,
                "adverse": adverse,
                "p_adverse": p,
                "var": var,
                "es": es,
            }
        )
    return PortfolioResults(
        mode=credit.mode,
        labels=labels,
        blocks=blocks,
        nu=nu,
        pds=pds,
        risk=pd.DataFrame(rows),
        mixture=pd.DataFrame(mix_rows),
    )


def tables(res: PortfolioResults) -> dict[str, pd.DataFrame]:
    sector_pds = res.pds.groupby(res.labels).mean().T
    sector_pds.columns.name = "sector"
    return {
        "portfolio_correlation_blocks": res.blocks,
        "portfolio_sector_pds": sector_pds,
        "portfolio_risk": res.risk.assign(mode=res.mode),
        "portfolio_scenario_mixture": res.mixture,
    }
