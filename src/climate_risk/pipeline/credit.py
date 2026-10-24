"""Stage 2 – climate spreads and the joint rate/credit repricing of corporate bonds."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from climate_risk.config import CDS_PATH_ENV, Config
from climate_risk.credit import cds, climate, transition
from climate_risk.data import store
from climate_risk.pipeline.rates import RatesResults, scenario_paths
from climate_risk.term_structure import nelson_siegel as ns
from climate_risk.term_structure.endpoint import ScenarioPath, decimal_year
from climate_risk.term_structure.projection import repricing

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]
SPREAD_MATURITIES = np.array([*range(1, 11), 15, 20, 30], dtype=float)
BOND_MATURITIES = (5.0, 10.0, 30.0)
DELAY_MATURITIES = (5.0, 10.0)
DELAY_LAST_YEAR = 2050
MIN_WEEKS = 52  # complete weeks of CDS changes needed for the dependence estimates


@dataclass(frozen=True)
class TreasuryCurve:
    """Nelson–Siegel zero curve with fixed factors, used as discount function (picklable)."""

    betas: pd.Series
    lam: float

    def zero_rates(self, t: ArrayLike) -> FloatArray:
        """Continuously compounded zero rates, decimal."""
        return ns.zero_rates(self.betas, np.asarray(t, dtype=float), self.lam) / 100.0

    def __call__(self, t: FloatArray) -> FloatArray:
        tt = np.maximum(np.asarray(t, dtype=float), 1e-6)
        return np.exp(-self.zero_rates(tt) * tt)


@dataclass(frozen=True)
class MarketInputs:
    """Credit inputs: the licensed CDS panel (mode "cds") or the public Baa spread ("demo")."""

    mode: str
    sectors: pd.Series  # ticker → sector
    reference: pd.DataFrame  # sector × PX1..PX10 (bp)
    issuer_quotes: pd.DataFrame | None  # licensed: kept in memory, never written
    weekly_changes: pd.DataFrame | None


@dataclass(frozen=True)
class CreditResults:
    mode: str  # "cds" (licensed panel) or "demo" (public Baa spread)
    sectors: pd.Series
    reference: pd.DataFrame  # sector × PX1..PX10 (bp)
    issuer_quotes: pd.DataFrame | None  # licensed: kept in memory, never written
    weekly_changes: pd.DataFrame | None
    sector_credit: dict[str, climate.SectorCredit]
    spreads: pd.DataFrame
    delayed: pd.DataFrame
    bond_values: pd.DataFrame
    sensitivity: pd.DataFrame
    discount: TreasuryCurve


def _window_curve(cfg: Config, rates: RatesResults) -> TreasuryCurve:
    """Average Treasury zero curve over the calibration window."""
    start, end = (pd.Period(p, "M") for p in cfg.credit.window)
    window = rates.fit.betas.loc[start:end]
    if window.empty:
        raise ValueError(f"no Treasury curves in the credit calibration window {start}–{end}")
    return TreasuryCurve(betas=window.mean(), lam=rates.fit.lam)


def _cds_inputs(cfg: Config, sectors: pd.Series) -> MarketInputs:
    start, end = cfg.credit.window
    raw = cds.load_cds(cfg.paths.cds_csv, tickers=sectors.index)
    issuers = cds.reference_spreads(
        cds.monthly_panel(raw, start, end), min_coverage=cfg.credit.min_coverage
    )
    if issuers.empty:
        raise ValueError(
            f"no issuer is quoted in at least {cfg.credit.min_coverage:.0%} of the months of "
            f"{start}–{end}"
        )
    changes = cds.weekly_log_changes(raw, "PX5", start, end, tickers=issuers.index)
    stale = cds.stale_issuers(changes, cfg.credit.max_stale_share)
    if stale:
        LOGGER.warning("dropping issuers with stale quotes: %s", ", ".join(stale))
    issuers = issuers.drop(index=stale)
    changes = changes.drop(columns=stale)
    if len(changes) < MIN_WEEKS:
        raise ValueError(
            f"only {len(changes)} complete weeks of CDS changes in {start}–{end}; "
            f"at least {MIN_WEEKS} are needed for the dependence estimates"
        )
    LOGGER.info("CDS mode: %d issuers, %d weekly observations", len(issuers), len(changes))
    return MarketInputs(
        mode="cds",
        sectors=sectors,
        reference=cds.sector_reference(issuers, sectors),
        issuer_quotes=issuers,
        weekly_changes=changes,
    )


def _market_inputs(cfg: Config) -> MarketInputs:
    if cfg.paths.cds_override is not None and not cfg.paths.cds_csv.exists():
        raise FileNotFoundError(
            f"${CDS_PATH_ENV} points to {cfg.paths.cds_csv}, which does not exist"
        )
    sectors = cds.load_sector_map(cfg.paths.sectors_csv)
    sectors = sectors[sectors.isin(transition.SECTORS)]
    load = _cds_inputs if cfg.paths.cds_csv.exists() else _demo_inputs
    inputs = load(cfg, sectors)
    missing = sorted(set(transition.SECTORS) - set(inputs.reference.index))
    if missing:
        raise ValueError(f"no eligible CDS issuers in sectors {missing}")
    return inputs


def _demo_inputs(cfg: Config, sectors: pd.Series) -> MarketInputs:
    start, end = cfg.credit.window
    LOGGER.warning(
        "licensed CDS panel not found at %s: DEMO mode with the public Baa spread "
        "(flat curve, identical across sectors)",
        cfg.paths.cds_csv,
    )
    baa = store.load_baa_spread(cfg.paths).loc[pd.Period(start, "M") : pd.Period(end, "M")]
    if baa.empty:
        raise ValueError(f"no Moody's Baa observations in the calibration window {start}–{end}")
    level_bp = 100.0 * float(baa.median())
    reference = pd.DataFrame(level_bp, index=list(transition.SECTORS), columns=list(cds.PX))
    return MarketInputs(
        mode="demo", sectors=sectors, reference=reference, issuer_quotes=None, weekly_changes=None
    )


def _rate_shift(
    paths: dict[str, ScenarioPath],
    baseline: str,
    policy: str,
    maturities: FloatArray,
    month: int = 0,
) -> FloatArray:
    """Risk-free repricing Δy(T) (decimal) when ``policy`` replaces ``baseline``."""
    return repricing(paths[policy], paths[baseline], maturities, month) / 100.0


def _spreads(
    cfg: Config,
    frame: pd.DataFrame,
    credits: dict[str, climate.SectorCredit],
    paths: dict[str, ScenarioPath],
    t0: float,
) -> pd.DataFrame:
    lgd = 1.0 - cfg.credit.recovery
    t_conv = cfg.scenarios.convergence_year
    rows = []
    for baseline, policy in cfg.scenarios.transitions:
        exposures = transition.sector_exposures(frame, baseline, t0)
        dy = _rate_shift(paths, baseline, policy, SPREAD_MATURITIES)
        for sector, credit in credits.items():
            for chi in cfg.credit.chi_sensitivity:
                xi = transition.asset_shock(
                    frame,
                    baseline,
                    policy,
                    exposures[sector],
                    revelation_now(paths) + SPREAD_MATURITIES,
                    chi,
                    t0,
                    t_conv,
                )
                table = climate.climate_spreads(
                    credit,
                    xi,
                    lgd,
                    SPREAD_MATURITIES,
                    dy,
                    rate_drift=cfg.credit.include_rate_drift,
                )
                rows.append(
                    table.reset_index().assign(
                        baseline=baseline, policy=policy, sector=sector, chi=chi
                    )
                )
    return pd.concat(rows, ignore_index=True)


def revelation_now(paths: dict[str, ScenarioPath]) -> float:
    """Decimal date at which "priced today" applies: the first projected month (after the
    jump-off), the month whose risk-free repricing :func:`repricing` returns by default."""
    months = next(iter(paths.values())).months
    return float(decimal_year(months[[0]])[0])


def revelation_months(months: pd.PeriodIndex, last_year: int) -> list[int]:
    """Indices of the revelation dates: the first projected month, then every January to
    ``last_year``."""
    januaries = [
        int(np.flatnonzero(months == pd.Period(f"{year}-01", "M"))[0])
        for year in range(months[0].year + 1, last_year + 1)
    ]
    return [0, *januaries]


def _delayed(
    cfg: Config,
    frame: pd.DataFrame,
    credits: dict[str, climate.SectorCredit],
    paths: dict[str, ScenarioPath],
    t0: float,
) -> pd.DataFrame:
    """Climate spread when the transition, under way since t0, is priced only at date v."""
    lgd = 1.0 - cfg.credit.recovery
    maturities = np.asarray(DELAY_MATURITIES)
    months = next(iter(paths.values())).months
    rows = []
    for baseline, policy in cfg.scenarios.transitions:
        exposures = transition.sector_exposures(frame, baseline, t0)
        for month in revelation_months(months, DELAY_LAST_YEAR):
            v = float(decimal_year(months[[month]])[0])
            dy = _rate_shift(paths, baseline, policy, maturities, month)
            for sector, credit in credits.items():
                xi = transition.asset_shock(
                    frame,
                    baseline,
                    policy,
                    exposures[sector],
                    v + maturities,
                    cfg.credit.chi,
                    t0,
                    cfg.scenarios.convergence_year,
                )
                table = climate.climate_spreads(
                    credit, xi, lgd, maturities, dy, rate_drift=cfg.credit.include_rate_drift
                )
                for maturity, row in table.iterrows():
                    rows.append(
                        {
                            "baseline": baseline,
                            "policy": policy,
                            "sector": sector,
                            "revelation_month": str(months[month]),
                            "revelation_year": v,
                            "maturity": maturity,
                            "xi": row["xi"],
                            "riskfree_shift_bp": row["riskfree_shift_bp"],
                            "climate_spread_bp": row["climate_spread_bp"],
                        }
                    )
    return pd.DataFrame(rows)


def _bond_values(cfg: Config, spreads: pd.DataFrame) -> pd.DataFrame:
    central = spreads[np.isclose(spreads["chi"], cfg.credit.chi)].set_index(
        ["baseline", "policy", "sector", "tenor"]
    )
    rows = []
    for baseline, policy in cfg.scenarios.transitions:
        for sector in transition.SECTORS:
            for maturity in BOND_MATURITIES:
                row = central.loc[(baseline, policy, sector, maturity)]
                d_rf = row["riskfree_shift_bp"] / 1e4
                impact = climate.bond_value_impact(
                    0.0,
                    float(d_rf),
                    row["spread_baseline_bp"] / 1e4,
                    row["spread_policy_bp"] / 1e4,
                    maturity,
                )
                rows.append(
                    {
                        "baseline": baseline,
                        "policy": policy,
                        "sector": sector,
                        "maturity": maturity,
                        "riskfree_shift_bp": row["riskfree_shift_bp"],
                        "climate_spread_bp": row["climate_spread_bp"],
                        "rate_effect_bp": row["rate_effect_bp"],
                        **impact,
                    }
                )
    return pd.DataFrame(rows)


def sensitivity(
    cfg: Config,
    frame: pd.DataFrame,
    credits: dict[str, climate.SectorCredit],
    rates: RatesResults,
    maturity: float = 10.0,
) -> pd.DataFrame:
    """One-at-a-time sensitivity of the headline climate spread and repricing (10y bonds)."""
    baseline, policy = cfg.scenarios.headline
    t0 = rates.jump_off.t0
    months = next(iter(rates.paths.values())).months
    lgd = 1.0 - cfg.credit.recovery
    mats = np.array([maturity])
    variants: list[tuple[str, Config, float]] = [("central", cfg, 1.0)]
    variants += [
        (f"chi = {c:g}", replace(cfg, credit=replace(cfg.credit, chi=c)), 1.0)
        for c in cfg.credit.chi_sensitivity
        if not np.isclose(c, cfg.credit.chi)
    ]
    variants += [(f"sigma_A x {m:g}", cfg, m) for m in (0.75, 1.25)]
    variants += [
        (
            f"convergence {y}",
            replace(cfg, scenarios=replace(cfg.scenarios, convergence_year=y)),
            1.0,
        )
        for y in (2040, 2060)
    ]
    no_catch_up = replace(cfg, scenarios=replace(cfg.scenarios, convergence_year=None))
    variants += [
        ("no catch-up (credit shock)", no_catch_up, 1.0),
        ("kappa = 0 (look-through)", replace(cfg, endpoint=replace(cfg.endpoint, kappa=0.0)), 1.0),
        (
            "no rate drift in PDs",
            replace(cfg, credit=replace(cfg.credit, include_rate_drift=False)),
            1.0,
        ),
    ]
    rows = []
    for label, alt, sigma_scale in variants:
        paths = {name: rates.paths[name] for name in (baseline, policy)}
        rates_differ = alt.endpoint != cfg.endpoint or (
            alt.scenarios.convergence_year not in (None, cfg.scenarios.convergence_year)
        )
        if rates_differ:
            paths = scenario_paths(alt, frame, rates.jump_off, months, (baseline, policy))
        dy = _rate_shift(paths, baseline, policy, mats)
        exposures = transition.sector_exposures(frame, baseline, t0)
        for sector in transition.SECTORS:
            credit = credits[sector]
            scaled = replace(
                credit, merton_fit=replace(credit.merton_fit, sigma=credit.sigma * sigma_scale)
            )
            xi = transition.asset_shock(
                frame,
                baseline,
                policy,
                exposures[sector],
                revelation_now(rates.paths) + mats,
                alt.credit.chi,
                t0,
                alt.scenarios.convergence_year,
            )
            row = climate.climate_spreads(
                scaled, xi, lgd, mats, dy, rate_drift=alt.credit.include_rate_drift
            ).iloc[0]
            value = climate.bond_value_impact(
                0.0,
                float(dy[0]),
                row["spread_baseline_bp"] / 1e4,
                row["spread_policy_bp"] / 1e4,
                maturity,
            )["total"]
            rows.append(
                {
                    "variant": label,
                    "sector": sector,
                    "riskfree_shift_bp": 1e4 * float(dy[0]),
                    "climate_spread_bp": row["climate_spread_bp"],
                    "bond_value_change": value,
                }
            )
    return pd.DataFrame(rows)


def run(cfg: Config, frame: pd.DataFrame, rates: RatesResults) -> CreditResults:
    inputs = _market_inputs(cfg)
    curve = _window_curve(cfg, rates)
    credits = {
        sector: climate.calibrate_sector(
            sector,
            inputs.reference.loc[sector].to_numpy(),
            cds.TENOR_YEARS,
            curve,
            curve.zero_rates(cds.TENOR_YEARS),
            cfg.credit.recovery,
        )
        for sector in transition.SECTORS
    }
    t0 = rates.jump_off.t0
    spreads = _spreads(cfg, frame, credits, rates.paths, t0)
    return CreditResults(
        mode=inputs.mode,
        sectors=inputs.sectors,
        reference=inputs.reference,
        issuer_quotes=inputs.issuer_quotes,
        weekly_changes=inputs.weekly_changes,
        sector_credit=credits,
        spreads=spreads,
        delayed=_delayed(cfg, frame, credits, rates.paths, t0),
        bond_values=_bond_values(cfg, spreads),
        sensitivity=sensitivity(cfg, frame, credits, rates),
        discount=curve,
    )


def tables(res: CreditResults) -> dict[str, pd.DataFrame]:
    """Sector-level outputs only (the licensed issuer data are never written)."""
    eligible = (
        res.sectors.reindex(res.issuer_quotes.index)
        if res.issuer_quotes is not None
        else pd.Series(dtype=str)
    )
    calibration = pd.DataFrame(
        {
            sector: {
                **{f"cds_{px}_bp": q for px, q in zip(cds.PX, credit.quotes_bp, strict=True)},
                **{
                    f"pd_{int(t)}y": q
                    for t, q in zip(credit.tenors, credit.baseline_pd(credit.tenors), strict=True)
                },
                "merton_sigma": credit.sigma,
                "merton_a": credit.merton_fit.a,
                "merton_leverage": credit.merton_fit.leverage,
                "merton_rmse_probit": credit.merton_fit.rmse_probit,
                "issuers": int((eligible == sector).sum()),
            }
            for sector, credit in res.sector_credit.items()
        }
    ).T
    calibration.index.name = "sector"
    return {
        "credit_calibration": calibration.assign(mode=res.mode),
        "climate_spreads": res.spreads,
        "delayed_revelation": res.delayed,
        "bond_value_impact": res.bond_values,
        "credit_sensitivity": res.sensitivity,
    }
