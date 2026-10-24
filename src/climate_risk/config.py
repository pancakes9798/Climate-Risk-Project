"""Immutable configuration of the pipeline.

Every number that drives a result lives here (or is estimated from data), so that a reader can
audit the calibration in one place. Values are documented with their source; sensitivity
analyses vary them through :func:`dataclasses.replace`. Each section validates itself on
construction (also after ``replace``), so an inconsistent set-up fails at once with a message
instead of deep inside the pipeline.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path

CDS_PATH_ENV = "CLIMATE_RISK_CDS_PATH"
SCENARIO_LAST_YEAR = 2100  # last year of the SSP quantifications, end of every projection
SUMMARY_MATURITY = 10.0  # maturity (years) of the yield reported in the summary tables
MIN_BACKTEST_MONTHS = 60  # estimation sample before the first backtest origin


def _source_root() -> Path:
    """Repository root when running from a source checkout, else the working directory."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd()


REPO_ROOT = _source_root()


def _cds_override() -> Path | None:
    value = os.environ.get(CDS_PATH_ENV)
    return Path(value).expanduser() if value else None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _increasing(values: tuple[float, ...]) -> bool:
    return all(b > a for a, b in itertools.pairwise(values))


@dataclass(frozen=True)
class Paths:
    """Filesystem layout, rooted at the repository by default.

    The location of the licensed CDS panel is read from ``$CLIMATE_RISK_CDS_PATH`` once, when
    the object is created, so a configuration does not change with the environment afterwards.
    """

    root: Path = field(default_factory=lambda: REPO_ROOT)
    cds_override: Path | None = field(default_factory=_cds_override)

    @property
    def raw(self) -> Path:
        """Downloaded source files (git-ignored, reproducible with ``climate-risk download``)."""
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        """Small, versioned extracts of the public data actually used by the models."""
        return self.root / "data" / "processed"

    @property
    def scenarios_csv(self) -> Path:
        return self.root / "data" / "scenarios" / "owid_ipcc_scenarios.csv"

    @property
    def sectors_csv(self) -> Path:
        return self.root / "data" / "sectors.csv"

    @property
    def cds_csv(self) -> Path:
        """Licensed CDS panel; never committed. Override with ``$CLIMATE_RISK_CDS_PATH``."""
        return self.cds_override or self.root / "data" / "private" / "cds.csv"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def tables(self) -> Path:
        return self.results / "tables"


@dataclass(frozen=True)
class TermStructureConfig:
    """Dynamic Nelson–Siegel set-up (Diebold & Li, 2006) on GSW zero-coupon yields."""

    maturities: tuple[float, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30)
    output_maturities: tuple[float, ...] = (1, 2, 3, 5, 7, 10, 20, 30)
    sample_start: str = "1986-01"  # first month with the full 1–30y GSW curve
    as_of: str = "2026-09"  # jump-off month of all projections
    # Decay λ (per year). The pooled-SSE profile over 1–30y GSW maturities is bimodal and flat
    # (8.4bp at λ≈0.12 vs 8.7bp at λ≈0.45): the global mode puts the curvature hump at 15y and
    # destroys the level/slope interpretation. The search is therefore restricted to curvature
    # peaks between 1.2y and 7.2y, i.e. a medium-term third factor as in Diebold & Li (2006).
    lambda_grid: tuple[float, float, int] = (0.25, 1.50, 251)
    endpoint_lag_quarters: int = 1  # use r* of the previous quarter (no look-ahead)

    def __post_init__(self) -> None:
        _require(
            self.maturities[0] > 0 and _increasing(self.maturities),
            "maturities must be positive and strictly increasing",
        )
        _require(all(m > 0 for m in self.output_maturities), "output maturities must be > 0")
        _require(
            SUMMARY_MATURITY in self.output_maturities,
            f"output_maturities must include the {SUMMARY_MATURITY:g}-year yield of the summary",
        )
        low, high, steps = self.lambda_grid
        _require(0 < low < high and steps >= 2, "lambda_grid must be (low, high, steps ≥ 2)")
        _require(self.sample_start < self.as_of, "sample_start must precede as_of")
        _require(self.endpoint_lag_quarters >= 0, "endpoint_lag_quarters must be ≥ 0")


@dataclass(frozen=True)
class ScenarioConfig:
    """SSP pathways (Riahi et al., 2017) used throughout."""

    reference: str = "SSP2 - Baseline"
    narratives: tuple[str, ...] = (
        "SSP1 - Baseline",
        "SSP2 - Baseline",
        "SSP3 - Baseline",
        "SSP4 - Baseline",
        "SSP5 - Baseline",
    )
    policy_family: tuple[str, ...] = ("SSP2 - Baseline", "SSP2 - 4.5", "SSP2 - 2.6", "SSP2 - 1.9")
    # (baseline, policy) pairs for the credit analysis: same SSP narrative, hence same IAM.
    transitions: tuple[tuple[str, str], ...] = (
        ("SSP1 - Baseline", "SSP1 - 2.6"),
        ("SSP1 - Baseline", "SSP1 - 1.9"),
        ("SSP2 - Baseline", "SSP2 - 4.5"),
        ("SSP2 - Baseline", "SSP2 - 2.6"),
        ("SSP2 - Baseline", "SSP2 - 1.9"),
        ("SSP5 - Baseline", "SSP5 - 2.6"),
        ("SSP5 - Baseline", "SSP5 - 1.9"),
    )
    # Headline transition reported in the summary figures (stringent policy, middle-of-the-road
    # narrative, MESSAGE-GLOBIOM marker model).
    headline: tuple[str, str] = ("SSP2 - Baseline", "SSP2 - 1.9")
    # Harmonised paths rejoin the native IAM path by this year (None: credit shocks never
    # catch up; used only in the credit sensitivity, the rates stage needs a year).
    convergence_year: int | None = 2050

    def __post_init__(self) -> None:
        _require(
            self.reference in self.narratives,
            f"reference pathway {self.reference!r} is not among the narratives",
        )
        _require(
            self.headline in self.transitions,
            f"headline transition {self.headline} is not among the transitions",
        )
        _require(
            self.policy_family[0] == self.headline[0],
            "policy_family must start with the baseline of the headline transition",
        )
        _require(
            self.convergence_year is None or self.convergence_year <= SCENARIO_LAST_YEAR,
            f"convergence_year must not be after {SCENARIO_LAST_YEAR}",
        )


@dataclass(frozen=True)
class EndpointConfig:
    """Mapping from scenario macro variables to the long-run anchor of the yield curve.

    * ``theta`` – elasticity of r* to trend growth; HLW (2023) estimate for the US, c = 1.113.
    * ``psi`` – pass-through of the carbon-cost share of GDP to the price level.
    * ``kappa`` – short-rate response to carbon-induced inflation (Taylor 1993: 1.5;
      0 = the central bank looks through it).
    """

    theta: float = 1.113
    psi: float = 1.0
    kappa: float = 1.5
    # Global carbon-pricing revenue ≈ USD 100bn in 2023-24, i.e. ≈ 0.1% of world GDP
    # (World Bank, State and Trends of Carbon Pricing 2024): the observed jump-off value to
    # which the scenario carbon-cost shares are harmonised.
    carbon_cost_share_t0: float = 0.001

    def __post_init__(self) -> None:
        _require(self.theta >= 0 and self.psi >= 0 and self.kappa >= 0, "θ, ψ, κ must be ≥ 0")
        _require(self.carbon_cost_share_t0 >= 0, "carbon_cost_share_t0 must be ≥ 0")


@dataclass(frozen=True)
class SimulationConfig:
    n_paths: int = 2000
    seed: int = 20_251_024
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

    def __post_init__(self) -> None:
        _require(self.n_paths >= 2, "n_paths must be at least 2")
        _require(
            self.quantiles[0] > 0 and self.quantiles[-1] < 1 and _increasing(self.quantiles),
            "quantiles must be strictly increasing and inside (0, 1)",
        )
        _require(0.5 in self.quantiles, "quantiles must include the median (0.5)")


@dataclass(frozen=True)
class BacktestConfig:
    first_origin: str = "2000-01"
    horizons: tuple[int, ...] = (1, 6, 12, 24, 60)
    maturities: tuple[float, ...] = (1, 2, 5, 10, 30)

    def __post_init__(self) -> None:
        _require(all(h >= 1 for h in self.horizons), "backtest horizons must be ≥ 1 month")
        _require(all(m > 0 for m in self.maturities), "backtest maturities must be > 0")


@dataclass(frozen=True)
class CreditConfig:
    """CDS calibration and Battiston–Monasterolo transition shock."""

    recovery: float = 0.40  # ISDA standard for senior unsecured CDS
    window: tuple[str, str] = ("2015-01", "2019-12")  # pre-COVID calibration window
    min_coverage: float = 0.90  # issuers quoted in fewer months of the window are dropped
    max_stale_share: float = 0.25  # issuers with unchanged 5y quotes in >25% of weeks are dropped
    # Fixed-face-value Merton (as in the σ_A calibration): the risk-free repricing at revelation
    # also moves the risk-neutral drift, hence the PDs.
    include_rate_drift: bool = True
    chi: float = 0.5  # elasticity of asset value to sector output shock (BM 2020 use 1.0)
    chi_sensitivity: tuple[float, ...] = (0.2, 0.5, 1.0)

    def __post_init__(self) -> None:
        _require(0 <= self.recovery < 1, "recovery must be in [0, 1)")
        _require(self.window[0] <= self.window[1], "calibration window must be (start, end)")
        _require(0 < self.min_coverage <= 1, "min_coverage must be in (0, 1]")
        _require(0 <= self.max_stale_share <= 1, "max_stale_share must be in [0, 1]")
        _require(
            self.chi > 0 and any(abs(c - self.chi) < 1e-12 for c in self.chi_sensitivity),
            f"chi = {self.chi} must be positive and one of chi_sensitivity {self.chi_sensitivity}",
        )


@dataclass(frozen=True)
class PortfolioConfig:
    horizon: int = 5
    n_sims: int = 200_000
    alpha: float = 0.99
    seed: int = 7
    # Risk-neutral / actual default-intensity ratio used to obtain physical PDs for the loss
    # distribution: Driessen (2005) estimates 2.4 for BBB-rated firms (1.8 AA, 2.6 A).
    risk_premium_ratio: float = 2.4

    def __post_init__(self) -> None:
        _require(self.horizon >= 1, "portfolio horizon must be at least one year")
        _require(self.n_sims >= 100, "n_sims must be at least 100")
        _require(0.5 < self.alpha < 1, "alpha must be in (0.5, 1)")
        _require(
            self.risk_premium_ratio >= 1,
            "risk_premium_ratio must be ≥ 1: risk-neutral default intensities exceed physical ones",
        )


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    term_structure: TermStructureConfig = field(default_factory=TermStructureConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    credit: CreditConfig = field(default_factory=CreditConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    def __post_init__(self) -> None:
        ts, bt = self.term_structure, self.backtest
        jump_off_year = int(ts.as_of[:4])
        _require(
            jump_off_year < SCENARIO_LAST_YEAR,
            f"the jump-off month {ts.as_of} must precede {SCENARIO_LAST_YEAR}",
        )
        convergence = self.scenarios.convergence_year
        _require(
            convergence is None or convergence > jump_off_year,
            f"convergence_year {convergence} must be after the jump-off year {jump_off_year}",
        )
        _require(
            bt.first_origin < ts.as_of, "the first backtest origin must precede the jump-off month"
        )
        start, first = (int(m[:4]) * 12 + int(m[5:7]) for m in (ts.sample_start, bt.first_origin))
        _require(
            first - start >= MIN_BACKTEST_MONTHS,
            f"the first backtest origin needs {MIN_BACKTEST_MONTHS} months of data before it",
        )
        _require(
            set(bt.maturities) <= set(ts.maturities),
            "backtest maturities must be among the estimation maturities",
        )
