# Changelog

## 2.0.0 — complete rebuild (October 2026)

Version 2 replaces the capstone notebooks with a tested Python package. It re-specifies every
model that, on review, could not support its conclusions. The original version is preserved
under the git tag [`v0.1-capstone`](https://github.com/pancakes9798/Climate-Risk-Project/tree/v0.1-capstone).

### Why: audit of v0.1

**Yield curve**

| Issue in v0.1 | Consequence | Fix in v2 |
|---|---|---|
| Nelson–Siegel factors estimated on *daily* data, called "monthly"; the join with month-end macro data dropped ~30% of months | AR(2) lags of irregular length | month-end curves, complete panel |
| AR(2) estimated at (gappy) monthly frequency, applied with 10-year steps | a decade treated as one month; persistence wrong by orders of magnitude | monthly VAR, monthly simulation |
| Projection started from Dec-2024 factors but labelled 2010 | "forecasts" overlapping history | jump-off at the last observed month |
| λ free per day in estimation, fixed at 1.0 in projection | betas not comparable, curve rebuilt with another model | one λ, profile-estimated |
| "Historical" climate regressors were SSP2-4.5 projections, linearly interpolated between 2005, 2010 and 2020 | 7 regressors carrying 2–3 pieces of information; coefficients not identified | structural, calibrated scenario channel (growth/r\*, carbon pricing) |
| Coefficients estimated on monthly raw changes, applied to standardised decadal changes; z-scored *within* each scenario | effects scaled by 10–100×; SSP1 vs SSP5 differences erased | harmonised, unit-consistent scenario paths |
| Deterministic single path | no uncertainty to 2100 | Monte Carlo fan charts |
| No validation | — | out-of-sample backtest vs random walk, Diebold–Mariano tests |

**Credit**

| Issue in v0.1 | Consequence | Fix in v2 |
|---|---|---|
| Sign error in the conditional PD: Φ((a − β′Z)/·) with β > 0 | higher carbon price and temperature *lowered* the PD | structural Merton threshold shift; sign by construction |
| "1-year" PD computed from the 10-year CDS (`PX10`); TTC average including the PG&E default of 2019 | PD of 3.6% vs 0.13% from the 1y CDS on 2015–2017 | full CDS bootstrap, calibration window 2015–2019, sector medians |
| Risk-neutral PD used in a Vasicek (physical) model | measures mixed | Q for pricing, P (Driessen ratio) for loss distributions |
| A 1-year spread added to the 10-year yield | maturity mismatch | maturity-matched zero-coupon spreads |
| Hard-coded, uncalibrated sensitivities [0.25, 0.30, 0.10]; correlations of deterministic trends | arbitrary results | IAM-based Battiston–Monasterolo shock; dependence from CDS co-movements |
| Copulas fitted on 91 interpolated trend points; t-copula with Pearson ρ and ν = 8 | non-i.i.d. data, inconsistent estimator | Kendall τ → ρ = sin(πτ/2), ν by pseudo-ML, on weekly CDS changes |

**Engineering**

| Issue in v0.1 | Fix in v2 |
|---|---|
| Absolute paths to one machine; `main.py` printed "Hello" | package with CLI: `climate-risk download / run / info` |
| Missing (`copulas`, `yfinance`) and wrong (`datetime`) dependencies; Python 3.9 (end of life); `pandas-datareader` broken on Python ≥3.12 | `uv` project on Python ≥3.11; minimal dependencies |
| Notebooks executed out of order with 1.5 MB of outputs; scenario switched by editing cells | deterministic pipeline; every scenario in one run |
| `simulate_betas_AR2` silently ignored two arguments; global seed set at import | explicit, seeded random generators |
| No tests, no CI | 177 offline tests (98% coverage), ruff, mypy, GitHub Actions on Python 3.11–3.13 |
| Third-party PDFs committed to a public repository | references in `references/references.bib` |

### Added

- Shifting-endpoint dynamic Nelson–Siegel model on Gürkaynak–Sack–Wright zero-coupon yields,
  anchored to the Holston–Laubach–Williams natural rate and Cleveland Fed inflation
  expectations.
- Scenario channels with harmonisation to the 2026 jump-off (additive for growth rates,
  multiplicative for the carbon bill and sector output, one convergence year):
  - r\* responding to growth differentials across SSPs (θ = 1.113, HLW);
  - a carbon-price inflation channel on short rates (κ = 1.5).
- An expectations-hypothesis scenario overlay shared by the yield projections and the bond
  repricing, so the two are consistent by construction.
- CDS hazard bootstrap (ISDA-style legs), Merton asset volatility, Battiston–Monasterolo
  climate spreads by sector and IAM.
- Delayed-revelation ("cost of waiting") analysis and an exact rate/credit decomposition of
  bond repricing.
- Portfolio loss distributions: t, Gaussian and Basel-IRB copulas; VaR and ES under physical
  and risk-neutral measures; scenario-probability mixtures.
- One-at-a-time sensitivity tables for every calibrated assumption (θ, κ, χ, σ_A, the
  convergence year, the rate drift in default probabilities).
- Fail-fast validation of the configuration and of the input data (scenario power mix,
  truncated downloads, gaps in monthly series, non-stationary dynamics, CDS coverage).
- Auto-generated results summary, light/dark figures, methodology document.
