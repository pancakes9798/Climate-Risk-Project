# Data

| Path | Content | Versioned |
|---|---|---|
| `processed/gsw_zero_curves_monthly.csv` | Month-end zero-coupon Treasury yields, 1–30y, % (continuously compounded) | yes |
| `processed/endpoint_drivers_monthly.csv` | Monthly r\* (HLW, lagged one quarter), 10y expected inflation and i\* = r\* + π\* | yes |
| `processed/hlw_us_quarterly.csv` | HLW one-sided US estimates: trend growth g, other determinants z, r\* | yes |
| `scenarios/owid_ipcc_scenarios.csv` | 26 SSP pathways, world, 2005–2100 | yes |
| `sectors.csv` | Ticker → sector map of the 67 issuers analysed | yes |
| `raw/` | Files exactly as downloaded, via `climate-risk download` | no (git-ignored) |
| `private/cds.csv` | Licensed CDS panel | **never** (git-ignored) |

The processed extracts pin the data vintage behind the published results. To refresh them:

```bash
uv run climate-risk download   # data/raw → data/processed
```

The jump-off month of the projections is set in `config.py` (`TermStructureConfig.as_of`), so
a refresh does not silently change the published numbers.

## Sources and terms

- **Treasury zero-coupon curve:** Gürkaynak, Sack & Wright (2007), Board of Governors of the
  Federal Reserve System, `feds200628.csv`. The Fed describes it as a staff research product,
  not an official statistical release.
- **Natural rate of interest:** Holston, Laubach & Williams (2023), Federal Reserve Bank of New
  York, one-sided estimates, "current estimates" workbook. The vintage used ends in 2026Q2.
- **10-year expected inflation:** Federal Reserve Bank of Cleveland (model of Haubrich,
  Pennacchi & Ritchken, 2012), via FRED series `EXPINF10YR`.
- **SSP scenarios:** IIASA SSP Database, marker quantifications (Riahi et al., 2017), as
  compiled by Our World in Data. The marker models are:

  | SSP | Marker model |
  |---|---|
  | SSP1 | IMAGE |
  | SSP2 | MESSAGE-GLOBIOM |
  | SSP3 | AIM/CGE |
  | SSP4 | GCAM |
  | SSP5 | REMIND-MAgPIE |

  Use is subject to the SSP Database terms; cite Riahi et al. (2017).
- **Moody's Baa – 10y Treasury spread:** FRED series `BAA10Y`. It is downloaded at run time
  and not redistributed. It is used only in demo mode.

## The licensed CDS panel

The credit calibration uses a daily panel of CDS par spreads, 661 issuers from 2015-01 to
2021-09, with this schema:

```
Date,Ticker,Company,PX1,PX2,PX3,PX4,PX5,PX6,PX7,PX8,PX9,PX10
2015-01-01,A,Agilent Technologies Inc,14.85,25.05,...,168.195
```

`PX1`…`PX10` are the 1- to 10-year par spreads in basis points.

The data are licensed and **not part of this repository**. Only sector-level aggregates
(medians over 19–25 issuers) and model outputs are published. To reproduce the full results,
place the file at `data/private/cds.csv` or point to it with

```bash
export CLIMATE_RISK_CDS_PATH=/path/to/cds.csv
```

**Demo mode.** Without the panel, the pipeline runs with a flat credit curve at the
2015–2019 median of the Moody's Baa spread, identical across sectors. Two consequences:

- sector differences then come only from the transition shock;
- the portfolio uses the Basel IRB correlation, since the copula cannot be estimated.

Every output records which mode produced it.

## Sector map

`sectors.csv` classifies issuers by their main business, following the Climate Policy Relevant
Sectors of Battiston et al. (2017). The 2015–2019 calibration keeps 65 issuers after the
coverage and stale-quote filters (19 / 25 / 21 by group).

- **Oil & gas:** 20 integrated and upstream producers from OECD countries. State-owned and
  sanctioned issuers are excluded, because sovereign risk dominates their spreads.
- **Utilities:** 26 generation-owning utilities. Pure networks, such as National Grid and Red
  Eléctrica, are excluded because their output is not driven by the generation mix. PG&E is
  excluded because of its 2019 bankruptcy.
- **Low-carbon control:** 21 investment-grade technology, pharmaceutical and telecom issuers
  with no direct exposure to the IAM sectors.
