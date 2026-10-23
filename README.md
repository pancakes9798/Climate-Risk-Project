# Credit Spreads, Yield Curve and Climate Risk

Long-horizon projections of the **risk-free yield curve** and the **credit-spread curve** under **climate transition scenarios (SSP)**, with a focus on carbon-intensive sectors.

> Developed as the capstone project of the Executive Master in *Quantitative Methods for Climate Risk* (École Polytechnique).

---

## Motivation

Traditional credit-risk models treat the yield curve as an exogenous macro-financial input and climate risk as an add-on stress scenario.
This project studies how climate transition pathways can change the **joint dynamics** of:

- the risk-free term structure,
- credit spreads,
- firm-level default probabilities.

The core idea: climate policy affects credit risk not only through firm fundamentals, but also through the shape and dynamics of the yield curve. Ignoring this channel can systematically understate long-horizon default risk.

**Goal:** forecast the yield curve and the credit-spread curve up to 2100, integrating climate transition risk.

## Methodology

- **Term-structure model** for the risk-free curve
- **Credit-spread decomposition** into macro and climate-related components
- **Scenario-based shocks** from the Shared Socioeconomic Pathways (SSP)
- **Firm-level exposures** calibrated on sectoral and balance-sheet information

Climate risk enters mainly through transition channels (carbon pricing, policy intensity, demand destruction) rather than physical risk.

## Results

<!-- TODO: add 1–2 key charts (e.g. projected curves by SSP scenario) and a short takeaway -->

## Repository structure

```
.
├── data/         # input data (where redistributable)
├── src/          # yield-curve and credit-spread code
├── Papers/       # reference papers
├── references/   # bibliography
└── main.py       # entry point
```

## Getting started

```bash
git clone https://github.com/pancakes9798/Climate-Risk-Project.git
cd Climate-Risk-Project
uv sync
uv run python main.py
```

## Author

Emanuele Migliaccio — PhD candidate in Quantitative Finance, Sapienza University of Rome
