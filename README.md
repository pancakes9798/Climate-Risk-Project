# Credit Spreads, Yield Curve and Climate Risk

This repository explores the interaction between credit spreads, the term structure of interest rates, and climate-related risk, with a focus on transition risk under different climate scenarios.

The core idea is that climate change — and, more importantly, climate policy — affects credit risk not only through firm-specific fundamentals, but also through the shape and dynamics of the yield curve. Ignoring this channel may lead to a systematic underestimation of long-horizon default risk.

---

## Motivation

Traditional credit risk models typically treat the yield curve as an exogenous macro-financial input and climate risk as an add-on stress scenario.  
This project instead studies how climate transition pathways can alter the joint dynamics of:

- the risk-free term structure,
- credit spreads,
- firm-level default probabilities,

with particular attention to **carbon-intensive sectors**.

The objective of the project is forecasting untill 2100 the yield curve and the credit spread curve integrating the climate risk.

---

## Methodology (High Level)

The project combines:

- term structure modeling for the yield curve,
- credit spread decomposition into macro and climate-related components,
- scenario-based shocks derived from Shared Socioeconomic Pathways (SSP),
- firm-level exposures calibrated using sectoral and balance-sheet information.

Climate risk enters the model primarily through transition-related channels (e.g. carbon pricing, policy intensity, demand destruction), rather than purely physical risk.

---

## Repository Structure

```text
.
├── data/              # Input data (where applicable)
├── src/               # Yield curve and credit spread code
└── README.md
