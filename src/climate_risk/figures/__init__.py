"""Publication figures, rendered in a light and a dark variant (README uses ``<picture>``)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib.figure import Figure

from climate_risk.figures import credit_figures, rates_figures
from climate_risk.figures.style import THEMES, Theme, save, themed

if TYPE_CHECKING:
    from climate_risk.pipeline import Results

LOGGER = logging.getLogger(__name__)
FigureFactory = Callable[["Results", Theme], Figure]


def catalogue(res: Results) -> dict[str, FigureFactory]:
    """Figure name → factory, in reading order."""
    scen = res.config.scenarios
    baseline, headline_policy = scen.headline
    policies = tuple(p for p in scen.policy_family if p != baseline)
    return {
        "01_lambda_profile": rates_figures.lambda_profile,
        "02_endpoint_history": rates_figures.endpoint_history,
        "03_backtest": rates_figures.backtest,
        "04_fan_10y": lambda r, t: rates_figures.fan_10y(r, t, scen.reference),
        "05_narratives_10y": lambda r, t: rates_figures.narratives(r, t, scen.narratives),
        "06_policy_rates": lambda r, t: rates_figures.policy_channels(r, t, baseline, policies),
        "07_credit_reference": credit_figures.credit_reference,
        "08_climate_spreads": lambda r, t: credit_figures.climate_spreads(r, t, baseline),
        "09_iam_uncertainty": credit_figures.iam_uncertainty,
        "10_delayed_revelation": lambda r, t: credit_figures.delayed_revelation(
            r, t, headline_policy
        ),
        "11_bond_value": lambda r, t: credit_figures.bond_value(r, t, headline_policy),
        "12_portfolio": credit_figures.portfolio,
    }


def render_all(res: Results, directory: Path) -> list[Path]:
    written = []
    for name, factory in catalogue(res).items():
        for theme in THEMES:
            with themed(theme):
                written.append(save(factory(res, theme), directory, name, theme))
    LOGGER.info("wrote %d figures to %s", len(written), directory)
    return written
