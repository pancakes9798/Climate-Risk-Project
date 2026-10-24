"""End-to-end pipeline: term structure → climate spreads → portfolio risk → tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from climate_risk.config import Config
from climate_risk.data import scenarios
from climate_risk.pipeline import credit, portfolio, rates

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Results:
    config: Config
    rates: rates.RatesResults
    credit: credit.CreditResults
    portfolio: portfolio.PortfolioResults
    tables: dict[str, pd.DataFrame]


def run_all(cfg: Config, *, with_backtest: bool = True) -> Results:
    frame = scenarios.load_scenarios(cfg.paths.scenarios_csv)
    LOGGER.info("stage 1/3: term structure")
    rates_res = rates.run(cfg, frame, with_backtest=with_backtest)
    LOGGER.info("stage 2/3: climate spreads")
    credit_res = credit.run(cfg, frame, rates_res)
    LOGGER.info("stage 3/3: portfolio risk")
    portfolio_res = portfolio.run(cfg, frame, credit_res, rates_res)
    tables = {
        **rates.tables(cfg, frame, rates_res),
        **credit.tables(credit_res),
        **portfolio.tables(portfolio_res),
    }
    return Results(
        config=cfg, rates=rates_res, credit=credit_res, portfolio=portfolio_res, tables=tables
    )


def write_tables(tables: dict[str, pd.DataFrame], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path, float_format="%.6g", index=not isinstance(table.index, pd.RangeIndex))
        written.append(path)
    LOGGER.info("wrote %d tables to %s", len(written), directory)
    return written
