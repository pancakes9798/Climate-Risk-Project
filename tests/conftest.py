"""Synthetic, licence-free data root for end-to-end tests (no network, no proprietary data)."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_risk.config import (
    REPO_ROOT,
    BacktestConfig,
    Config,
    Paths,
    PortfolioConfig,
    SimulationConfig,
)
from climate_risk.credit.cds import PX
from climate_risk.data import store
from climate_risk.term_structure.nelson_siegel import loadings

MATURITIES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30)
SECTOR_LEVEL = {"Oil & gas": 70.0, "Utilities": 55.0, "Low-carbon control": 35.0}


def _endpoint(months: pd.PeriodIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(months)
    rstar = 2.5 - 2.0 * np.arange(n) / n + np.cumsum(rng.normal(0, 0.02, n))
    inflation = 3.0 - 0.8 * np.arange(n) / n + rng.normal(0, 0.05, n)
    frame = pd.DataFrame({"rstar": rstar, "inflation": inflation}, index=months)
    frame["istar"] = frame["rstar"] + frame["inflation"]
    frame.index.name = "month"
    return frame


def _curves(drivers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    months = pd.period_range("1986-01", "2026-09", freq="M", name="month")
    istar = drivers["istar"].reindex(months).to_numpy()
    cycle = np.zeros((len(months), 3))
    mean = np.array([0.8, -1.8, -1.5])
    cycle[0] = mean
    for t in range(1, len(months)):
        cycle[t] = (
            mean
            + np.diag([0.98, 0.95, 0.92]) @ (cycle[t - 1] - mean)
            + rng.normal(0, [0.25, 0.3, 0.6])
        )
    factors = cycle + np.column_stack([istar, np.zeros_like(istar), np.zeros_like(istar)])
    yields = factors @ loadings(np.array(MATURITIES, dtype=float), 0.47).T
    yields += rng.normal(0, 0.05, yields.shape)
    return pd.DataFrame(yields, index=months, columns=[float(m) for m in MATURITIES])


def _cds_panel(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2015-01-01", "2019-12-31")
    shape = np.linspace(0.25, 1.6, 10)
    sectors, rows = [], []
    common = np.cumsum(rng.normal(0, 0.01, len(dates)))
    for sector, level in SECTOR_LEVEL.items():
        sector_factor = np.cumsum(rng.normal(0, 0.008, len(dates)))
        for k in range(4):
            ticker = f"{sector[:3].upper()}{k}"
            sectors.append({"ticker": ticker, "company": f"{sector} issuer {k}", "sector": sector})
            idio = np.cumsum(rng.normal(0, 0.01, len(dates)))
            scale = level * (0.8 + 0.1 * k) * np.exp(common + sector_factor + idio)
            quotes = np.outer(scale, shape)
            frame = pd.DataFrame(quotes, columns=list(PX))
            frame.insert(0, "Company", f"{sector} issuer {k}")
            frame.insert(0, "Ticker", ticker)
            frame.insert(0, "Date", dates.strftime("%Y-%m-%d"))
            rows.append(frame)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(sectors)


@pytest.fixture(scope="session")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(2026)
    root = tmp_path_factory.mktemp("repo")
    paths = Paths(root=root)
    paths.processed.mkdir(parents=True)
    drivers = _endpoint(pd.period_range("1982-01", "2026-09", freq="M", name="month"), rng)
    drivers.to_csv(paths.processed / store.ENDPOINT_FILE)
    _curves(drivers, rng).to_csv(paths.processed / store.CURVES_FILE)
    paths.scenarios_csv.parent.mkdir(parents=True)
    shutil.copy(Paths(root=REPO_ROOT).scenarios_csv, paths.scenarios_csv)
    cds, sectors = _cds_panel(rng)
    sectors.to_csv(paths.sectors_csv, index=False)
    (root / "data" / "private").mkdir(parents=True)
    cds.to_csv(root / "data" / "private" / "cds.csv", index=False)
    paths.raw.mkdir(parents=True)
    days = pd.bdate_range("2014-01-01", "2020-12-31")
    baa = pd.DataFrame(
        {
            "observation_date": days.strftime("%Y-%m-%d"),
            "BAA10Y": 2.0 + 0.2 * np.sin(np.arange(len(days)) / 200),
        }
    )
    baa.to_csv(paths.raw / store.RAW_FILES["baa"][0], index=False)
    return root


def small_config(root: Path) -> Config:
    return replace(
        Config(),
        paths=Paths(root=root),
        simulation=SimulationConfig(n_paths=120, seed=1),
        backtest=BacktestConfig(first_origin="2024-01", horizons=(1, 12), maturities=(2, 10)),
        portfolio=PortfolioConfig(n_sims=6_000, seed=3),
    )


@pytest.fixture
def no_cds_override(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("CLIMATE_RISK_CDS_PATH", raising=False)
    return
