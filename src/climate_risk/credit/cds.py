"""Licensed CDS panel: loading, cleaning and the reference curves used for calibration.

The panel has one row per (date, issuer) with par spreads in basis points for the 1y–10y tenors
(``PX1`` … ``PX10``). It is never redistributed with the repository; results are reported at the
sector level.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

PX = tuple(f"PX{k}" for k in range(1, 11))
TENOR_YEARS = np.arange(1.0, 11.0)
TENOR_YEARS.setflags(write=False)  # shared module constant
LOGGER = logging.getLogger(__name__)


def load_cds(path: Path, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Read and clean the CDS panel; rows with missing or negative quotes are dropped."""
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in ("Date", "Ticker", "Company", *PX) if c not in header]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    frame = pd.read_csv(path, usecols=["Date", "Ticker", "Company", *PX])
    if tickers is not None:
        frame = frame[frame["Ticker"].isin(list(tickers))]
    frame = frame.rename(columns={"Date": "date", "Ticker": "ticker", "Company": "company"})
    frame["date"] = pd.to_datetime(frame["date"])
    quotes = frame[list(PX)].apply(pd.to_numeric, errors="coerce")
    valid = quotes.notna().all(axis=1) & (quotes >= 0).all(axis=1)
    out = frame.loc[valid, ["date", "ticker", "company"]].join(quotes[valid])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def monthly_panel(cds: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Last quote of each issuer in each month of [start, end], indexed by (month, ticker)."""
    month = cds["date"].dt.to_period("M")
    window = cds[(month >= pd.Period(start, "M")) & (month <= pd.Period(end, "M"))]
    window = window.assign(month=window["date"].dt.to_period("M"))
    last = window.groupby(["month", "ticker"], sort=True)[list(PX)].last()
    return last


def reference_spreads(panel: pd.DataFrame, min_coverage: float = 0.9) -> pd.DataFrame:
    """Time-median spread curve of every issuer quoted in at least ``min_coverage`` of months."""
    months = panel.index.get_level_values("month").unique()
    counts = panel.groupby(level="ticker").size()
    eligible = counts[counts >= min_coverage * len(months)].index
    return (
        panel[panel.index.get_level_values("ticker").isin(eligible)]
        .groupby(level="ticker")
        .median()
    )


def sector_reference(issuer_spreads: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Cross-issuer median curve per sector (issuers missing from ``sectors`` are ignored)."""
    labelled = issuer_spreads.join(sectors.rename("sector"), how="inner")
    return labelled.groupby("sector")[list(PX)].median()


def weekly_log_changes(
    cds: pd.DataFrame, column: str, start: str, end: str, tickers: Iterable[str]
) -> pd.DataFrame:
    """Wednesday-to-Wednesday log changes of one tenor, weeks × issuers (complete weeks only)."""
    names = list(tickers)
    wide = cds[cds["ticker"].isin(names)].pivot_table(index="date", columns="ticker", values=column)
    month = wide.index.to_period("M")
    wide = wide[(month >= pd.Period(start, "M")) & (month <= pd.Period(end, "M"))]
    weekly = wide.resample("W-WED").last()
    changes = np.log(weekly.where(weekly > 0)).diff().iloc[1:]
    selected = changes[[n for n in names if n in changes.columns]]
    complete = selected.dropna()
    if len(complete) < len(selected):
        LOGGER.warning(
            "%d of %d weeks dropped: some issuer has no quote in them",
            len(selected) - len(complete),
            len(selected),
        )
    return complete


def stale_issuers(changes: pd.DataFrame, max_zero_share: float = 0.25) -> list[str]:
    """Issuers whose quote is unchanged in more than ``max_zero_share`` of the weeks.

    Persistently unchanged CDS quotes are composite/stale marks rather than traded prices;
    they would bias both the reference curves and the dependence estimates.
    """
    share = (changes == 0).mean()
    return sorted(share[share > max_zero_share].index)


def load_sector_map(path: Path) -> pd.Series:
    """Ticker → sector assignment (curated from public company information)."""
    table = pd.read_csv(path)
    return table.set_index("ticker")["sector"]
