"""Readers for the public data sources (no network access here; see :mod:`.download`).

Conventions
-----------
* Monthly series are indexed by a ``PeriodIndex`` with frequency ``"M"`` named ``month``;
  quarterly series by a ``PeriodIndex`` with frequency ``"Q"`` named ``quarter``.
* Interest rates and inflation are expressed in percent per annum.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
HLW_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/economists/williams/data/"
    "Holston_Laubach_Williams_current_estimates.xlsx"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

_GSW_ZERO_COLUMN = re.compile(r"SVENY(\d{2})")


class DataFormatError(ValueError):
    """Raised when a source file does not have the expected layout."""


def _cell_text(value: object) -> str:
    """Spreadsheet cell as stripped text; empty string for missing cells."""
    return "" if pd.isna(value) else str(value).strip()


def _find_header_row(path: Path, prefix: str, max_lines: int = 50) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line.lstrip('"').startswith(prefix):
                return line_number
            if line_number >= max_lines:
                break
    raise DataFormatError(f"{path.name}: no header line starting with {prefix!r}")


def read_gsw_zero_yields(path: Path) -> pd.DataFrame:
    """Daily Gürkaynak–Sack–Wright zero-coupon yields.

    Returns a frame indexed by date whose columns are maturities in years (floats) and whose
    values are continuously compounded zero-coupon yields in percent.
    """
    header_row = _find_header_row(path, "Date,")
    raw = pd.read_csv(path, skiprows=header_row, index_col="Date", parse_dates=["Date"])
    columns = {c: float(m.group(1)) for c in raw.columns if (m := _GSW_ZERO_COLUMN.fullmatch(c))}
    if not columns:
        raise DataFormatError(f"{path.name}: no SVENYxx zero-coupon yield columns")
    yields = raw[list(columns)].apply(pd.to_numeric, errors="coerce").rename(columns=columns)
    return yields.sort_index()


def month_end_curves(daily: pd.DataFrame, maturities: Sequence[float]) -> pd.DataFrame:
    """Last trading day of each month on which the whole curve ``maturities`` is observed."""
    missing = [m for m in maturities if m not in daily.columns]
    if missing:
        raise DataFormatError(f"maturities not available in the curve data: {missing}")
    complete = daily[list(maturities)].dropna()
    last = complete.groupby(complete.index.to_period("M")).tail(1)
    out = last.set_axis(last.index.to_period("M"), axis=0)
    out.index.name = "month"
    return out


def read_fred_csv(path: Path) -> pd.Series:
    """A single FRED series downloaded from ``fredgraph.csv`` (missing values are ``.``)."""
    frame = pd.read_csv(path, na_values=["."])
    if frame.shape[1] != 2:
        raise DataFormatError(f"{path.name}: expected two columns (date, value)")
    date_col, value_col = frame.columns
    series = pd.Series(
        pd.to_numeric(frame[value_col], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(frame[date_col])),
        name=str(value_col),
    )
    return series.dropna()


def to_monthly(series: pd.Series, how: str = "last") -> pd.Series:
    """Aggregate a dated series to calendar months (``last`` observation or ``mean``)."""
    grouped = series.groupby(series.index.to_period("M"))
    out = grouped.last() if how == "last" else grouped.mean()
    out.index.name = "month"
    return out


def read_hlw_us(path: Path) -> pd.DataFrame:
    """US one-sided estimates of Holston, Laubach & Williams (2023).

    Returns a quarterly frame with columns ``g`` (trend growth), ``z`` (other determinants)
    and ``rstar`` (natural rate of interest), all in percent per annum.
    """
    sheet = pd.read_excel(path, sheet_name="HLW Estimates", header=None)
    first_col = np.array([_cell_text(x) for x in sheet.iloc[:, 0]])
    header_rows = np.flatnonzero(first_col == "Date")
    if header_rows.size == 0:
        raise DataFormatError(f"{path.name}: 'Date' header not found in 'HLW Estimates'")
    header = int(header_rows[0])
    groups = [_cell_text(x) for x in sheet.iloc[header - 1]]
    labels = {"g": "Trend Growth", "z": "Other Determinants", "rstar": "Natural Rate"}
    columns: dict[str, int] = {}
    for key, label in labels.items():
        hits = [i for i, text in enumerate(groups) if label in text]
        if not hits or _cell_text(sheet.iat[header, hits[0]]) != "US":
            raise DataFormatError(f"{path.name}: US column for {label!r} not found")
        columns[key] = hits[0]
    body = sheet.iloc[header + 1 :, [0, *columns.values()]]
    body.columns = ["date", *columns.keys()]
    body = body[pd.to_datetime(body["date"], errors="coerce").notna()]
    quarters = pd.PeriodIndex(pd.to_datetime(body["date"]), freq="Q", name="quarter")
    values = body[list(columns)].apply(pd.to_numeric, errors="coerce")
    return values.set_axis(quarters, axis=0).dropna()


def quarterly_to_monthly(quarterly: pd.DataFrame, lag_quarters: int = 0) -> pd.DataFrame:
    """Assign each quarter's value to its three months, optionally lagged by whole quarters.

    With ``lag_quarters=1`` month *m* uses the estimate for the quarter *before* the one
    containing *m*, i.e. information that is fully observed when *m* starts.
    """
    shifted = quarterly.set_axis(quarterly.index + lag_quarters, axis=0)
    months = pd.period_range(
        shifted.index[0].asfreq("M", how="start"),
        shifted.index[-1].asfreq("M", how="end"),
        freq="M",
    )
    quarter_of_month = months.asfreq("Q")
    out = shifted.reindex(quarter_of_month).set_axis(months, axis=0)
    out.index.name = "month"
    return out
