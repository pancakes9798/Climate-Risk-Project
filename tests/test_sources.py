from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_risk.data import sources

GSW_TEXT = """\
"Note: This is not an official Federal Reserve statistical release."

Series,Compounding Convention,Mnemonic(s)
Zero-coupon yield,Continuously Compounded,SVENYXX

Date,BETA0,SVENY01,SVENY02,SVENY10
2020-01-30,1.0,1.50,1.60,1.90
2020-01-31,1.0,1.51,1.61,NA
2020-02-27,1.0,1.40,1.45,1.70
2020-02-28,1.0,1.41,1.46,1.71
"""


def test_read_gsw_zero_yields_parses_maturities_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "gsw.csv"
    path.write_text(GSW_TEXT)
    curves = sources.read_gsw_zero_yields(path)
    assert list(curves.columns) == [1.0, 2.0, 10.0]
    assert np.isnan(curves.loc["2020-01-31", 10.0])
    assert curves.loc["2020-02-28", 2.0] == pytest.approx(1.46)


def test_month_end_curves_takes_last_complete_day(tmp_path: Path) -> None:
    path = tmp_path / "gsw.csv"
    path.write_text(GSW_TEXT)
    monthly = sources.month_end_curves(sources.read_gsw_zero_yields(path), [1.0, 10.0])
    assert list(monthly.index.astype(str)) == ["2020-01", "2020-02"]
    # 31 Jan lacks the 10y yield, so 30 Jan is the last complete observation of January
    assert monthly.loc[pd.Period("2020-01", "M"), 1.0] == pytest.approx(1.50)
    assert monthly.loc[pd.Period("2020-02", "M"), 10.0] == pytest.approx(1.71)


def test_month_end_curves_rejects_unknown_maturity(tmp_path: Path) -> None:
    path = tmp_path / "gsw.csv"
    path.write_text(GSW_TEXT)
    with pytest.raises(sources.DataFormatError):
        sources.month_end_curves(sources.read_gsw_zero_yields(path), [30.0])


def test_missing_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(sources.DataFormatError):
        sources.read_gsw_zero_yields(path)


def test_read_fred_csv_handles_dots(tmp_path: Path) -> None:
    path = tmp_path / "fred.csv"
    path.write_text("observation_date,BAA10Y\n2020-01-02,2.1\n2020-01-03,.\n2020-01-06,2.3\n")
    series = sources.read_fred_csv(path)
    assert series.name == "BAA10Y"
    assert len(series) == 2
    monthly = sources.to_monthly(series, how="mean")
    assert monthly.iloc[0] == pytest.approx(2.2)
    assert sources.to_monthly(series).iloc[0] == pytest.approx(2.3)


def test_read_fred_csv_rejects_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "fred.csv"
    path.write_text("a,b,c\n2020-01-01,1,2\n")
    with pytest.raises(sources.DataFormatError):
        sources.read_fred_csv(path)


def _write_hlw_workbook(path: Path) -> None:
    rows = [
        ["This spreadsheet contains updated estimates", None, None, None, None, None, None],
        [None] * 7,
        [
            None,
            None,
            "Trend Growth (g), Annualized",
            None,
            "Other Determinants (z)",
            None,
            "Natural Rate (r*)",
        ],
        ["Date", None, "US", "Canada", "US", "Canada", "US"],
        [pd.Timestamp("2020-01-01"), None, 1.8, 1.5, -1.0, -0.5, 0.8],
        [pd.Timestamp("2020-04-01"), None, 1.7, 1.4, -1.1, -0.4, 0.6],
        [None] * 7,
    ]
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="HLW Estimates", header=False, index=False)


def test_read_hlw_us_locates_us_columns(tmp_path: Path) -> None:
    path = tmp_path / "hlw.xlsx"
    _write_hlw_workbook(path)
    hlw = sources.read_hlw_us(path)
    assert list(hlw.columns) == ["g", "z", "rstar"]
    assert list(hlw.index.astype(str)) == ["2020Q1", "2020Q2"]
    assert hlw.loc[pd.Period("2020Q2", "Q"), "rstar"] == pytest.approx(0.6)


def test_read_hlw_us_without_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "hlw.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([[1, 2]]).to_excel(writer, sheet_name="HLW Estimates", header=False)
    with pytest.raises(sources.DataFormatError):
        sources.read_hlw_us(path)


def test_quarterly_to_monthly_with_lag() -> None:
    quarterly = pd.DataFrame(
        {"rstar": [1.0, 2.0]}, index=pd.PeriodIndex(["2020Q1", "2020Q2"], freq="Q")
    )
    same = sources.quarterly_to_monthly(quarterly)
    assert same.loc[pd.Period("2020-03", "M"), "rstar"] == 1.0
    assert same.loc[pd.Period("2020-04", "M"), "rstar"] == 2.0
    lagged = sources.quarterly_to_monthly(quarterly, lag_quarters=1)
    assert lagged.index[0] == pd.Period("2020-04", "M")
    assert lagged.loc[pd.Period("2020-06", "M"), "rstar"] == 1.0
    assert lagged.loc[pd.Period("2020-09", "M"), "rstar"] == 2.0
