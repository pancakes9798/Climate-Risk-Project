from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_risk.credit import cds


def _write_cds(path: Path) -> None:
    dates = pd.bdate_range("2015-01-01", "2015-04-30")
    rows = []
    for ticker, level in [("AAA", 10.0), ("BBB", 50.0), ("CCC", 80.0)]:
        for i, date in enumerate(dates):
            quotes = {f"PX{k}": level * (1 + 0.1 * k) + 0.01 * i for k in range(1, 11)}
            rows.append(
                {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Company": f"{ticker} Corp",
                    **quotes,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[5, "PX3"] = np.nan  # an incomplete quote is dropped
    frame.loc[6, "PX4"] = -1.0  # and so is a negative one
    frame = frame[~((frame["Ticker"] == "CCC") & (frame["Date"] > "2015-02-15"))]  # sparse issuer
    frame.to_csv(path, index=False)


@pytest.fixture
def raw(tmp_path: Path) -> pd.DataFrame:
    path = tmp_path / "cds.csv"
    _write_cds(path)
    return cds.load_cds(path)


def test_load_validates_and_cleans(raw: pd.DataFrame) -> None:
    assert list(raw.columns) == ["date", "ticker", "company", *cds.PX]
    assert raw[list(cds.PX)].notna().all().all()
    assert (raw[list(cds.PX)] >= 0).all().all()
    business_days = len(pd.bdate_range("2015-01-01", "2015-04-30"))
    sparse_days = len(pd.bdate_range("2015-01-01", "2015-02-15"))
    assert len(raw) == 2 * business_days + sparse_days - 2


def test_load_filters_tickers(tmp_path: Path) -> None:
    path = tmp_path / "cds.csv"
    _write_cds(path)
    assert set(cds.load_cds(path, tickers=["BBB"])["ticker"]) == {"BBB"}


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "cds.csv"
    pd.DataFrame({"Date": ["2015-01-01"], "Ticker": ["A"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="PX1"):
        cds.load_cds(path)


def test_monthly_panel_and_reference_spreads(raw: pd.DataFrame) -> None:
    panel = cds.monthly_panel(raw, "2015-01", "2015-04")
    assert panel.index.names == ["month", "ticker"]
    ref = cds.reference_spreads(panel, min_coverage=0.9)
    assert list(ref.index) == ["AAA", "BBB"]  # CCC covers only 2 of 4 months
    # month-end quotes of AAA's 1y tenor: 11.21, 11.41, 11.63, 11.85 → median 11.52
    assert ref.loc["AAA", "PX1"] == pytest.approx(11.52)


def test_sector_reference_is_cross_issuer_median(raw: pd.DataFrame) -> None:
    ref = cds.reference_spreads(cds.monthly_panel(raw, "2015-01", "2015-04"), min_coverage=0.9)
    sectors = pd.Series({"AAA": "S1", "BBB": "S1"})
    by_sector = cds.sector_reference(ref, sectors)
    assert by_sector.loc["S1", "PX10"] == pytest.approx(ref["PX10"].median())


def test_stale_issuers_are_flagged() -> None:
    changes = pd.DataFrame({"live": [0.01, -0.02, 0.0, 0.03], "stale": [0.0, 0.0, 0.0, 0.01]})
    assert cds.stale_issuers(changes, max_zero_share=0.25) == ["stale"]


def test_weekly_log_changes_are_wednesday_sampled(raw: pd.DataFrame) -> None:
    changes = cds.weekly_log_changes(raw, "PX5", "2015-01", "2015-04", tickers=["AAA", "BBB"])
    assert list(changes.columns) == ["AAA", "BBB"]
    assert (changes.index.day_of_week == 2).all()
    assert changes.notna().all().all()
