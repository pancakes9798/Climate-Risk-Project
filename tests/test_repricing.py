import numpy as np
import pandas as pd
import pytest

from climate_risk.term_structure import projection
from climate_risk.term_structure.endpoint import ScenarioPath

MONTHS = pd.period_range("2026-10", periods=120, freq="M", name="month")


def _path(level: np.ndarray | float, slope: np.ndarray | float) -> ScenarioPath:
    n = len(MONTHS)
    zeros = np.zeros(n)
    return ScenarioPath(
        scenario="x",
        months=MONTHS,
        growth=zeros,
        reference_growth=zeros,
        rstar=zeros,
        istar=np.broadcast_to(np.asarray(level, dtype=float), (n,)).copy(),
        carbon_share=zeros,
        carbon_inflation=zeros,
        slope_shift=np.broadcast_to(np.asarray(slope, dtype=float), (n,)).copy(),
    )


def test_short_rate_is_level_plus_slope() -> None:
    factors = pd.DataFrame({"level": 4.0, "slope": -1.5, "curvature": 0.0}, index=MONTHS)
    np.testing.assert_allclose(projection.short_rate(factors), 2.5)


def test_constant_endpoint_gives_a_flat_overlay() -> None:
    out = projection.overlay(_path(3.5, 0.0), [1.0, 10.0, 30.0])
    np.testing.assert_allclose(out, 3.5)


def test_overlay_averages_the_short_rate_path_and_holds_it_flat_after_the_grid() -> None:
    slope = np.zeros(120)
    slope[:24] = 0.6  # short rates 60bp higher for two years
    out = projection.overlay(_path(4.0, slope), [1.0, 2.0, 10.0, 20.0])
    np.testing.assert_allclose(out[0], [4.6, 4.6, 4.0 + 0.6 * 24 / 120, 4.0 + 0.6 * 24 / 240])
    np.testing.assert_allclose(out[-1], 4.0)  # beyond the grid the last value (no shift) holds


def test_repricing_is_the_overlay_gap_at_the_revelation_month() -> None:
    slope = np.zeros(120)
    slope[12:36] = 0.3
    policy, baseline = _path(3.9, slope), _path(4.0, 0.0)
    gap_now = projection.repricing(policy, baseline, [1.0, 5.0])
    np.testing.assert_allclose(gap_now, [-0.1, -0.1 + 0.3 * 24 / 60])
    gap_later = projection.repricing(policy, baseline, [1.0], month=12)
    np.testing.assert_allclose(gap_later, [0.2])


def test_overlay_rejects_sub_monthly_maturities() -> None:
    with pytest.raises(ValueError, match="month"):
        projection.overlay(_path(4.0, 0.0), [0.01])
