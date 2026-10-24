import numpy as np
import pytest
from scipy.stats import norm

from climate_risk.credit import merton

T = np.arange(1.0, 11.0)
RATES = np.full(10, 0.02)


def test_merton_pd_closed_form() -> None:
    q = merton.merton_pd(a=0.5, sigma=0.2, T=np.array([5.0]), zero_rates=np.array([0.03]))
    expected = norm.cdf(-(0.5 + 0.03 * 5 - 0.5 * 0.04 * 5) / (0.2 * np.sqrt(5)))
    assert q[0] == pytest.approx(expected)


def test_calibration_recovers_parameters() -> None:
    q = merton.merton_pd(a=0.45, sigma=0.17, T=T, zero_rates=RATES)
    fit = merton.calibrate(q, T, RATES)
    assert fit.a == pytest.approx(0.45, rel=1e-4)
    assert fit.sigma == pytest.approx(0.17, rel=1e-4)
    assert fit.rmse_probit < 1e-6


def test_shift_with_zero_shock_is_identity() -> None:
    q = np.array([0.01, 0.05, 0.12])
    np.testing.assert_allclose(merton.shifted_pd(q, T[:3], sigma=0.2, xi=np.zeros(3)), q)


def test_shift_equals_merton_with_rescaled_assets() -> None:
    q_base = merton.merton_pd(a=0.6, sigma=0.25, T=T, zero_rates=RATES)
    xi = np.full(10, -0.3)
    shifted = merton.shifted_pd(q_base, T, sigma=0.25, xi=xi)
    direct = merton.merton_pd(a=0.6 + np.log(0.7), sigma=0.25, T=T, zero_rates=RATES)
    np.testing.assert_allclose(shifted, direct, rtol=1e-10)
    assert np.all(shifted > q_base)


def test_positive_shock_lowers_pd_and_extreme_shock_is_bounded() -> None:
    q = np.array([0.05])
    assert merton.shifted_pd(q, np.array([5.0]), 0.2, np.array([0.3]))[0] < 0.05
    assert merton.shifted_pd(q, np.array([5.0]), 0.2, np.array([-1.0]))[0] == pytest.approx(
        1.0, abs=1e-6
    )


def test_zero_coupon_spread_small_pd_limit() -> None:
    q = np.array([1e-4])
    assert merton.zero_coupon_spread(q, np.array([1.0]), lgd=0.6)[0] == pytest.approx(
        0.6e-4, rel=1e-3
    )
    s = merton.zero_coupon_spread(np.array([0.2]), np.array([10.0]), lgd=0.6)[0]
    assert s == pytest.approx(-np.log(1 - 0.12) / 10)


@pytest.mark.parametrize("bad", [0.0, 1.0, np.nan])
def test_calibration_rejects_degenerate_pds(bad: float) -> None:
    q = merton.merton_pd(a=0.45, sigma=0.17, T=T, zero_rates=RATES)
    q[3] = bad
    with pytest.raises(ValueError, match="strictly between"):
        merton.calibrate(q, T, RATES)


class _Solution:
    def __init__(self, x: list[float], success: bool) -> None:
        self.x, self.success, self.message = np.array(x), success, "stopped"
        self.fun = np.zeros(3)


def test_calibration_reports_non_convergence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(merton, "least_squares", lambda *_, **__: _Solution([0.5, 0.2], False))
    q = merton.merton_pd(a=0.45, sigma=0.17, T=T, zero_rates=RATES)
    with pytest.raises(RuntimeError, match="did not converge"):
        merton.calibrate(q, T, RATES)


def test_calibration_warns_when_volatility_hits_a_bound(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(merton, "least_squares", lambda *_, **__: _Solution([0.5, 2.0], True))
    q = merton.merton_pd(a=0.45, sigma=0.17, T=T, zero_rates=RATES)
    with caplog.at_level("WARNING"):
        fit = merton.calibrate(q, T, RATES)
    assert fit.sigma == pytest.approx(2.0)
    assert "at its bound" in caplog.text
