import numpy as np
import pandas as pd
import pytest

from climate_risk.term_structure import nelson_siegel as ns

MATURITIES = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)


def _synthetic_yields(lam: float, n: int = 60, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    betas = np.column_stack(
        [
            5 + rng.normal(0, 0.5, n),
            -2 + rng.normal(0, 0.8, n),
            1 + rng.normal(0, 1.0, n),
        ]
    )
    yields = betas @ ns.loadings(MATURITIES, lam).T + rng.normal(0, noise, (n, MATURITIES.size))
    months = pd.period_range("2000-01", periods=n, freq="M", name="month")
    return pd.DataFrame(yields, index=months, columns=MATURITIES)


def test_loadings_limits_and_shape() -> None:
    load = ns.loadings(np.array([1e-10, 1.0, 1e4]), lam=0.5)
    assert load.shape == (3, 3)
    np.testing.assert_allclose(load[:, 0], 1.0)
    np.testing.assert_allclose(load[0, 1:], [1.0, 0.0], atol=1e-8)  # tau -> 0
    np.testing.assert_allclose(load[2, 1:], [0.0, 0.0], atol=1e-3)  # tau -> infinity


def test_curvature_loading_peaks_where_expected() -> None:
    # The curvature loading peaks at λτ ≈ 1.7933; Diebold–Li's λ = 0.0609/month puts the peak at
    # 29.4 months, which they round to "30 months".
    taus = np.linspace(0.5, 10, 20_000)
    lam = 0.0609 * 12
    curvature = ns.loadings(taus, lam=lam)[:, 2]
    assert taus[np.argmax(curvature)] == pytest.approx(1.7933 / lam, abs=1e-3)
    assert taus[np.argmax(curvature)] == pytest.approx(2.5, abs=0.06)


def test_fit_betas_recovers_exact_factors() -> None:
    yields = _synthetic_yields(lam=0.45)
    betas = ns.fit_betas(yields, lam=0.45)
    rebuilt = ns.fitted_yields(betas, MATURITIES, lam=0.45)
    np.testing.assert_allclose(rebuilt.to_numpy(), yields.to_numpy(), atol=1e-10)
    assert list(betas.columns) == list(ns.FACTORS)


def test_profile_lambda_finds_true_decay() -> None:
    yields = _synthetic_yields(lam=0.37, noise=0.01)
    fit = ns.fit(yields, grid=(0.05, 1.5, 291))
    assert fit.lam == pytest.approx(0.37, abs=0.011)
    assert fit.rmse_bp.max() < 2.0  # residual noise is 1bp
    assert fit.sse_profile.idxmin() == pytest.approx(fit.lam, abs=0.006)  # grid step 0.005


def test_pooled_rmse_profile_minimum_at_true_decay() -> None:
    yields = _synthetic_yields(lam=0.6, noise=0.02)
    profile = ns.pooled_rmse_profile(yields, np.linspace(0.3, 0.9, 61))
    assert profile.idxmin() == pytest.approx(0.6, abs=0.011)
    # in-sample residuals lose 3 of 9 degrees of freedom per date: 2bp · sqrt(6/9)
    assert profile.min() == pytest.approx(2.0 * np.sqrt(6 / 9), rel=0.05)


def test_fit_rejects_missing_values() -> None:
    yields = _synthetic_yields(lam=0.5)
    yields.iloc[3, 2] = np.nan
    with pytest.raises(ValueError, match="missing"):
        ns.fit_betas(yields, lam=0.5)


def test_zero_rate_curve_is_continuous_in_maturity() -> None:
    betas = pd.Series({"level": 5.0, "slope": -2.0, "curvature": 1.0})
    curve = ns.zero_rates(betas, np.array([0.25, 0.5, 1.0]), lam=0.5)
    assert curve[0] < curve[1] < curve[2] < 5.0


def test_decay_at_the_edge_of_the_grid_is_flagged(caplog: pytest.LogCaptureFixture) -> None:
    months = pd.period_range("2000-01", periods=24, freq="M")
    taus = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    betas = pd.DataFrame({"level": 4.0, "slope": -2.0, "curvature": 1.0}, index=months)
    yields = ns.fitted_yields(betas, taus, 0.1).set_axis(list(taus), axis=1)
    with caplog.at_level("WARNING"):
        fit = ns.fit(yields, (0.25, 1.5, 26))
    assert fit.lam == pytest.approx(0.25, abs=0.06)
    assert "edge of the search grid" in caplog.text


def test_curvature_peak_is_exact() -> None:
    lam = 0.473
    taus = np.linspace(0.01, 30.0, 300_001)
    peak = taus[np.argmax(ns.loadings(taus, lam)[:, 2])]
    assert ns.CURVATURE_ARGMAX / lam == pytest.approx(peak, abs=1e-3)
