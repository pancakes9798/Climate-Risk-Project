import numpy as np
import pandas as pd
import pytest

from climate_risk.term_structure import backtest
from climate_risk.term_structure import nelson_siegel as ns

LAM = 0.5
MATS = [1.0, 5.0, 10.0]


def _panel(n: int = 420, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Mean-reverting factors around a slowly drifting endpoint."""
    rng = np.random.default_rng(seed)
    months = pd.period_range("1995-01", periods=n, freq="M", name="month")
    istar = pd.Series(np.linspace(5.0, 3.0, n), index=months)
    x = np.zeros((n, 3))
    x[0] = [2.0, -3.0, 2.0]
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0, 0.15, 3)
    betas = x + np.column_stack([istar.to_numpy(), np.zeros(n), np.zeros(n)])
    yields = pd.DataFrame(betas @ ns.loadings(MATS, LAM).T, index=months, columns=MATS)
    return yields, istar


@pytest.fixture(scope="module")
def errors() -> pd.DataFrame:
    yields, istar = _panel()
    return backtest.forecast_errors(
        yields,
        istar,
        LAM,
        horizons=(1, 12),
        maturities=MATS,
        first_origin=pd.Period("2001-01", "M"),
    )


def test_error_frame_layout(errors: pd.DataFrame) -> None:
    assert set(errors["model"]) == set(backtest.MODELS)
    assert set(errors["horizon"]) == {1, 12}
    assert {"origin", "maturity", "error"} <= set(errors.columns)


def test_random_walk_error_is_change_in_yield(errors: pd.DataFrame) -> None:
    yields, _ = _panel()
    row = errors.query("model == 'Random walk' and horizon == 12 and maturity == 5.0").iloc[0]
    origin = row["origin"]
    expected = yields.loc[origin, 5.0] - yields.loc[origin + 12, 5.0]
    assert row["error"] == pytest.approx(expected)


def test_model_based_forecasts_beat_random_walk_at_long_horizon(errors: pd.DataFrame) -> None:
    table = backtest.rmse_table(errors)
    rel = backtest.relative_rmse(table)
    assert rel.loc[("Shifting-endpoint VAR(1)", 12)].max() < 1.0
    assert (table.loc[("Random walk", 1)] > 0).all()


def _error_frame(model_errors: np.ndarray, bench_errors: np.ndarray, horizon: int) -> pd.DataFrame:
    n = model_errors.size
    origins = pd.period_range("2000-01", periods=n, freq="M")
    return pd.DataFrame(
        {
            "origin": np.concatenate([origins, origins]),
            "horizon": horizon,
            "maturity": 1.0,
            "model": ["Random walk"] * n + ["DNS AR(1)"] * n,
            "error": np.concatenate([bench_errors, model_errors]),
        }
    )


def test_diebold_mariano_detects_better_model() -> None:
    rng = np.random.default_rng(5)
    model_err = rng.normal(0, 1.0, 300)
    bench_err = model_err + rng.normal(0, 1.0, 300)
    stat, pvalue = backtest.diebold_mariano(
        _error_frame(model_err, bench_err, 1), "DNS AR(1)", horizon=1, maturity=1.0
    )
    d = bench_err**2 - model_err**2
    n = d.size
    manual = d.mean() / np.sqrt(d.var() / n) * np.sqrt((n - 1) / n)  # HLN factor for h = 1
    assert stat == pytest.approx(manual)
    assert pvalue < 1e-6


def test_diebold_mariano_uses_hac_lags_for_overlapping_errors() -> None:
    rng = np.random.default_rng(6)
    base = rng.normal(0, 1.0, 400)
    model_err = np.convolve(base, np.ones(6), mode="same")  # MA(5) overlap as in h = 6
    bench_err = model_err + np.convolve(rng.normal(0, 1, 400), np.ones(6), mode="same")
    frame = _error_frame(model_err, bench_err, 6)
    stat_h6, _ = backtest.diebold_mariano(frame, "DNS AR(1)", horizon=6, maturity=1.0)
    frame_h1 = frame.assign(horizon=1)
    stat_h1, _ = backtest.diebold_mariano(frame_h1, "DNS AR(1)", horizon=1, maturity=1.0)
    assert 0 < stat_h6 < stat_h1  # HAC variance accounts for the positive autocorrelation


def test_diebold_mariano_identical_forecasts() -> None:
    frame = pd.DataFrame(
        {
            "origin": pd.period_range("2000-01", periods=30, freq="M").repeat(2),
            "horizon": 1,
            "maturity": 1.0,
            "model": ["Random walk", "DNS AR(1)"] * 30,
            "error": np.repeat(np.random.default_rng(1).normal(size=30), 2),
        }
    )
    stat, pvalue = backtest.diebold_mariano(frame, "DNS AR(1)", horizon=1, maturity=1.0)
    assert stat == 0.0
    assert pvalue == 1.0


def test_diebold_mariano_falls_back_to_bartlett_weights() -> None:
    # strongly negatively autocorrelated differential: equal-weight variance turns negative
    rng = np.random.default_rng(9)
    n, horizon = 300, 12
    pattern = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    model_err = rng.normal(0, 0.1, n)
    bench_err = np.sqrt(np.maximum(model_err**2 + 0.05 + pattern * 0.5, 0.0))
    frame = _error_frame(model_err, bench_err, horizon)
    d = bench_err**2 - model_err**2
    assert backtest._long_run_variance(d, horizon, bartlett=False) <= 0.0
    stat, pvalue = backtest.diebold_mariano(frame, "DNS AR(1)", horizon=horizon, maturity=1.0)
    assert np.isfinite(stat)
    assert 0.0 <= pvalue <= 1.0
