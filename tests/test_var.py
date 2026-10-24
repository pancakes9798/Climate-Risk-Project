import numpy as np
import pandas as pd
import pytest

from climate_risk.term_structure import var

PHI = np.array([[0.97, 0.02, 0.0], [0.0, 0.93, 0.05], [0.0, 0.0, 0.85]])
MEAN = np.array([1.5, -1.0, 0.5])
CHOL = np.diag([0.25, 0.35, 0.6])


def _simulate(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.empty((n, 3))
    x[0] = MEAN
    for t in range(1, n):
        x[t] = MEAN + PHI @ (x[t - 1] - MEAN) + CHOL @ rng.standard_normal(3)
    return pd.DataFrame(x, columns=["level", "slope", "curvature"])


def test_fit_var1_recovers_parameters() -> None:
    model = var.fit_var1(_simulate(20_000))
    np.testing.assert_allclose(model.phi, PHI, atol=0.02)
    np.testing.assert_allclose(model.mean, MEAN, atol=0.15)
    np.testing.assert_allclose(model.sigma, CHOL @ CHOL.T, rtol=0.05, atol=0.005)
    assert model.is_stationary
    assert model.names == ("level", "slope", "curvature")


def test_half_lives_match_eigenvalues() -> None:
    model = var.VAR1(
        intercept=np.zeros(2), phi=np.diag([0.5, 0.9]), sigma=np.eye(2), names=("a", "b"), nobs=10
    )
    np.testing.assert_allclose(sorted(model.half_lives), [1.0, np.log(0.5) / np.log(0.9)])


def test_ar1_diagonal_has_no_cross_dynamics() -> None:
    model = var.fit_ar1_diagonal(_simulate(5_000))
    assert np.count_nonzero(model.phi - np.diag(np.diag(model.phi))) == 0
    np.testing.assert_allclose(np.diag(model.phi), np.diag(PHI), atol=0.03)


def test_forecast_converges_to_mean() -> None:
    model = var.fit_var1(_simulate(5_000))
    path = model.forecast(np.array([5.0, 3.0, -4.0]), horizon=2_000)
    np.testing.assert_allclose(path[-1], model.mean, atol=1e-6)
    assert path.shape == (2_000, 3)


def test_simulate_with_zero_shocks_equals_forecast() -> None:
    model = var.fit_var1(_simulate(2_000))
    x0 = np.array([2.0, -2.0, 1.0])
    paths = model.simulate(x0, np.zeros((24, 5, 3)))
    np.testing.assert_allclose(paths[:, 0, :], model.forecast(x0, 24))
    assert paths.shape == (24, 5, 3)


def test_simulated_unconditional_moments() -> None:
    model = var.fit_var1(_simulate(5_000))
    rng = np.random.default_rng(3)
    paths = model.simulate(model.mean, rng.standard_normal((400, 4_000, 3)))
    sample = paths[-1]
    np.testing.assert_allclose(sample.mean(axis=0), model.mean, atol=0.1)
    # sampling s.e. of a covariance entry with 4000 draws is ~0.03
    np.testing.assert_allclose(np.cov(sample.T), model.unconditional_covariance(), atol=0.12)


def test_fit_requires_enough_observations() -> None:
    with pytest.raises(ValueError, match="observations"):
        var.fit_var1(_simulate(4))


def test_half_lives_of_unit_and_explosive_roots_are_infinite() -> None:
    model = var.VAR1(
        intercept=np.zeros(3),
        phi=np.diag([1.0, 1.02, 0.5]),
        sigma=np.eye(3),
        names=("a", "b", "c"),
        nobs=10,
    )
    np.testing.assert_allclose(model.half_lives, [np.inf, np.inf, 1.0])
    assert not model.is_stationary
