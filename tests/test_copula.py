import numpy as np
import pandas as pd
import pytest
from scipy import stats

from climate_risk.portfolio import copula


def _equicorrelation(d: int, rho: float) -> np.ndarray:
    return np.full((d, d), rho) + (1 - rho) * np.eye(d)


def test_tau_to_rho_known_values() -> None:
    np.testing.assert_allclose(copula.tau_to_rho(np.array([0.0, 1 / 3, 1.0])), [0.0, 0.5, 1.0])


def test_kendall_matrix_is_symmetric_with_unit_diagonal() -> None:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    tau = copula.kendall_tau_matrix(data)
    np.testing.assert_allclose(tau.to_numpy(), tau.to_numpy().T)
    np.testing.assert_allclose(np.diag(tau), 1.0)
    expected = stats.kendalltau(data["a"], data["c"]).statistic
    assert tau.loc["a", "c"] == pytest.approx(expected)


def test_block_correlation_averages_within_and_between_groups() -> None:
    names = ["a1", "a2", "b1", "b2"]
    rho = pd.DataFrame(
        [[1.0, 0.6, 0.2, 0.3], [0.6, 1.0, 0.1, 0.2], [0.2, 0.1, 1.0, 0.4], [0.3, 0.2, 0.4, 1.0]],
        index=names,
        columns=names,
    )
    groups = pd.Series({"a1": "A", "a2": "A", "b1": "B", "b2": "B"})
    full, blocks = copula.block_correlation(rho, groups)
    assert blocks.loc["A", "A"] == pytest.approx(0.6)
    assert blocks.loc["A", "B"] == pytest.approx(0.2)
    assert full.loc["a1", "b2"] == pytest.approx(0.2)
    assert full.loc["b1", "b2"] == pytest.approx(0.4)
    assert np.all(np.linalg.eigvalsh(full.to_numpy()) > 0)


def test_simulated_t_copula_matches_kendall_tau() -> None:
    rng = np.random.default_rng(1)
    R = _equicorrelation(3, 0.5)
    u = copula.simulate_t(R, nu=6.0, n=20_000, rng=rng)
    tau = stats.kendalltau(u[:, 0], u[:, 1]).statistic
    assert tau == pytest.approx(2 / np.pi * np.arcsin(0.5), abs=0.02)
    assert u.min() > 0
    assert u.max() < 1


def test_t_loglik_approaches_gaussian_for_large_nu() -> None:
    rng = np.random.default_rng(2)
    R = _equicorrelation(3, 0.3)
    u = copula.simulate_gaussian(R, n=500, rng=rng)
    x = stats.norm.ppf(u)
    gaussian = np.sum(stats.multivariate_normal(cov=R).logpdf(x) - stats.norm.logpdf(x).sum(1))
    assert copula.t_copula_loglik(u, R, nu=1e6) == pytest.approx(gaussian, rel=1e-3)


def test_fit_nu_recovers_tail_dependence() -> None:
    rng = np.random.default_rng(3)
    R = _equicorrelation(5, 0.4)
    u = copula.simulate_t(R, nu=5.0, n=4_000, rng=rng)
    nu_hat = copula.fit_nu(u, R)
    assert 3.5 < nu_hat < 7.5
    thin = copula.simulate_gaussian(R, n=4_000, rng=rng)
    assert copula.fit_nu(thin, R) > 30


def test_pseudo_observations_are_scaled_ranks() -> None:
    data = pd.DataFrame({"x": [3.0, 1.0, 2.0]})
    np.testing.assert_allclose(copula.pseudo_observations(data)[:, 0], [0.75, 0.25, 0.5])
