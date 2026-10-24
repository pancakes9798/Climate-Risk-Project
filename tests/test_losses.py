import numpy as np
import pytest

from climate_risk.portfolio import copula, losses


def test_default_losses_counts_threshold_breaches() -> None:
    u = np.array([[0.01, 0.5], [0.2, 0.01], [0.9, 0.9]])
    out = losses.default_losses(
        u, pds=np.array([0.05, 0.05]), weights=np.array([0.5, 0.5]), lgd=0.6
    )
    np.testing.assert_allclose(out, [0.3, 0.3, 0.0])


def test_var_es_on_known_distribution() -> None:
    sample = np.arange(1, 101, dtype=float)  # uniform on 1..100
    var, es = losses.var_es(sample, alpha=0.95)
    assert var == pytest.approx(95.0)
    assert es == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert es >= var


def test_var_es_handles_atoms() -> None:
    sample = np.concatenate([np.zeros(98), np.ones(2)])
    var, es = losses.var_es(sample, alpha=0.99)
    assert var == 1.0
    assert es == pytest.approx(1.0)


def test_mixture_interpolates_between_scenarios() -> None:
    rng = np.random.default_rng(0)
    base = rng.exponential(1.0, 5_000)
    adverse = rng.exponential(3.0, 5_000)
    at_zero = losses.mixture_var_es([base, adverse], [1.0, 0.0], alpha=0.99)
    at_one = losses.mixture_var_es([base, adverse], [0.0, 1.0], alpha=0.99)
    assert at_zero == pytest.approx(losses.var_es(base, 0.99))
    assert at_one == pytest.approx(losses.var_es(adverse, 0.99))
    middle = losses.mixture_var_es([base, adverse], [0.5, 0.5], alpha=0.99)
    assert at_zero[1] < middle[1] < at_one[1]


def test_basel_correlation_bounds() -> None:
    assert losses.basel_correlation(np.array([1e-6]))[0] == pytest.approx(0.24, abs=1e-3)
    assert losses.basel_correlation(np.array([0.5]))[0] == pytest.approx(0.12, abs=1e-3)


@pytest.mark.slow
def test_monte_carlo_matches_vasicek_large_portfolio_limit() -> None:
    rng = np.random.default_rng(1)
    d, rho, pd_, lgd = 1_000, 0.2, 0.02, 0.6
    R = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    u = copula.simulate_gaussian(R, n=20_000, rng=rng)
    sample = losses.default_losses(u, np.full(d, pd_), np.full(d, 1 / d), lgd)
    var, _ = losses.var_es(sample, alpha=0.99)
    assert var == pytest.approx(losses.vasicek_quantile(pd_, rho, 0.99, lgd), rel=0.06)
