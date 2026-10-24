import numpy as np
import pytest

from climate_risk.credit import hazard

TENORS = np.arange(1.0, 11.0)


def flat_discount(rate: float):
    return lambda t: np.exp(-rate * np.asarray(t, dtype=float))


def test_survival_of_piecewise_constant_hazard() -> None:
    curve = hazard.HazardCurve(tenors=np.array([1.0, 3.0]), hazards=np.array([0.01, 0.03]))
    np.testing.assert_allclose(
        curve.survival(np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])),
        np.exp(-np.array([0.0, 0.005, 0.01, 0.04, 0.07, 0.10])),
    )
    assert curve.default_probability(np.array([3.0]))[0] == pytest.approx(1 - np.exp(-0.07))


def test_credit_triangle_holds_for_flat_curve() -> None:
    curve = hazard.HazardCurve(tenors=TENORS, hazards=np.full(10, 0.02))
    spreads = hazard.par_spreads(curve, TENORS, flat_discount(0.03), recovery=0.4)
    np.testing.assert_allclose(spreads, 0.02 * 0.6 * 1e4, rtol=0.01)


def test_bootstrap_reprices_market_quotes_exactly() -> None:
    quotes = np.array([15, 25, 39, 52, 65, 78, 88, 97, 104, 110], dtype=float)
    discount = flat_discount(0.025)
    curve = hazard.bootstrap(quotes, TENORS, discount, recovery=0.4)
    np.testing.assert_allclose(hazard.par_spreads(curve, TENORS, discount, 0.4), quotes, atol=1e-7)
    assert np.all(np.diff(curve.hazards) > 0)  # upward-sloping CDS curve


def test_bootstrap_recovers_known_hazards() -> None:
    true = hazard.HazardCurve(tenors=TENORS, hazards=np.linspace(0.004, 0.03, 10))
    discount = flat_discount(0.01)
    quotes = hazard.par_spreads(true, TENORS, discount, 0.4)
    fitted = hazard.bootstrap(quotes, TENORS, discount, recovery=0.4)
    np.testing.assert_allclose(fitted.hazards, true.hazards, rtol=1e-6)


def test_bootstrap_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="negative"):
        hazard.bootstrap(np.array([-1.0, 10.0]), np.array([1.0, 2.0]), flat_discount(0.0), 0.4)
    with pytest.raises(ValueError, match="recovery"):
        hazard.bootstrap(np.array([10.0]), np.array([1.0]), flat_discount(0.0), 1.0)
    with pytest.raises(ValueError, match="increasing"):
        hazard.bootstrap(np.array([10.0, 20.0]), np.array([2.0, 1.0]), flat_discount(0.0), 0.4)


def test_to_physical_scales_cumulative_hazard() -> None:
    q = np.array([0.0, 0.1, 0.5])
    np.testing.assert_allclose(hazard.to_physical(q, 1.0), q)
    physical = hazard.to_physical(q, 2.0)
    np.testing.assert_allclose(-np.log1p(-physical), -np.log1p(-q) / 2.0)
    with pytest.raises(ValueError, match="ratio"):
        hazard.to_physical(q, 0.5)


def test_bootstrap_flags_arbitrageable_inversion() -> None:
    # a 2y spread far below the 1y one would require a negative forward hazard
    with pytest.raises(ValueError, match="arbitrage"):
        hazard.bootstrap(np.array([500.0, 50.0]), np.array([1.0, 2.0]), flat_discount(0.0), 0.4)
