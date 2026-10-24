import numpy as np
import pandas as pd
import pytest

from climate_risk.term_structure import nelson_siegel as ns
from climate_risk.term_structure import projection
from climate_risk.term_structure.endpoint import ScenarioPath
from climate_risk.term_structure.model import ShiftingEndpointDNS, endpoint_cycle, fit_model
from climate_risk.term_structure.var import VAR1

LAM = 0.45
MATS = (1.0, 10.0, 30.0)
MONTHS = pd.period_range("2026-10", "2030-12", freq="M", name="month")


def _path(name: str, level: float, slope: float) -> ScenarioPath:
    n = len(MONTHS)
    return ScenarioPath(
        scenario=name,
        months=MONTHS,
        growth=np.zeros(n),
        reference_growth=np.zeros(n),
        rstar=np.zeros(n),
        istar=np.full(n, level),
        carbon_share=np.zeros(n),
        carbon_inflation=np.zeros(n),
        slope_shift=np.full(n, slope),
    )


def _model(scale: float) -> ShiftingEndpointDNS:
    var = VAR1(
        intercept=np.array([0.05, -0.1, 0.1]),
        phi=np.diag([0.9, 0.9, 0.8]),
        sigma=scale * np.diag([0.1, 0.2, 0.5]),
        names=ns.FACTORS,
        nobs=100,
    )
    return ShiftingEndpointDNS(lam=LAM, var=var)


def test_endpoint_cycle_removes_istar_from_level_only() -> None:
    months = pd.period_range("2000-01", periods=3, freq="M", name="month")
    betas = pd.DataFrame(
        {"level": [5.0, 6.0, 7.0], "slope": [-1.0] * 3, "curvature": [0.5] * 3}, index=months
    )
    istar = pd.Series([4.0, 4.5, np.nan], index=months)
    cycle = endpoint_cycle(betas, istar)
    assert list(cycle["level"]) == [1.0, 1.5]
    assert list(cycle["slope"]) == [-1.0, -1.0]


def test_fit_model_returns_stationary_var_on_cycle() -> None:
    rng = np.random.default_rng(0)
    months = pd.period_range("1990-01", periods=400, freq="M", name="month")
    istar = pd.Series(np.linspace(6, 3, 400), index=months)
    cycle = np.zeros((400, 3))
    for t in range(1, 400):
        cycle[t] = 0.9 * cycle[t - 1] + rng.normal(0, 0.2, 3)
    betas = pd.DataFrame(cycle, index=months, columns=list(ns.FACTORS))
    betas["level"] += istar
    model = fit_model(betas, istar, lam=LAM)
    assert model.var.is_stationary
    np.testing.assert_allclose(np.diag(model.var.phi), 0.9, atol=0.06)


def test_deterministic_projection_is_cycle_yield_plus_overlay() -> None:
    model = _model(scale=1e-12)
    x0 = model.var.mean
    result = projection.project(model, x0, {"A": _path("A", 3.0, 0.2)}, MATS, n_paths=50, seed=1)
    table = result.yields.loc["A"]
    # a permanent short-rate shift of 3.0 + 0.2 moves every maturity by 3.2 under the EH
    expected = ns.zero_rates(x0, MATS, LAM) + 3.2
    for maturity, value in zip(MATS, expected, strict=True):
        assert table.loc[(2030, maturity), "mean"] == pytest.approx(value, abs=1e-6)
        assert table.loc[(2030, maturity), "q05"] == pytest.approx(value, abs=1e-5)


def test_common_random_numbers_make_scenario_gaps_deterministic() -> None:
    model = _model(scale=1.0)
    paths = {"A": _path("A", 3.0, 0.0), "B": _path("B", 4.0, 0.5)}
    result = projection.project(model, model.var.mean, paths, MATS, n_paths=400, seed=3)
    gap = result.yields.loc["B"] - result.yields.loc["A"]
    np.testing.assert_allclose(gap.to_numpy(), 1.5, atol=1e-9)  # (4.0 + 0.5) - 3.0 everywhere


def test_quantiles_are_ordered_and_reproducible() -> None:
    model = _model(scale=1.0)
    paths = {"A": _path("A", 3.0, 0.0)}
    first = projection.project(model, model.var.mean, paths, MATS, n_paths=300, seed=11)
    second = projection.project(model, model.var.mean, paths, MATS, n_paths=300, seed=11)
    pd.testing.assert_frame_equal(first.yields, second.yields)
    table = first.yields
    assert (table["q05"] <= table["q50"]).all()
    assert (table["q50"] <= table["q95"]).all()
    assert set(first.factors.columns) == {"level", "slope", "curvature"}


def test_project_rejects_misaligned_paths() -> None:
    model = _model(scale=1.0)
    other = ScenarioPath(**{**_path("B", 3.0, 0.0).__dict__, "months": MONTHS + 1})
    with pytest.raises(ValueError, match="months"):
        projection.project(model, model.var.mean, {"A": _path("A", 3, 0), "B": other}, MATS)


@pytest.mark.parametrize(
    ("q", "label"),
    [(0.05, "q05"), (0.5, "q50"), (0.025, "q2_5"), (0.975, "q97_5"), (0.995, "q99_5")],
)
def test_quantile_labels(q: float, label: str) -> None:
    assert projection.quantile_label(q) == label


def test_colliding_quantile_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="distinct labels"):
        projection.project(
            _model(1.0),
            np.zeros(3),
            {"A": _path("A", 3.0, 0.0)},
            MATS,
            n_paths=10,
            quantiles=(0.5, 0.5 + 1e-10),
        )


def test_non_stationary_cycle_is_rejected() -> None:
    months = pd.period_range("2000-01", periods=240, freq="M")
    trend = np.cumsum(np.full(len(months), 0.05))
    betas = pd.DataFrame(
        {"level": 3.0 + trend, "slope": -1.0 + 0.0 * trend, "curvature": 0.5 + 0.0 * trend},
        index=months,
    )
    rng = np.random.default_rng(0)
    betas = betas + rng.normal(0, 0.01, betas.shape)
    explosive = betas.assign(level=1.03 ** np.arange(len(months)))
    with pytest.raises(ValueError, match="not stationary"):
        fit_model(explosive, pd.Series(0.0, index=months), lam=LAM)


def test_cycle_with_missing_months_is_rejected() -> None:
    months = pd.period_range("2000-01", periods=60, freq="M")
    rng = np.random.default_rng(1)
    betas = pd.DataFrame(rng.normal(size=(60, 3)), index=months, columns=list(ns.FACTORS))
    istar = pd.Series(2.0, index=months).drop(months[30])
    with pytest.raises(ValueError, match="missing months"):
        fit_model(betas, istar, lam=LAM)
