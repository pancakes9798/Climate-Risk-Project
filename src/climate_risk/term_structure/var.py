"""First-order vector autoregression for the yield-curve factors.

    xₜ = c + Φ xₜ₋₁ + ηₜ,   ηₜ ~ N(0, Σ)

Estimated equation by equation by OLS (identical regressors, hence equal to SUR/Gaussian ML).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_lyapunov

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VAR1:
    intercept: FloatArray
    phi: FloatArray
    sigma: FloatArray
    names: tuple[str, ...]
    nobs: int

    @property
    def k(self) -> int:
        return len(self.names)

    @property
    def mean(self) -> FloatArray:
        """Unconditional mean (I − Φ)⁻¹ c."""
        return np.asarray(np.linalg.solve(np.eye(self.k) - self.phi, self.intercept), np.float64)

    @property
    def eigenvalues(self) -> NDArray[np.complex128]:
        return np.linalg.eigvals(self.phi).astype(np.complex128)

    @property
    def is_stationary(self) -> bool:
        return bool(np.max(np.abs(self.eigenvalues)) < 1.0)

    @property
    def half_lives(self) -> FloatArray:
        """Half-life (in periods) of each eigen-mode; ``inf`` for unit or explosive roots."""
        modulus = np.abs(self.eigenvalues)
        with np.errstate(divide="ignore"):
            return np.where(modulus < 1.0, np.log(0.5) / np.log(modulus), np.inf)

    def unconditional_covariance(self) -> FloatArray:
        """Solves V = Φ V Φ' + Σ."""
        return np.asarray(solve_discrete_lyapunov(self.phi, self.sigma))

    def forecast(self, x0: FloatArray, horizon: int) -> FloatArray:
        """Conditional mean path x̂_{t+1..t+h} (shape ``horizon × k``)."""
        path = np.empty((horizon, self.k))
        x = np.asarray(x0, dtype=float)
        for h in range(horizon):
            x = self.intercept + self.phi @ x
            path[h] = x
        return path

    def simulate(self, x0: FloatArray, shocks: FloatArray) -> FloatArray:
        """Paths driven by standard-normal ``shocks`` of shape ``(steps, paths, k)``.

        Passing the same shocks to several scenarios gives common random numbers, so that
        scenario differences are not contaminated by Monte Carlo noise.
        """
        steps, n_paths, k = shocks.shape
        if k != self.k:
            raise ValueError(f"shocks have {k} components, model has {self.k}")
        chol = np.linalg.cholesky(self.sigma)
        out = np.empty_like(shocks, dtype=float)
        x = np.broadcast_to(np.asarray(x0, dtype=float), (n_paths, k))
        for t in range(steps):
            x = self.intercept + x @ self.phi.T + shocks[t] @ chol.T
            out[t] = x
        return out


def _check(data: pd.DataFrame) -> FloatArray:
    values = data.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("factor data contain missing values")
    k = values.shape[1]
    if values.shape[0] < 3 * (k + 1):
        raise ValueError(f"too few observations ({values.shape[0]}) for a {k}-variable VAR(1)")
    return values


def fit_var1(data: pd.DataFrame) -> VAR1:
    """Unrestricted VAR(1) with intercept."""
    values = _check(data)
    y, x_lag = values[1:], values[:-1]
    design = np.column_stack([np.ones(len(x_lag)), x_lag])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    coef = np.asarray(solution, dtype=np.float64)
    resid = y - design @ coef
    dof = len(y) - design.shape[1]
    return VAR1(
        intercept=coef[0],
        phi=coef[1:].T,
        sigma=resid.T @ resid / dof,
        names=tuple(map(str, data.columns)),
        nobs=len(y),
    )


def fit_ar1_diagonal(data: pd.DataFrame) -> VAR1:
    """Separate AR(1) per factor (Diebold & Li, 2006); shocks may still be correlated."""
    values = _check(data)
    y, x_lag = values[1:], values[:-1]
    k = values.shape[1]
    intercept, slopes, resid = np.empty(k), np.empty(k), np.empty_like(y)
    for j in range(k):
        design = np.column_stack([np.ones(len(x_lag)), x_lag[:, j]])
        coef, *_ = np.linalg.lstsq(design, y[:, j], rcond=None)
        intercept[j], slopes[j] = coef
        resid[:, j] = y[:, j] - design @ coef
    dof = len(y) - 2
    return VAR1(
        intercept=intercept,
        phi=np.diag(slopes),
        sigma=resid.T @ resid / dof,
        names=tuple(map(str, data.columns)),
        nobs=len(y),
    )
