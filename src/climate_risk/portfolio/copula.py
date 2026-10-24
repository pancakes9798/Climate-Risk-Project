"""Gaussian and Student-t copulas for default dependence.

Calibration follows the semi-parametric approach of Demarta & McNeil (2005):

1. pairwise Kendall's τ of issuers' weekly CDS-spread changes. In a structural model the
   spread is a decreasing function of asset value, and Kendall's τ is invariant to strictly
   monotone transformations, so it measures the dependence of the assets themselves;
2. the elliptical-copula identity ρ = sin(π τ / 2), averaged within and between sectors to a
   block (sector) correlation matrix — a parsimonious, positive-definite estimate;
3. the degrees of freedom ν by maximum pseudo-likelihood on rank-based pseudo-observations,
   holding the correlation matrix fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

FloatArray = NDArray[np.float64]


def tau_to_rho(tau: FloatArray) -> FloatArray:
    return np.sin(0.5 * np.pi * np.asarray(tau, dtype=float))


def kendall_tau_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Kendall's τ between the columns of ``data``."""
    values = data.to_numpy(dtype=float)
    d = values.shape[1]
    tau = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau[i, j] = tau[j, i] = stats.kendalltau(values[:, i], values[:, j]).statistic
    return pd.DataFrame(tau, index=data.columns, columns=data.columns)


def block_correlation(rho: pd.DataFrame, groups: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average pairwise correlations within and between groups.

    Returns the full issuer-level matrix and the group-level table of average correlations.
    """
    labels = groups.reindex(rho.index)
    if labels.isna().any():
        raise ValueError("every issuer needs a group label")
    names = list(dict.fromkeys(labels))
    values = rho.to_numpy(dtype=float)
    off_diagonal = ~np.eye(len(labels), dtype=bool)
    blocks = pd.DataFrame(np.nan, index=names, columns=names)
    for g1 in names:
        rows = (labels == g1).to_numpy()
        for g2 in names:
            cols = (labels == g2).to_numpy()
            mask = np.outer(rows, cols) & off_diagonal
            blocks.loc[g1, g2] = values[mask].mean() if mask.any() else np.nan
    code = labels.to_numpy()
    full = np.array(blocks.loc[code, code], dtype=float)  # writable copy
    np.fill_diagonal(full, 1.0)
    if np.linalg.eigvalsh(full).min() <= 0:
        raise ValueError("block correlation matrix is not positive definite")
    return pd.DataFrame(full, index=rho.index, columns=rho.index), blocks


def pseudo_observations(data: pd.DataFrame) -> FloatArray:
    """Rank-based uniforms rank/(n + 1), column by column."""
    ranks = data.rank(axis=0, method="average").to_numpy(dtype=float)
    return ranks / (len(data) + 1.0)


def t_copula_loglik(u: FloatArray, R: FloatArray, nu: float) -> float:
    """Log-likelihood of the Student-t copula with correlation ``R`` and ``nu`` dof."""
    d = R.shape[0]
    x = stats.t.ppf(u, df=nu)
    chol = np.linalg.cholesky(R)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    z = solve_triangular(chol, x.T, lower=True)
    quad = np.sum(z * z, axis=0)
    log_joint = (
        gammaln(0.5 * (nu + d))
        - gammaln(0.5 * nu)
        - 0.5 * d * np.log(nu * np.pi)
        - 0.5 * log_det
        - 0.5 * (nu + d) * np.log1p(quad / nu)
    )
    log_margins = stats.t.logpdf(x, df=nu).sum(axis=1)
    return float(np.sum(log_joint - log_margins))


def fit_nu(u: FloatArray, R: FloatArray, bounds: tuple[float, float] = (2.05, 200.0)) -> float:
    """Maximum pseudo-likelihood degrees of freedom (searched on a log scale)."""
    result = minimize_scalar(
        lambda log_nu: -t_copula_loglik(u, R, float(np.exp(log_nu))),
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(np.exp(result.x))


def _correlated_normals(R: FloatArray, n: int, rng: np.random.Generator) -> FloatArray:
    chol = np.linalg.cholesky(R)
    return rng.standard_normal((n, R.shape[0])) @ chol.T


def simulate_gaussian(R: FloatArray, n: int, rng: np.random.Generator) -> FloatArray:
    return stats.norm.cdf(_correlated_normals(R, n, rng))


def simulate_t(R: FloatArray, nu: float, n: int, rng: np.random.Generator) -> FloatArray:
    z = _correlated_normals(R, n, rng)
    w = rng.chisquare(nu, size=(n, 1))
    return stats.t.cdf(z / np.sqrt(w / nu), df=nu)
