"""Figures for the term-structure stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from climate_risk.figures.style import (
    HAIRLINE,
    MARKER,
    RING,
    Theme,
    dot,
    end_label,
    headline,
    number,
    style_axes,
    subplots,
)
from climate_risk.term_structure import nelson_siegel as ns
from climate_risk.term_structure.projection import overlay, quantile_label

if TYPE_CHECKING:
    from climate_risk.pipeline import Results

NARRATIVE_LABELS = {
    "SSP1 - Baseline": "SSP1 sustainability",
    "SSP2 - Baseline": "SSP2 middle of the road",
    "SSP3 - Baseline": "SSP3 regional rivalry",
    "SSP4 - Baseline": "SSP4 inequality",
    "SSP5 - Baseline": "SSP5 fossil-fuelled",
}
POLICY_COLOR_SLOT = {"SSP2 - 1.9": 0, "SSP2 - 2.6": 1, "SSP2 - 4.5": 2}


def _december(years: pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(year=int(y), month=12, day=31) for y in years])


def lambda_profile(res: Results, theme: Theme) -> Figure:
    profile = res.tables["ns_lambda_profile"]
    lam_hat = float(res.tables["rates_summary"].loc["lambda_per_year", "value"])
    low, high = res.config.term_structure.lambda_grid[:2]
    rmse = res.tables["ns_rmse_by_maturity"]["rmse_bp"]
    fig, (left, right) = subplots(
        1, 2, figsize=(7.6, 3.1), gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.28}
    )
    for ax in (left, right):
        style_axes(ax, theme)
    for start, stop in ((profile.index.min(), low), (high, profile.index.max())):
        if stop > start:
            left.axvspan(start, stop, color=theme.wash, lw=0, zorder=0)
    left.plot(profile.index, profile["pooled_rmse_bp"], color=theme.series[0])
    at_hat = float(np.interp(lam_hat, profile.index, profile["pooled_rmse_bp"]))
    dot(left, theme, lam_hat, at_hat, theme.series[0])
    peak = ns.CURVATURE_ARGMAX / lam_hat
    leader = {"arrowstyle": "-", "color": theme.muted, "lw": HAIRLINE, "shrinkB": 5}
    top = float(profile["pooled_rmse_bp"].max())
    left.annotate(
        f"$\\hat{{\\lambda}}$ = {lam_hat:.3f}\ncurvature peak {peak:.1f}y",
        xy=(lam_hat, at_hat),
        xytext=(lam_hat + 0.12, at_hat + 0.62 * (top - at_hat)),
        fontsize=8,
        color=theme.secondary,
        arrowprops=leader,
    )
    lam_min = float(profile["pooled_rmse_bp"].idxmin())
    rmse_min = float(profile["pooled_rmse_bp"].min())
    dot(left, theme, lam_min, rmse_min, theme.muted)
    left.annotate(
        f"global minimum\n$\\lambda$ = {lam_min:.2f}, hump at {ns.CURVATURE_ARGMAX / lam_min:.0f}y",
        xy=(lam_min, rmse_min),
        xytext=(lam_min - 0.06, rmse_min + 0.33 * (top - rmse_min)),
        fontsize=8,
        color=theme.muted,
        arrowprops=leader,
    )
    left.set_xlabel(r"decay $\lambda$ (per year)")
    left.set_ylabel("pooled RMSE (bp)")
    left.set_title("Profile of the cross-sectional fit")
    positions = np.arange(len(rmse))
    right.bar(positions, rmse.to_numpy(), width=0.55, color=theme.series[0], lw=0)
    right.set_xticks(positions, [f"{m:g}" for m in rmse.index])
    right.set_xlabel("maturity (years)")
    right.set_ylabel(r"RMSE at $\hat{\lambda}$ (bp)")
    right.set_title("Fit error by maturity")
    fig.subplots_adjust(top=0.76)
    headline(
        fig,
        theme,
        "Nelson–Siegel decay: a flat, bimodal likelihood",
        "Shaded: decays outside the search range, which keeps the curvature hump between "
        f"{ns.CURVATURE_ARGMAX / high:.1f}y and {ns.CURVATURE_ARGMAX / low:.1f}y "
        "(Diebold & Li, 2006)",
    )
    return fig


def endpoint_history(res: Results, theme: Theme) -> Figure:
    betas = res.rates.fit.betas
    istar = res.rates.drivers["istar"].reindex(betas.index)
    x = betas.index.to_timestamp()
    cycle = betas["level"] - istar
    fig, (top, bottom) = subplots(
        2,
        1,
        figsize=(7.6, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.0], "hspace": 0.3},
    )
    for ax in (top, bottom):
        style_axes(ax, theme)
    top.plot(x, betas["level"], color=theme.series[0], label="level factor")
    top.plot(x, istar, color=theme.series[1], label="endpoint i* = r* + π*")
    top.legend(loc="upper right")
    top.set_ylabel("percent")
    top.set_title("Level factor and its shifting endpoint")
    bottom.plot(x, cycle, color=theme.series[0])
    mean = float(cycle.mean())
    bottom.axhline(mean, color=theme.muted, lw=HAIRLINE)
    end_label(bottom, theme, x[-1], mean, f"mean {number(mean, '.2f')}")
    bottom.set_ylabel("pp")
    bottom.set_title("Cycle: level minus endpoint (modelled by the VAR)")
    fig.subplots_adjust(top=0.84)
    headline(
        fig,
        theme,
        "The level of yields tracks the natural rate",
        "r*: Holston–Laubach–Williams (one-sided, lagged one quarter); "
        "π*: Cleveland Fed 10-year expected inflation",
    )
    return fig


def backtest(res: Results, theme: Theme) -> Figure:
    rel = res.tables["backtest_relative_rmse"]
    first = pd.Period(res.config.backtest.first_origin, "M").strftime("%b %Y")
    models = ["DNS AR(1)", "DNS VAR(1)", "Shifting-endpoint VAR(1)"]
    horizons = sorted(rel.index.get_level_values("horizon").unique())
    positions = np.arange(len(horizons))
    panels = [m for m in (2.0, 10.0) if m in rel.columns] or list(rel.columns[:2])
    fig, grid = subplots(
        1, len(panels), figsize=(7.6, 3.1), sharey=True, squeeze=False, gridspec_kw={"wspace": 0.12}
    )
    axes = grid[0]
    for ax, maturity in zip(axes, panels, strict=True):
        style_axes(ax, theme)
        ax.axhline(1.0, color=theme.axis, lw=HAIRLINE)
        for slot, model in enumerate(models):
            values = rel.loc[model][maturity].reindex(horizons).to_numpy()
            ax.plot(
                positions,
                values,
                color=theme.series[slot],
                marker="o",
                markersize=MARKER,
                markeredgecolor=theme.surface,
                markeredgewidth=RING,
                label=model,
            )
        ax.set_xticks(positions, [f"{h}m" for h in horizons])
        ax.set_title(f"{maturity:g}-year yield")
        ax.set_xlabel("forecast horizon")
    axes[0].set_ylabel("RMSE relative to random walk")
    axes[0].annotate(
        "random walk",
        xy=(positions[0], 1.0),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color=theme.muted,
    )
    axes[-1].legend(loc="upper left")
    fig.subplots_adjust(top=0.76)
    headline(
        fig,
        theme,
        "Out-of-sample: the shifting endpoint is the best DNS at long horizons",
        f"Expanding-window forecasts from every month since {first}; values below 1 "
        "beat the random walk",
    )
    return fig


def _bands(quantiles: tuple[float, ...]) -> list[tuple[str, str, float]]:
    """Symmetric quantile pairs, widest first: (lower label, upper label, coverage)."""
    return [
        (quantile_label(q), quantile_label(1 - q), 1 - 2 * q)
        for q in quantiles
        if q < 0.5 and any(np.isclose(1 - q, other) for other in quantiles)
    ]


def fan_10y(res: Results, theme: Theme, scenario: str) -> Figure:
    history = res.rates.sample[10.0].loc[pd.Period("2000-01", "M") :]
    proj = res.tables["yield_projection_10y"].loc[scenario]
    median = proj[quantile_label(0.5)]
    x = _december(proj.index)
    color = theme.series[0]
    fig, ax = subplots(figsize=(7.6, 3.4))
    style_axes(ax, theme)
    bands = _bands(res.config.simulation.quantiles)
    for lower, upper, _ in bands:
        ax.fill_between(x, proj[lower], proj[upper], color=color, alpha=0.10, lw=0)
    ax.plot(x, median, color=color)
    ax.plot(history.index.to_timestamp(), history.to_numpy(), color=theme.secondary)
    jump = res.rates.jump_off.month.to_timestamp(how="end")
    ax.axvline(jump, color=theme.axis, lw=HAIRLINE)
    ax.annotate(
        f"jump-off {res.rates.jump_off.month.strftime('%b %Y')}",
        xy=(jump, 0.02),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color=theme.muted,
    )
    end_label(ax, theme, x[-1], float(median.iloc[-1]), f"{number(median.iloc[-1], '.2f')}%")
    handles: list[Line2D | Patch] = [
        Line2D([], [], color=theme.secondary, label="GSW 10y zero yield"),
        Line2D([], [], color=color, label="median projection"),
    ]
    # overlapping fills: the k-th band from the outside is k layers of alpha 0.10 deep
    handles += [
        Patch(facecolor=color, alpha=1 - 0.9 ** (k + 1), lw=0, label=f"{100 * cover:.0f}% band")
        for k, (_, _, cover) in reversed(list(enumerate(bands)))
    ]
    ax.legend(handles=handles, loc="upper right", ncols=2)
    ax.set_ylabel("percent")
    fig.subplots_adjust(top=0.8)
    anchor = (
        "r* held at its jump-off value (reference pathway)"
        if scenario == res.config.scenarios.reference
        else f"r* moves with growth relative to {res.config.scenarios.reference}"
    )
    headline(
        fig,
        theme,
        f"10-year yield to {x[-1].year} · {NARRATIVE_LABELS.get(scenario, scenario)}",
        f"Shifting-endpoint DNS, {res.config.simulation.n_paths:,} Monte Carlo paths; {anchor}",
    )
    return fig


def narratives(res: Results, theme: Theme, names: tuple[str, ...]) -> Figure:
    proj = res.tables["yield_projection_10y"]
    fig, ax = subplots(figsize=(7.6, 3.4))
    style_axes(ax, theme)
    ends = []
    for slot, name in enumerate(names):
        series = proj.loc[name]["mean"]
        x = _december(series.index)
        ax.plot(x, series.to_numpy(), color=theme.series[slot], label=NARRATIVE_LABELS[name])
        ends.append((float(series.iloc[-1]), name, x[-1]))
    ends.sort()
    for i, (value, name, xe) in enumerate(ends):
        gap_ok = all(abs(value - other) > 0.12 for j, (other, _, _) in enumerate(ends) if j != i)
        if gap_ok:
            end_label(ax, theme, xe, value, name.split(" ")[0])
    ax.legend(loc="upper right", ncols=2)
    ax.set_ylabel("expected 10y yield, percent")
    fig.subplots_adjust(top=0.8)
    headline(
        fig,
        theme,
        "Socio-economic pathways move rates far more than climate policy",
        "Mean 10-year zero yield; scenarios differ through world GDP-per-capita growth "
        f"relative to {res.config.scenarios.reference} "
        f"($\\theta$ = {res.config.endpoint.theta:.2f}, HLW)",
    )
    return fig


def policy_channels(res: Results, theme: Theme, baseline: str, policies: tuple[str, ...]) -> Figure:
    factors = res.rates.projection.factors
    paths = res.rates.paths
    base = factors.loc[baseline]
    base_10y = overlay(paths[baseline], [10.0])[:, 0]
    x = base.index.to_timestamp()
    fig, (left, right) = subplots(
        1, 2, figsize=(7.6, 3.1), sharey=True, gridspec_kw={"wspace": 0.1}
    )
    for ax in (left, right):
        style_axes(ax, theme)
        ax.axhline(0.0, color=theme.axis, lw=HAIRLINE)
    for policy in policies:
        gap = factors.loc[policy] - base
        color = theme.series[POLICY_COLOR_SLOT.get(policy, 3)]
        left.plot(x, 100 * (gap["level"] + gap["slope"]), color=color, label=policy)
        gap_10y = overlay(paths[policy], [10.0])[:, 0] - base_10y
        right.plot(x, 100 * gap_10y, color=color, label=policy)
    left.set_title("Expected short rate")
    right.set_title("10-year yield (expectations hypothesis)")
    left.set_ylabel(f"difference vs {baseline} (bp)")
    right.legend(loc="upper right")
    fig.subplots_adjust(top=0.76)
    headline(
        fig,
        theme,
        "Transition policy and the risk-free curve: basis points, not percentage points",
        "Carbon-price pass-through moves short rates while the carbon bill catches up "
        f"($\\kappa$ = {res.config.endpoint.kappa:g}, convergence "
        f"{res.config.scenarios.convergence_year}); growth differences move r*",
    )
    return fig
