"""Figures for the credit and portfolio stages (sector-level data only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from climate_risk.credit import cds
from climate_risk.credit.transition import SECTORS
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

if TYPE_CHECKING:
    from climate_risk.pipeline import Results

SECTOR_SLOT = {"Utilities": 0, "Oil & gas": 1, "Low-carbon control": 2}
POLICY_SLOT = {"1.9": 0, "2.6": 1, "4.5": 2}
IAM = {"SSP1": "SSP1 · IMAGE", "SSP2": "SSP2 · MESSAGE-GLOBIOM", "SSP5": "SSP5 · REMIND-MAgPIE"}


def _mode_note(res: Results) -> str:
    start, end = (pd.Period(p, "M").year for p in res.config.credit.window)
    if res.credit.mode == "cds":
        return f"CDS panel {start}–{end}, sector medians"
    return "DEMO MODE: public Moody's Baa spread, identical across sectors"


def credit_reference(res: Results, theme: Theme) -> Figure:
    reference = res.credit.reference
    calibration = res.tables["credit_calibration"]
    tenors = cds.TENOR_YEARS
    fig, (left, right) = subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"wspace": 0.25})
    for ax in (left, right):
        style_axes(ax, theme)
    for sector, slot in SECTOR_SLOT.items():
        color = theme.series[slot]
        left.plot(tenors, reference.loc[sector].to_numpy(dtype=float), color=color, label=sector)
        pds = [float(calibration.loc[sector, f"pd_{int(t)}y"]) for t in tenors]
        right.plot(tenors, 100 * np.asarray(pds), color=color, label=sector)
        sigma = float(calibration.loc[sector, "merton_sigma"])
        end_label(right, theme, tenors[-1], 100 * pds[-1], rf"$\sigma_A$ = {sigma:.2f}")
    left.set_title("CDS par spreads")
    left.set_ylabel("basis points")
    right.set_title("Risk-neutral default probability")
    right.set_ylabel("cumulative, percent")
    for ax in (left, right):
        ax.set_xlabel("tenor (years)")
    left.legend(loc="lower right")
    fig.subplots_adjust(top=0.76)
    headline(
        fig,
        theme,
        "Baseline credit curves by sector",
        f"{_mode_note(res)}; bootstrapped with the GSW curve, "
        f"R = {100 * res.config.credit.recovery:.0f}%; "
        r"$\sigma_A$ = Merton asset volatility",
    )
    return fig


def _central(res: Results) -> pd.DataFrame:
    spreads = res.tables["climate_spreads"]
    return spreads[np.isclose(spreads["chi"], res.config.credit.chi)]


def climate_spreads(res: Results, theme: Theme, baseline: str) -> Figure:
    spreads = res.tables["climate_spreads"]
    family = spreads[spreads["baseline"] == baseline]
    chis = sorted(family["chi"].unique())
    fig, axes = subplots(1, 2, figsize=(7.6, 3.2), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, sector in zip(axes, ("Oil & gas", "Utilities"), strict=True):
        style_axes(ax, theme)
        ax.axvspan(10, 30, color=theme.wash, lw=0, zorder=0)
        ax.axhline(0.0, color=theme.axis, lw=HAIRLINE)
        rows = family[family["sector"] == sector]
        for policy in sorted(rows["policy"].unique(), key=lambda p: -float(p.split(" - ")[1])):
            color = theme.series[POLICY_SLOT[policy.split(" - ")[1]]]
            sel = rows[rows["policy"] == policy]
            central = sel[np.isclose(sel["chi"], res.config.credit.chi)].sort_values("tenor")
            if policy.endswith("1.9"):
                low = sel[np.isclose(sel["chi"], chis[0])].sort_values("tenor")
                high = sel[np.isclose(sel["chi"], chis[-1])].sort_values("tenor")
                ax.fill_between(
                    central["tenor"],
                    low["climate_spread_bp"],
                    high["climate_spread_bp"],
                    color=color,
                    alpha=0.10,
                    lw=0,
                )
            ax.plot(central["tenor"], central["climate_spread_bp"], color=color, label=policy)
        ax.set_title(sector)
        ax.set_xlabel("bond maturity (years)")
        ax.annotate(
            "extrapolated",
            xy=(20, 0.98),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=8,
            color=theme.muted,
        )
    axes[0].set_ylabel("climate spread (bp)")
    axes[1].legend(loc="upper left")
    fig.subplots_adjust(top=0.76)
    chi = res.config.credit.chi
    headline(
        fig,
        theme,
        "Climate spread if the transition is priced today",
        f"Battiston–Monasterolo shock vs {baseline}, $\\chi$ = {chi:g} "
        f"(band for 1.9: $\\chi$ = {chis[0]:g}–{chis[-1]:g}); {_mode_note(res)}",
    )
    return fig


def iam_uncertainty(res: Results, theme: Theme, maturity: float = 10.0) -> Figure:
    central = _central(res)
    central = central[np.isclose(central["tenor"], maturity)]
    rows = [(sector, family) for sector in ("Oil & gas", "Utilities") for family in IAM]
    fig, ax = subplots(figsize=(7.6, 3.2))
    style_axes(ax, theme, grid_axis="x")
    ax.axvline(0.0, color=theme.axis, lw=HAIRLINE)
    labels, utility_values = [], []
    for i, (sector, family) in enumerate(rows):
        sel = central[(central["sector"] == sector) & central["policy"].str.startswith(family)]
        values = {
            p.split(" - ")[1]: v
            for p, v in zip(sel["policy"], sel["climate_spread_bp"], strict=True)
        }
        if sector == "Utilities":
            utility_values += [values["2.6"], values["1.9"]]
        y = len(rows) - 1 - i
        ax.plot([values["2.6"], values["1.9"]], [y, y], color=theme.axis, lw=HAIRLINE * 2)
        for key, color in (("2.6", theme.ramp_light), ("1.9", theme.ramp_strong)):
            ax.plot(
                [values[key]],
                [y],
                marker="o",
                markersize=MARKER + 1,
                color=color,
                markeredgecolor=theme.surface,
                markeredgewidth=RING,
                linestyle="none",
                label=f"{key} W/m²" if i == 0 else None,
                zorder=4,
            )
        labels.append(f"{sector} · {IAM[family]}")
    ax.set_yticks(range(len(rows)), labels[::-1])
    ax.set_xlabel(f"{maturity:g}-year climate spread (bp)")
    ax.legend(loc="lower right", title="policy target", title_fontsize=8)
    fig.subplots_adjust(top=0.78, left=0.3)
    headline(
        fig,
        theme,
        "Model uncertainty: the same target, different IAMs",
        f"Each SSP has its own marker IAM; the utilities' {maturity:g}-year climate spread "
        f"ranges from {number(min(utility_values), '+.0f')} to "
        f"{number(max(utility_values), '+.0f')} bp across models",
    )
    return fig


def delayed_revelation(res: Results, theme: Theme, policy: str, maturity: float = 10.0) -> Figure:
    delayed = res.tables["delayed_revelation"]
    sel = delayed[(delayed["policy"] == policy) & np.isclose(delayed["maturity"], maturity)]
    fig, ax = subplots(figsize=(7.6, 3.2))
    style_axes(ax, theme)
    ax.axhline(0.0, color=theme.axis, lw=HAIRLINE)
    for sector in ("Oil & gas", "Utilities"):
        rows = sel[sel["sector"] == sector].sort_values("revelation_year")
        color = theme.series[SECTOR_SLOT[sector]]
        ax.plot(rows["revelation_year"], rows["climate_spread_bp"], color=color, label=sector)
        end_label(
            ax,
            theme,
            float(rows["revelation_year"].iloc[-1]),
            float(rows["climate_spread_bp"].iloc[-1]),
            f"{number(rows['climate_spread_bp'].iloc[-1])} bp",
        )
    ax.set_xlabel("year in which the market prices the transition")
    ax.set_ylabel(f"{maturity:g}-year climate spread (bp)")
    ax.legend(loc="upper left")
    fig.subplots_adjust(top=0.8)
    oil_gas = sel[sel["sector"] == "Oil & gas"].sort_values("revelation_year")
    now, late = oil_gas["climate_spread_bp"].iloc[[0, -1]]
    headline(
        fig,
        theme,
        f"The cost of waiting: oil & gas from {number(now)} to {number(late)} bp",
        f"{policy} under way from the jump-off but priced only in the given year "
        f"($\\chi$ = {res.config.credit.chi:g})",
    )
    return fig


def bond_value(res: Results, theme: Theme, policy: str) -> Figure:
    impacts = res.tables["bond_value_impact"]
    sel = impacts[impacts["policy"] == policy]
    maturities = sorted(sel["maturity"].unique())
    sectors = list(SECTORS)
    fig, axes = subplots(
        1, len(maturities), figsize=(7.6, 3.0), sharey=True, gridspec_kw={"wspace": 0.12}
    )
    height = 0.28
    for ax, maturity in zip(axes, maturities, strict=True):
        style_axes(ax, theme, grid_axis="x")
        ax.axvline(0.0, color=theme.axis, lw=HAIRLINE)
        rows = sel[np.isclose(sel["maturity"], maturity)].set_index("sector").reindex(sectors)
        y = np.arange(len(sectors))[::-1].astype(float)
        ax.barh(
            y + height / 2 + 0.02,
            100 * rows["rate"],
            height=height,
            color=theme.series[0],
            lw=0,
            label="risk-free channel",
        )
        ax.barh(
            y - height / 2 - 0.02,
            100 * rows["credit"],
            height=height,
            color=theme.series[1],
            lw=0,
            label="credit channel",
        )
        for yy, total in zip(y, 100 * rows["total"], strict=True):
            dot(ax, theme, float(total), float(yy), theme.ink)
        ax.set_title(f"{maturity:g}-year bond")
        ax.set_xlabel("value change (%)")
    axes[0].set_yticks(np.arange(len(sectors))[::-1], sectors)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            color=theme.ink,
            linestyle="none",
            markersize=MARKER,
            markeredgecolor=theme.surface,
        )
    )
    labels.append("total")
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(top=0.74, bottom=0.2)
    headline(
        fig,
        theme,
        f"Repricing of zero-coupon bonds when {policy} is revealed",
        "Risk-free channel: expectations-hypothesis repricing of the Treasury curve; "
        "credit channel: climate spread",
    )
    return fig


def portfolio(res: Results, theme: Theme) -> Figure:
    risk = res.tables["portfolio_risk"]
    main = risk["copula"].iloc[0]
    sel = risk[(risk["copula"] == main) & (risk["measure"] == "Physical")].sort_values("es")
    mixture = res.tables["portfolio_scenario_mixture"]
    fig, (left, right) = subplots(
        1, 2, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.45}
    )
    style_axes(left, theme, grid_axis="x")
    style_axes(right, theme)
    y = np.arange(len(sel))
    colors = [theme.muted if s.startswith("Baseline") else theme.series[0] for s in sel["scenario"]]
    left.barh(y, 100 * sel["es"], height=0.5, color=colors, lw=0)
    left.set_yticks(y, [s.replace(" (market-implied)", "") for s in sel["scenario"]])
    for yy, value in zip(y, 100 * sel["es"], strict=True):
        left.annotate(
            f"{value:.1f}",
            xy=(value, yy),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=theme.secondary,
        )
    left.set_xlabel(f"Expected Shortfall {100 * res.config.portfolio.alpha:.0f}% (% of notional)")
    left.set_title("By scenario")
    right.plot(
        100 * mixture["p_adverse"],
        100 * mixture["es"],
        color=theme.series[0],
        label="Expected Shortfall",
    )
    right.plot(
        100 * mixture["p_adverse"],
        100 * mixture["var"],
        color=theme.series[1],
        label="Value-at-Risk",
    )
    right.set_xlabel(f"probability of {mixture['adverse'].iloc[0]} (%)")
    right.set_ylabel("% of notional")
    right.set_title("Scenario uncertainty")
    right.legend(loc="center left")
    fig.subplots_adjust(top=0.76)
    nu = res.portfolio.nu
    copula_note = f"t-copula ($\\nu$ = {nu:.1f}) calibrated on CDS co-movements" if nu else main
    headline(
        fig,
        theme,
        f"Default losses of an equally weighted portfolio over "
        f"{res.config.portfolio.horizon} years",
        f"Physical PDs (risk-neutral intensity / {res.config.portfolio.risk_premium_ratio:g}, "
        f"Driessen 2005); {copula_note}",
    )
    return fig
