"""Chart theme: one validated categorical palette, recessive chrome, light and dark variants.

The categorical order is fixed and was validated for colour-vision deficiency (adjacent CVD
ΔE ≥ 9 in OKLab×100) on both surfaces; series are never assigned by rank. Text always uses the
ink tokens, never a series colour.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PX = 0.75  # one CSS pixel in points


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    ramp_light: str  # lighter step of the sequential blue (dumbbell "before")
    ramp_strong: str  # stronger step of the sequential blue (dumbbell "after")
    wash: str  # neutral wash for excluded regions

    @property
    def suffix(self) -> str:
        return "" if self.name == "light" else f"-{self.name}"


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    ramp_light="#86b6ef",
    ramp_strong="#1c5cab",
    wash="#f0efec",
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    ramp_light="#256abf",
    ramp_strong="#9ec5f4",
    wash="#262625",
)
THEMES = (LIGHT, DARK)

LINE = 2 * PX
HAIRLINE = 1 * PX
MARKER = 6.0  # ≥ 8 px diameter
RING = 2 * PX


def number(value: float, spec: str = ".0f") -> str:
    """Format a number for a chart label, with a typographic minus sign."""
    return format(value, spec).replace("-", "\u2212")


def subplots(*args: Any, figsize: tuple[float, float], **kwargs: Any) -> tuple[Figure, Any]:
    """A figure created outside pyplot (no global state, no GUI backend) and its axes."""
    fig = Figure(figsize=figsize)
    return fig, fig.subplots(*args, **kwargs)


@contextmanager
def themed(theme: Theme) -> Iterator[None]:
    rc: dict[Any, Any] = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.facecolor": theme.surface,
        "figure.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "text.color": theme.ink,
        "axes.labelcolor": theme.secondary,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelcolor": theme.secondary,
        "ytick.labelcolor": theme.secondary,
        "axes.edgecolor": theme.axis,
        "axes.linewidth": HAIRLINE,
        "lines.linewidth": LINE,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": theme.secondary,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "mathtext.fontset": "dejavusans",
        "axes.titlelocation": "left",
        "axes.titlecolor": theme.ink,
    }
    with mpl.rc_context(rc):
        yield


def style_axes(
    ax: Axes, theme: Theme, *, grid_axis: Literal["x", "y", "both"] | None = "y"
) -> None:
    """Recessive chrome: hairline solid grid, left/bottom baseline only, no tick marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
        ax.spines[side].set_linewidth(HAIRLINE)
    ax.tick_params(length=0, pad=4, labelsize=8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=theme.grid, linewidth=HAIRLINE, linestyle="-")
    ax.set_axisbelow(True)


def headline(fig: Figure, theme: Theme, title: str, subtitle: str) -> None:
    """Left-aligned figure title in primary ink and one line of context in secondary ink."""
    fig.text(
        0.01, 0.995, title, ha="left", va="top", fontsize=11, fontweight="bold", color=theme.ink
    )
    fig.text(0.01, 0.94, subtitle, ha="left", va="top", fontsize=8.5, color=theme.secondary)


def end_label(ax: Axes, theme: Theme, x: float, y: float, text: str) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8,
        color=theme.secondary,
    )


def dot(ax: Axes, theme: Theme, x: float, y: float, color: str) -> None:
    ax.plot(
        [x],
        [y],
        marker="o",
        markersize=MARKER,
        color=color,
        markeredgecolor=theme.surface,
        markeredgewidth=RING,
        zorder=5,
    )


def save(fig: Figure, directory: Path, name: str, theme: Theme) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}{theme.suffix}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    return path
