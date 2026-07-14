"""figure_style.py — shared figure constants and matplotlib helpers."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Figure sizes
# ---------------------------------------------------------------------------
MM_PER_INCH:         float = 25.4
FIG_WIDTH_FULL_IN:   float = 185.0 / MM_PER_INCH   # ≈ 7.28 in
FIG_WIDTH_SINGLE_IN: float = 85.0  / MM_PER_INCH   # ≈ 3.35 in

# Layout margins (inches).  Axes height is the reference.
AX_H_IN: float = 1.85
MAR_L:   float = 0.55
MAR_R:   float = 0.06
MAR_T:   float = 0.25
MAR_B:   float = 0.38
GAP_H:   float = 0.40
GAP_W:   float = 0.38

# Typography
FONT_SIZE_BODY:        int = 6
FONT_SIZE_TICK:        int = 5
FONT_SIZE_PANEL_LABEL: int = 8
PANEL_LABEL_X:         float = -0.09
PANEL_LABEL_Y:         float =  1.04

# Lines, ticks, grids
LINE_WIDTH_DATA:   float = 0.5
SPINE_LINEWIDTH:   float = 0.35
TICK_MAJOR_WIDTH:  float = 0.35
TICK_MAJOR_LENGTH: float = 2.5
GRID_ALPHA:        float = 0.12
GRID_LINEWIDTH:    float = 0.35
LINE_ALPHA:        float = 0.70
MARKER_SIZE:       float = 2.35
MARKER_EDGEWIDTH:  float = 0.35
DPI_PNG:           int   = 1000

# Line styles for the 15 descriptor curves in Fig 1 (cycles every 5).
DESCRIPTOR_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# Short legend labels for the 15 descriptors.
SHORT_LABEL: Dict[str, str] = {
    "Decarbonisation_Outcome":  "Decarb. Outcome",
    "EU_Policy_Alignment":      "EU Policy",
    "Electrification_Pace":     "Electrification",
    "Energy_Security_Pressure": "Energy Security",
    "Fossil_Price_Pressure":    "Fossil Price",
    "Grid_Development":         "Grid Dev.",
    "Hydrogen_Role":            "Hydrogen",
    "Industrial_Energy_Demand": "Ind. Demand",
    "Investment_Availability":  "Investment",
    "Land_Use_Conflict":        "Land Use",
    "Permitting_Pace":          "Permitting",
    "Policy_Stringency":        "Policy String.",
    "Public_Acceptance":        "Public Accept.",
    "Renewables_Deployment":    "Renewables",
    "Technology_Costs":         "Tech. Costs",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_rcparams() -> None:
    """Set global rcParams to match the project style guide."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":          FONT_SIZE_BODY,
        "axes.labelsize":     FONT_SIZE_BODY,
        "xtick.labelsize":    FONT_SIZE_TICK,
        "ytick.labelsize":    FONT_SIZE_TICK,
        "legend.fontsize":    FONT_SIZE_BODY,
        "axes.linewidth":     SPINE_LINEWIDTH,
        "xtick.major.width":  TICK_MAJOR_WIDTH,
        "ytick.major.width":  TICK_MAJOR_WIDTH,
        "xtick.major.size":   TICK_MAJOR_LENGTH,
        "ytick.major.size":   TICK_MAJOR_LENGTH,
    })


def style_ax(ax: plt.Axes, grid: bool = True) -> None:
    """Left + bottom spines only; thin lines; tick sizes from constants."""
    for name, spine in ax.spines.items():
        if name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_linewidth(SPINE_LINEWIDTH)
    ax.tick_params(axis="both", which="major",
                   width=TICK_MAJOR_WIDTH, length=TICK_MAJOR_LENGTH,
                   labelsize=FONT_SIZE_TICK)
    if grid:
        ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)


def panel_label(ax: plt.Axes, letter: str) -> None:
    """Bold panel letter at the standard outside-axes position."""
    ax.text(PANEL_LABEL_X, PANEL_LABEL_Y, letter,
            transform=ax.transAxes,
            fontsize=FONT_SIZE_PANEL_LABEL, fontweight="bold",
            va="bottom", ha="right")


def set_layout(fig: plt.Figure, nrows: int, ncols: int, w_in: float) -> None:
    """Resize fig so every axes is exactly AX_H_IN tall."""
    h    = MAR_T + nrows * AX_H_IN + max(nrows - 1, 0) * GAP_H + MAR_B
    ax_w = (w_in - MAR_L - MAR_R - max(ncols - 1, 0) * GAP_W) / ncols
    fig.set_size_inches(w_in, h)
    fig.subplots_adjust(
        left   = MAR_L / w_in,
        right  = 1 - MAR_R / w_in,
        bottom = MAR_B / h,
        top    = 1 - MAR_T / h,
        hspace = (GAP_H / AX_H_IN) if nrows > 1 else 0,
        wspace = (GAP_W / ax_w)    if ncols > 1 else 0,
    )
