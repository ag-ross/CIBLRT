"""plot_irf.py — CIB-LRT impulse response visualisation.

Produces six figures and three CSV files in workings/outputs/:
  fig1  IRF per-descriptor curves sc05→sc03 with MC ribbon (5th–95th pct).
  fig2  ‖R(τ)‖₂ norm ensemble for all 20 directed scenario pairs.
  fig3  Susceptibility network at sc01 (two panels: t₁ = 1 and t₁ = 5).
  fig4  2×2 sensitivity panel vs target_rho ∈ [0.50, 0.99].
  fig5  Calibration-horizon sweep t₁ ∈ [0.5, 10.0]: budgets and ‖ρ‖_F.
  fig6  MC noise-scale sweep σ ∈ [0.1, 1.0]: budget distribution.

Requires outputs/cib_lrt_results.pkl from run_analysis.py.
"""

from __future__ import annotations

import csv as _csv
import pickle
import sys
import warnings
from pathlib import Path

# scipy.linalg.expm intermediate squaring can overflow for fast-decaying modes;
# the final result is numerically valid — suppress.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"scipy\.linalg\._matfuncs")
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE  = Path(__file__).resolve().parent
_ROOT  = _HERE.parent

if str(_HERE) not in sys.path:
    sys.path.append(str(_HERE))

# Fall back to a local PyCIB clone if `cib` is not importable.
try:
    import cib  # noqa: F401
except ImportError:
    _PYCIB = _ROOT / "PyCIB-main"
    if not _PYCIB.is_dir():
        raise ImportError(
            "PyCIB not found.  Install it from https://github.com/ag-ross/PyCIB "
            "or place the unpacked source at "
            f"{_PYCIB}."
        ) from None
    sys.path.insert(0, str(_PYCIB))

from shock_visualization._onion_layout import onion_pos
from shock_visualization._fig2_drawing import (
    SIDE_LENGTH_BASE, SIDE_LENGTH_MAX, EDGE_ALPHA, EDGE_KEEP_PERCENTILE,
    NODE_EDGECOLOR, NODE_LINEWIDTH,
    data_coord_side_lengths, draw_square_nodes,
    filter_edges_top_percent, draw_band_edges,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------------------------------------------------------------------
# Duck-typed graph for onion_pos (no networkx required)
# ---------------------------------------------------------------------------

class _SimpleGraph:
    """Minimal duck-typed DiGraph satisfying the onion_pos interface."""

    def __init__(self):
        self._node_list: list = []
        self._node_set: set = set()
        self._edges: list = []  # [(u, v, attrs_dict)]

    def add_node(self, n):
        if n not in self._node_set:
            self._node_set.add(n)
            self._node_list.append(n)

    def add_edge(self, u, v, **attrs):
        if u not in self._node_set:
            self.add_node(u)
        if v not in self._node_set:
            self.add_node(v)
        self._edges.append((u, v, attrs))

    def nodes(self):
        return list(self._node_list)

    def edges(self, data=False):
        if data:
            return [(u, v, d) for u, v, d in self._edges]
        return [(u, v) for u, v, _ in self._edges]


# Node colour palette (deterministic; one colour per descriptor)
_NODE_PALETTE_SV = [
    "#0072B2", "#E69F00", "#009E73", "#D62728", "#5D4037",
    "#C9A227", "#A6D854", "#555555", "#CC79A7", "#17BECF",
    "#FF00FF", "#56B4E9", "#B8860B", "#8B008B", "#FF8C00",
]


def _node_colors_sv(nodes) -> dict:
    """Deterministic colour assignment matching the shock_visualization palette."""
    order = sorted(nodes, key=lambda n: (str(n), n))
    return {n: _NODE_PALETTE_SV[i % len(_NODE_PALETTE_SV)] for i, n in enumerate(order)}


from data import build_matrix, STATE_ORDER_BY_DESCRIPTOR
from lrt import (
    build_effective_matrix,
    cib_lrt_rescaled,
    implied_perturbation,
    impulse_response_curves,
    delta_z_from_scenarios,
    susceptibility_matrix,
    io3_rescale_for_stability,
    _stability_report,
    STABILITY_TOLERANCE,
    _expm,
)
from cib.analysis import ScenarioAnalyzer

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
from figure_style import (
    MM_PER_INCH, FIG_WIDTH_FULL_IN, FIG_WIDTH_SINGLE_IN,
    AX_H_IN, MAR_L as _MAR_L, MAR_R as _MAR_R, MAR_T as _MAR_T,
    MAR_B as _MAR_B, GAP_H as _GAP_H, GAP_W as _GAP_W,
    FONT_SIZE_BODY, FONT_SIZE_TICK, FONT_SIZE_PANEL_LABEL,
    PANEL_LABEL_X, PANEL_LABEL_Y,
    LINE_WIDTH_DATA, SPINE_LINEWIDTH,
    TICK_MAJOR_WIDTH, TICK_MAJOR_LENGTH,
    GRID_ALPHA, GRID_LINEWIDTH, DPI_PNG,
    apply_rcparams as _apply_rcparams,
    style_ax        as _style_ax,
    panel_label     as _panel_label,
    set_layout      as _set_layout,
)

# Backwards-compatible alias used elsewhere in this module.
FIG_WIDTH_IN = FIG_WIDTH_FULL_IN

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
OUTPUT_DIR = _HERE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_FILE = OUTPUT_DIR / "cib_lrt_results.pkl"

from config import (
    T1_IRF, TAU_N_TIMESCALES, TAU_MIN, DT, DT_CSV, IO3_TARGET,
    N_MC, MC_SIGMA, MC_RNG_SEED_IRF, MC_RNG_SEED_SIGMA_BASE,
    CI_LO, CI_HI,
    SENSITIVITY_RHOS, MC_SIGMA_SWEEP, MC_SIGMA_N, T1_SWEEP,
)

from figure_style import SHORT_LABEL  # noqa: E402

SC_LABELS = [
    "sc01: High emissions / Land-Use Low",
    "sc02: High emissions / Land-Use Med",
    "sc03: Medium / Permitting Med",
    "sc04: Medium / Permitting Fast",
    "sc05: Net-zero / High policy",
]

# Descriptor colours — one per descriptor (tab20, 15 total)
_CMAP        = matplotlib.colormaps["tab20"]
COLORS       = [_CMAP(i) for i in range(15)]

# Scenario colours and line-styles (5 scenarios)
SCENARIO_COLOURS = [
    "#E7298A",   # sc01
    "#D95F02",   # sc02
    "#7570B3",   # sc03
    "#1B9E77",   # sc04
    "#66A61E",   # sc05
]
SCENARIO_LINESTYLES = [":", "--", (0, (4, 2)), (0, (6, 2)), "-"]
# Per-scenario markers (matching example figure); applied only where linestyles
# would be hard to distinguish (sc04, idx=3, whose dash pattern is similar to sc03).
SCENARIO_MARKERS = ["+", "|", "s", "x", ".", ","]

def _scenario_marker_kw(si: int, n_pts: int, color: str) -> dict:
    """Return marker kwargs for scenario si if it needs one, else {}."""
    if si != 3:   # only sc04 (0-indexed) needs a marker
        return {}
    step = max(1, n_pts // 4)
    return dict(marker=SCENARIO_MARKERS[si], markerfacecolor="none",
                markeredgecolor=color, markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGEWIDTH, markevery=step)

# Directed-pair colours and line-styles (20 pairs)
_PAIR_CMAP   = matplotlib.colormaps["Set1"]
_PAIR_CMAP2  = matplotlib.colormaps["Set2"]
PAIR_COLORS  = [_PAIR_CMAP(i) for i in range(9)] + [_PAIR_CMAP2(i) for i in range(8)]
PAIR_LINESTYLES = ["-", "--", "-.", ":"]
# Markers for pair plots: cycle 1 (idx 0-3) has no markers; subsequent cycles
# get a marker set so repeated linestyles remain distinguishable.
_PAIR_MARKER_CYCLES = [
    ["+", "x", "s", "|"],   # linestyle cycle 2 (idx 4-7)
    [".", "^", "v", "D"],   # linestyle cycle 3 (idx 8-11)
    ["1", "*", "p", "h"],   # linestyle cycle 4 (idx 12-15)
    ["+", "x", "s", "|"],   # linestyle cycle 5 (idx 16-19)
]

def _pair_marker_kw(pair_idx: int, n_pts: int) -> dict:
    """Return marker kwargs for a pair line, or {} for the first linestyle cycle."""
    n_ls = len(PAIR_LINESTYLES)
    if pair_idx < n_ls:
        return {}
    cycle     = (pair_idx - n_ls) // n_ls
    pos       = (pair_idx - n_ls) % n_ls
    mk        = _PAIR_MARKER_CYCLES[cycle % len(_PAIR_MARKER_CYCLES)][pos]
    step      = max(1, n_pts // 4)
    offset    = (pos * step // n_ls) % step
    return dict(marker=mk, markerfacecolor="none", markeredgecolor=None,
                markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGEWIDTH,
                markevery=(offset, step))

from figure_style import (
    DESCRIPTOR_LINESTYLES, MARKER_SIZE, MARKER_EDGEWIDTH, LINE_ALPHA,
)  # noqa: E402

# Markers for the 15 per-descriptor curves in Fig 1 (cycles every 6).
DESCRIPTOR_MARKERS = ["+", "x", "s", "|", ".", "^"]


# ---------------------------------------------------------------------------
# Monte Carlo helper
# ---------------------------------------------------------------------------

def _monte_carlo_irf(
    W_raw:      np.ndarray,
    dz:         np.ndarray,
    rng:        np.random.Generator,
    n_mc:       int   = N_MC,
    sigma:      float = MC_SIGMA,
    t1:         float = T1_IRF,
    tau_max:    float = 50.0,
    dt:         float = DT,
    target_rho: float = IO3_TARGET,
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, int]]:
    """MC IRF realisations: noise → IO-3 rescale → IRF.  Returns (taus, curves, rejections)."""
    N    = W_raw.shape[0]
    taus = np.arange(0.0, tau_max + dt / 2.0, dt)
    T    = len(taus)
    mc_curves: List[np.ndarray] = []
    rejections: Dict[str, int] = {
        "zero_rho":         0,
        "unstable":         0,
        "non_finite_expm":  0,
        "singular_solve":   0,
    }

    for _ in range(n_mc):
        noise = rng.normal(0.0, sigma, (N, N))
        np.fill_diagonal(noise, 0.0)
        W_n = W_raw + noise

        eigs_Wn = np.linalg.eigvals(W_n)
        rho_n   = float(np.max(np.abs(eigs_Wn)))
        if rho_n < 1e-10:
            rejections["zero_rho"] += 1
            continue
        alpha_n  = target_rho / rho_n
        W_sc     = W_n * alpha_n
        eigs_Msc = eigs_Wn * alpha_n - 1.0  # M_sc = αW_n − I
        if not np.all(eigs_Msc.real < -STABILITY_TOLERANCE):
            rejections["unstable"] += 1
            continue
        M_sc = W_sc - np.eye(N)

        Mt = M_sc.T
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            lhs    = _expm(Mt * t1) - np.eye(N)
            exp_dt = _expm(Mt * dt)
        if np.any(~np.isfinite(lhs)) or np.any(~np.isfinite(exp_dt)):
            rejections["non_finite_expm"] += 1
            continue
        try:
            eps_mc = np.linalg.solve(lhs, Mt @ dz)
        except np.linalg.LinAlgError:
            rejections["singular_solve"] += 1
            continue

        curves    = np.zeros((T, N))
        curves[0] = eps_mc
        for k in range(1, T):
            curves[k] = exp_dt @ curves[k - 1]
        mc_curves.append(curves)

    return taus, mc_curves, rejections


def _summarise_rejections(n_mc: int, n_accepted: int, rejections: Dict[str, int]) -> str:
    """Format a one-line acceptance summary plus per-category counts."""
    parts = [f"{n_accepted}/{n_mc} accepted"]
    for cat, count in rejections.items():
        if count > 0:
            parts.append(f"{count} {cat}")
    return " (".join([parts[0], "; ".join(parts[1:]) + ")"]) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------
# Fig 3 helper — susceptibility network (shock_visualization style)
# ---------------------------------------------------------------------------

def _build_susc_graph(rho_mat: np.ndarray, names: List[str]) -> _SimpleGraph:
    """Build a _SimpleGraph from ρ; edge i→k encodes ρ_{ki}."""
    G = _SimpleGraph()
    for name in names:
        G.add_node(name)
    N = len(names)
    for i in range(N):
        for k in range(N):
            if i == k:
                continue
            val = float(rho_mat[k, i])
            if abs(val) > 1e-10:
                G.add_edge(names[i], names[k],
                           abs_weight=abs(val), signed_weight=val)
    return G


def _draw_susc_network_sv(
    ax:               plt.Axes,
    rho_mat:          np.ndarray,
    desc_names:       List[str],
    pos:              dict,
    ring_radii:       List[float],
    node_colors:      dict,
    node_side_lengths: dict,
    *,
    panel_label: str = "a",
    subtitle:    str = "t₁ = 1",
) -> None:
    """Draw the susceptibility network using shock_visualization primitives."""
    N = len(desc_names)

    # --- build full edge list with per-edge signed weights ----------------
    edges_all: List[Tuple[str, str]] = []
    weights_all: List[float] = []
    colors_all: List[str] = []

    for i in range(N):
        for k in range(N):
            if i == k:
                continue
            val = float(rho_mat[k, i])
            if abs(val) < 1e-10:
                continue
            u, v = desc_names[i], desc_names[k]
            if u not in pos or v not in pos:
                continue
            edges_all.append((u, v))
            weights_all.append(abs(val))
            colors_all.append("#c0392b" if val > 0 else "#2980b9")

    # --- percentile filter (preserving per-edge colours) ------------------
    if edges_all:
        packed = sorted(
            zip(edges_all, weights_all, colors_all),
            key=lambda x: (-x[1], str(x[0][0]), str(x[0][1])),
        )
        e_s = [p[0] for p in packed]
        w_s = [p[1] for p in packed]
        c_s = [p[2] for p in packed]

        arr = np.asarray(w_s, dtype=float)
        thresh = float(np.percentile(arr, EDGE_KEEP_PERCENTILE))
        keep = [idx for idx, w in enumerate(w_s) if w >= thresh]
        edges_show  = [e_s[idx] for idx in keep]
        weights_show = [w_s[idx] for idx in keep]
        colors_show  = [c_s[idx] for idx in keep]
    else:
        edges_show, weights_show, colors_show = [], [], []

    # --- edge widths proportional to |ρ| ----------------------------------
    if weights_show:
        max_w = max(weights_show)
        edge_widths = [
            0.8 + 3.0 * (w / max_w) ** 0.7 if max_w > 0 else 0.8
            for w in weights_show
        ]
    else:
        edge_widths = []

    # --- draw -------------------------------------------------------------
    draw_band_edges(
        ax, pos, ring_radii, node_side_lengths,
        edges_show, edge_widths, colors_show,
        edge_color=None, alpha=EDGE_ALPHA,
    )
    draw_square_nodes(ax, pos, node_side_lengths, node_colors)

    ax.set_facecolor("#f8f9fa")
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    lim = (max(ring_radii) + 0.3) if ring_radii else 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")

    # subtitle (bottom centre, inside axes)
    ax.text(0.50, 0.02, subtitle,
            transform=ax.transAxes,
            fontsize=FONT_SIZE_BODY, ha="center", va="bottom")

    # panel letter (top-left, inside axes)
    ax.text(0.10, 0.90, panel_label,
            transform=ax.transAxes,
            fontsize=FONT_SIZE_PANEL_LABEL, fontweight="bold",
            va="top", ha="left")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    _apply_rcparams()

    # ------------------------------------------------------------------
    # Load from cache produced by run_analysis.py
    # ------------------------------------------------------------------
    cache: Optional[dict] = None
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded CIB-LRT results from cache ({CACHE_FILE.name})")

    if cache is not None:
        desc_names = cache["desc_names"]
        N          = len(desc_names)
        stable_by_tag = {tag: v for tag, v in cache["lrt"].items() if v["stable"]}
        stable_data = list(stable_by_tag.values())
        print(f"  {len(stable_data)} stable scenario(s) in cache.")
    else:
        # No cache: fall back to exhaustive attractor search.
        # Run `python run_analysis.py` first for faster repeated use.
        print("Cache not found — running exhaustive attractor search ...")
        matrix     = build_matrix()
        desc_names = list(matrix.descriptors.keys())
        N          = len(desc_names)
        analyzer   = ScenarioAnalyzer(matrix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_exact = analyzer.find_all_consistent_exact()
        unique_sc = result_exact.scenarios
        print(f"  Exhaustive search complete: {len(unique_sc)} consistent scenario(s) found.")
        stable_by_tag: dict = {}
        for sc_idx, sc in enumerate(unique_sc):
            res, alpha = cib_lrt_rescaled(matrix, sc, target_rho=IO3_TARGET)
            if res.stability.is_stable:
                W_r, _ = build_effective_matrix(matrix, sc)
                tau_max = max(res.stability.recommended_tau_max(TAU_N_TIMESCALES), TAU_MIN)
                tag = f"sc{sc_idx + 1:02d}"
                stable_by_tag[tag] = {
                    "scenario": sc.to_dict(),
                    "M":        res.M,
                    "W_raw":    W_r,
                    "alpha":    alpha,
                    "tau_max":  tau_max,
                }
        stable_data = list(stable_by_tag.values())

    n_sc      = len(stable_data)
    tau_maxes = [sd["tau_max"] for sd in stable_data]
    short_labels = [SHORT_LABEL.get(d, d) for d in desc_names]

    # -----------------------------------------------------------------------
    # Fig 1 — per-descriptor IRF with Monte Carlo CI ribbon (sc05 → sc03)
    # -----------------------------------------------------------------------
    print(f"\n[FIG_01] IRF sc05→sc03 single-panel with {N_MC}-sample Monte Carlo CI ribbon ...")

    def _find_scenario(data, **state_requirements) -> dict:
        """Return the first stable_data entry whose scenario matches all state requirements."""
        for entry in data:
            sc = entry["scenario"]
            if all(sc.get(desc) == state for desc, state in state_requirements.items()):
                return entry
        raise ValueError(
            f"No stable scenario found matching {state_requirements!r}. "
            "Check attractor enumeration or update the structural identifiers."
        )

    sd_from = _find_scenario(stable_data, Policy_Stringency="High",   Hydrogen_Role="Major")
    sd_to   = _find_scenario(stable_data, Policy_Stringency="Medium", Permitting_Pace="Medium")
    M_from       = sd_from["M"]
    tau_max_fig1 = sd_from["tau_max"]

    dz = delta_z_from_scenarios(
        desc_names, STATE_ORDER_BY_DESCRIPTOR,
        sd_from["scenario"], sd_to["scenario"]
    )
    eps_det = implied_perturbation(M_from, dz, T1_IRF)
    taus_det, curves_det = impulse_response_curves(M_from, eps_det, tau_max_fig1, DT)

    rng                          = np.random.default_rng(MC_RNG_SEED_IRF)
    taus_mc, mc_list, mc_rejects = _monte_carlo_irf(
        sd_from["W_raw"], dz, rng,
        tau_max = tau_max_fig1,
        dt      = DT_CSV,
    )
    n_accepted = len(mc_list)
    mc_arr  = np.array(mc_list)
    mc_p05  = np.percentile(mc_arr, CI_LO, axis=0)
    mc_p95  = np.percentile(mc_arr, CI_HI, axis=0)
    mc_mean = mc_arr.mean(axis=0)
    print(f"  Monte Carlo: {_summarise_rejections(N_MC, n_accepted, mc_rejects)}")

    # -----------------------------------------------------------------------
    # Fig 1 — single-panel IRF at single-column width (85 mm)
    # All 15 descriptor curves on one axes; MC ribbon for all descriptors.
    # -----------------------------------------------------------------------
    fig1c, ax1c = plt.subplots()
    _step = max(1, len(taus_det) // 4)  # ~4 markers per curve
    # Markers only for linestyle-cycle 2 (j=5–9) and cycle 3 (j=10–14);
    # cycle 1 (j=0–4) uses linestyle alone — matches example figure convention.
    _cycle2_markers = ["+", "x", "s", "|", "."]
    _cycle3_markers = ["^", "v", "D", "1", "*"]
    for j in range(N):
        col   = COLORS[j]
        ls    = DESCRIPTOR_LINESTYLES[j % len(DESCRIPTOR_LINESTYLES)]
        short = SHORT_LABEL.get(desc_names[j], desc_names[j])
        ax1c.fill_between(
            taus_mc, mc_p05[:, j], mc_p95[:, j],
            color=col, alpha=0.12, linewidth=0,
        )
        plot_kw = dict(color=col, linestyle=ls, lw=LINE_WIDTH_DATA,
                       alpha=LINE_ALPHA, label=short)
        if 5 <= j < 10:
            mk = _cycle2_markers[j - 5]
            _offset = ((j - 5) * _step // 5) % _step
            plot_kw.update(marker=mk, markerfacecolor="none",
                           markeredgecolor=col, markersize=MARKER_SIZE,
                           markeredgewidth=MARKER_EDGEWIDTH,
                           markevery=(_offset, _step))
        elif j >= 10:
            mk = _cycle3_markers[j - 10]
            _offset = ((j - 10) * _step // 5) % _step
            plot_kw.update(marker=mk, markerfacecolor="none",
                           markeredgecolor=col, markersize=MARKER_SIZE,
                           markeredgewidth=MARKER_EDGEWIDTH,
                           markevery=(_offset, _step))
        ax1c.plot(taus_det, curves_det[:, j], **plot_kw)
    ax1c.axhline(0, color="0.5", lw=0.3, ls=":")
    ax1c.set_xlabel(r"$\tau$ (dimensionless model time)", fontsize=FONT_SIZE_BODY)
    ax1c.set_ylabel(r"IRF  $R(\tau)$", fontsize=FONT_SIZE_BODY)
    ax1c.legend(
        ncol=2, fontsize=FONT_SIZE_TICK, loc="upper right", frameon=False,
        handlelength=1.2, labelspacing=0.15, columnspacing=0.6, borderpad=0.3,
    )
    _style_ax(ax1c)
    _ylo, _yhi = ax1c.get_ylim()
    ax1c.set_ylim(_ylo, _yhi + 0.5)

    # -----------------------------------------------------------------------
    # Fig 1 — zoom inset: τ ∈ [0, 10], R ∈ [-1.5, -0.5]
    # Positioned at approximately (x=30, y=-1) in data coordinates.
    # -----------------------------------------------------------------------
    # Apply layout before canvas.draw so transforms are correct for inset placement.
    _set_layout(fig1c, 1, 1, FIG_WIDTH_SINGLE_IN)
    fig1c.canvas.draw()

    # Anchor inset lower-left corner at data coordinates (20, -2.0).
    _d2a = ax1c.transData + ax1c.transAxes.inverted()
    ax_xy = _d2a.transform([[20.0, -2.0]])[0]
    anchor_x = float(np.clip(ax_xy[0], 0.15, 0.65))
    anchor_y = float(np.clip(ax_xy[1] + 0.02, 0.02, 0.75))

    axins = inset_axes(
        ax1c,
        width="25.6%",
        height="32%",
        bbox_to_anchor=(anchor_x, anchor_y, 1, 1),
        bbox_transform=ax1c.transAxes,
        loc="lower left",
        borderpad=0.0,
    )

    x_zoom = (0.0, 10.0)
    y_zoom = (-1.5, -0.5)

    # Plot all descriptor curves inside the inset (deterministic; no MC ribbon)
    for j in range(N):
        col = COLORS[j]
        ls  = DESCRIPTOR_LINESTYLES[j % len(DESCRIPTOR_LINESTYLES)]
        ins_kw = dict(color=col, linestyle=ls,
                      lw=LINE_WIDTH_DATA * 0.9, alpha=LINE_ALPHA)
        if 5 <= j < 10:
            mk = _cycle2_markers[j - 5]
            _offset = ((j - 5) * _step // 5) % _step
            ins_kw.update(marker=mk, markerfacecolor="none",
                          markeredgecolor=col, markersize=MARKER_SIZE,
                          markeredgewidth=MARKER_EDGEWIDTH,
                          markevery=(_offset, _step))
        elif j >= 10:
            mk = _cycle3_markers[j - 10]
            _offset = ((j - 10) * _step // 5) % _step
            ins_kw.update(marker=mk, markerfacecolor="none",
                          markeredgecolor=col, markersize=MARKER_SIZE,
                          markeredgewidth=MARKER_EDGEWIDTH,
                          markevery=(_offset, _step))
        axins.plot(taus_det, curves_det[:, j], **ins_kw)
    axins.axhline(0, color="0.5", lw=0.3, ls=":")
    axins.set_xlim(*x_zoom)
    axins.set_ylim(*y_zoom)
    axins.yaxis.tick_right()
    axins.yaxis.set_label_position("right")
    axins.tick_params(
        labelsize=4, which="both",
        length=TICK_MAJOR_LENGTH, pad=1.0, width=TICK_MAJOR_WIDTH,
    )
    for sp in axins.spines.values():
        sp.set_linewidth(0.35)

    # Dashed rectangle on the main axes marking the zoomed region
    ax1c.add_patch(Rectangle(
        (x_zoom[0], y_zoom[0]),
        x_zoom[1] - x_zoom[0],
        y_zoom[1] - y_zoom[0],
        linewidth=0.5, edgecolor="0.35", facecolor="none",
        linestyle="--", zorder=5,
    ))

    # Connection lines from zoom rectangle to inset corners.
    for (xA, yA), (xB, yB) in [
        ((x_zoom[1], y_zoom[0]), (x_zoom[0], y_zoom[0])),  # bottom corner
        ((x_zoom[1], y_zoom[1]), (x_zoom[0], y_zoom[1])),  # top corner
    ]:
        fig1c.add_artist(ConnectionPatch(
            xyA=(xA, yA), coordsA=ax1c.transData,
            xyB=(xB, yB), coordsB=axins.transData,
            color="0.45", lw=0.4, alpha=0.8,
        ))

    out1c = OUTPUT_DIR / "FIG_01.pdf"
    fig1c.savefig(out1c, dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig1c)
    print(f"  Saved → {out1c.relative_to(_ROOT)}")

    # -----------------------------------------------------------------------
    # Fig 2 — ‖R(τ)‖₂ norm ensemble across all directed scenario pairs
    # -----------------------------------------------------------------------
    print("\n[FIG_03] IRF norm ensemble across all directed scenario pairs ...")

    fig2, ax2 = plt.subplots()
    pair_idx   = 0
    irf_records: List[dict] = []

    for i, sd_f in enumerate(stable_data):
        tau_max_i = tau_maxes[i]
        for j, sd_t in enumerate(stable_data):
            if i == j:
                continue
            dz_ij = delta_z_from_scenarios(
                desc_names, STATE_ORDER_BY_DESCRIPTOR,
                sd_f["scenario"], sd_t["scenario"]
            )
            eps_ij        = implied_perturbation(sd_f["M"], dz_ij, T1_IRF)
            taus_ij, c_ij = impulse_response_curves(sd_f["M"], eps_ij, tau_max_i, DT)
            norms_ij      = np.linalg.norm(c_ij, axis=1)
            ham_ij        = int(np.sum(dz_ij != 0))
            budget_ij     = float(np.linalg.norm(eps_ij))

            _col2 = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
            _mkw2 = _pair_marker_kw(pair_idx, len(taus_ij))
            if "markeredgecolor" in _mkw2 and _mkw2["markeredgecolor"] is None:
                _mkw2["markeredgecolor"] = _col2
            ax2.plot(
                taus_ij, norms_ij,
                color = _col2,
                ls    = PAIR_LINESTYLES[(i * n_sc + j) % len(PAIR_LINESTYLES)],
                lw    = LINE_WIDTH_DATA,
                alpha = LINE_ALPHA,
                label = rf"sc{i+1:02d}$\to$sc{j+1:02d}  (H={ham_ij})",
                **_mkw2,
            )

            taus_csv, c_csv = impulse_response_curves(
                sd_f["M"], eps_ij, tau_max_i, DT_CSV
            )
            for ti, tau_val in enumerate(taus_csv):
                row = {
                    "from_sc": f"sc{i+1:02d}", "to_sc":   f"sc{j+1:02d}",
                    "hamming": ham_ij,          "norm_dz": round(float(np.linalg.norm(dz_ij)), 6),
                    "budget":  round(budget_ij, 6), "tau":  round(float(tau_val), 4),
                }
                for k, dn in enumerate(desc_names):
                    row[dn] = round(float(c_csv[ti, k]), 8)
                irf_records.append(row)
            pair_idx += 1

    ax2.axhline(0, color="0.5", lw=0.3)
    ax2.set_xlabel(r"$\tau$ (dimensionless model time)", fontsize=FONT_SIZE_BODY)
    ax2.set_ylabel(r"IRF norm  $\|R(\tau)\|_2$", fontsize=FONT_SIZE_BODY)
    ax2.legend(ncol=2, fontsize=FONT_SIZE_TICK, loc="upper right",
               frameon=False, handlelength=1.0, labelspacing=0.2, columnspacing=0.8)
    _style_ax(ax2)
    _set_layout(fig2, 1, 1, FIG_WIDTH_SINGLE_IN)
    out2 = OUTPUT_DIR / "FIG_03.pdf"
    fig2.savefig(out2, dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved → {out2.relative_to(_ROOT)}")

    # -----------------------------------------------------------------------
    # Fig 3 — susceptibility network graph (sc01), t₁ = 1 and t₁ = 5
    # -----------------------------------------------------------------------
    print("\n[FIG_04] Susceptibility network graphs for sc01 (t₁ = 1 and t₁ = 5) ...")

    M_sc01 = stable_by_tag["sc01"]["M"]
    rho1   = susceptibility_matrix(M_sc01, 1.0)
    rho5   = susceptibility_matrix(M_sc01, 5.0)

    # --- build graphs and shared layout (rho5 is richer — better for layout)
    G_rho5 = _build_susc_graph(rho5, desc_names)
    pos, ring_radii = onion_pos(G_rho5)
    node_colors_fig3 = _node_colors_sv(desc_names)

    # --- node sizes: column sums of |ρ|; shared vmin/vmax across panels ---
    def _col_sums(rho_mat: np.ndarray) -> dict:
        N_d = len(desc_names)
        return {
            desc_names[i]: sum(abs(rho_mat[k, i]) for k in range(N_d) if k != i)
            for i in range(N_d)
        }

    cs1 = _col_sums(rho1)
    cs5 = _col_sums(rho5)
    # Shared scale across both panels so node sizes are directly comparable.
    _all_cs_vals = list(cs1.values()) + list(cs5.values())
    _shared_vmin = min(_all_cs_vals)
    _shared_vmax = max(_all_cs_vals)
    node_sizes1 = data_coord_side_lengths(cs1, SIDE_LENGTH_BASE, SIDE_LENGTH_MAX,
                                          vmin=_shared_vmin, vmax=_shared_vmax)
    node_sizes5 = data_coord_side_lengths(cs5, SIDE_LENGTH_BASE, SIDE_LENGTH_MAX,
                                          vmin=_shared_vmin, vmax=_shared_vmax)

    # --- figure layout ---------------------------------------------------
    # Height is derived from layout constants so each axis is square in
    # display inches, ensuring set_aspect("equal") leaves no blank margins.
    # wspace is tuned empirically to close the inter-panel gap at this scale.
    N_PANELS_3      = 2
    _F3_LEFT        = 0.02
    _F3_RIGHT       = 0.98
    _F3_TOP         = 0.97
    LEGEND_BOTTOM_3 = 0.13
    _F3_WSPACE      = -0.55
    _ax_w_frac = (_F3_RIGHT - _F3_LEFT) / (N_PANELS_3 + (N_PANELS_3 - 1) * _F3_WSPACE)
    _ax_h_frac = _F3_TOP - LEGEND_BOTTOM_3
    _fig3_h_in = (_ax_w_frac * FIG_WIDTH_FULL_IN) / _ax_h_frac * 0.512
    fig3, (ax3a, ax3b) = plt.subplots(
        1, N_PANELS_3,
        figsize=(FIG_WIDTH_FULL_IN, _fig3_h_in),
        facecolor="white",
    )
    plt.subplots_adjust(
        left=_F3_LEFT, right=_F3_RIGHT, top=_F3_TOP,
        bottom=LEGEND_BOTTOM_3, wspace=_F3_WSPACE,
    )

    _draw_susc_network_sv(
        ax3a, rho1, desc_names, pos, ring_radii, node_colors_fig3, node_sizes1,
        panel_label="a", subtitle="",
    )
    _draw_susc_network_sv(
        ax3b, rho5, desc_names, pos, ring_radii, node_colors_fig3, node_sizes5,
        panel_label="b", subtitle="",
    )

    # --- legends: node colours (top row) + edge type (bottom row) ---------
    _legend_kw = dict(
        frameon=False, fontsize=FONT_SIZE_BODY,
        handleheight=0.6, handletextpad=0.6,
        labelspacing=0.35, columnspacing=1.2,
    )
    node_order_fig3 = sorted(desc_names, key=lambda n: (str(n), n))
    node_handles_fig3 = [
        mpatches.Patch(
            facecolor=node_colors_fig3[n],
            edgecolor=NODE_EDGECOLOR, linewidth=NODE_LINEWIDTH,
            label=SHORT_LABEL.get(n, n),
        )
        for n in node_order_fig3
    ]
    leg_nodes_fig3 = fig3.legend(
        handles=node_handles_fig3,
        loc="lower center", bbox_to_anchor=(0.5, 0.09),
        ncol=5, handlelength=0.6, **_legend_kw,
    )
    fig3.legend(
        handles=[
            mlines.Line2D([], [], color="#c0392b", lw=1.0,
                          label=r"Positive susceptibility ($\rho > 0$)"),
            mlines.Line2D([], [], color="#2980b9", lw=1.0,
                          label=r"Negative susceptibility ($\rho < 0$)"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.04),
        ncol=2, handlelength=1.5, **_legend_kw,
    )
    fig3.add_artist(leg_nodes_fig3)

    out3 = OUTPUT_DIR / "FIG_04.pdf"
    fig3.draw_without_rendering()
    renderer3 = fig3.canvas.get_renderer()
    tight3 = fig3.get_tightbbox(renderer3)
    w3, h3 = tight3.x1 - tight3.x0, tight3.y1 - tight3.y0
    top_trim_frac    = 0.06
    right_trim_frac  = 0.07
    left_trim_frac   = 0.05
    bottom_trim_frac = 0.00
    crop3 = Bbox.from_bounds(
        tight3.x0 + w3 * left_trim_frac,
        tight3.y0 + h3 * bottom_trim_frac,
        w3 * (1.0 - left_trim_frac - right_trim_frac),
        h3 * (1.0 - bottom_trim_frac - top_trim_frac),
    )
    fig3.savefig(out3, dpi=DPI_PNG, bbox_inches=crop3, pad_inches=0.01)
    plt.close(fig3)
    print(f"  Saved → {out3.relative_to(_ROOT)}")

    # -----------------------------------------------------------------------
    # Fig 4 — IO-3 target-ρ sensitivity analysis (2 × 2 panel)
    # -----------------------------------------------------------------------
    print("\n[FIG_S1] IO-3 target-ρ sensitivity analysis ...")

    dz_01_02 = delta_z_from_scenarios(
        desc_names, STATE_ORDER_BY_DESCRIPTOR,
        stable_by_tag["sc01"]["scenario"], stable_by_tag["sc02"]["scenario"]
    )
    raw_W_list   = [sd["W_raw"] for sd in stable_data]
    rho_raw_list = [float(np.max(np.abs(np.linalg.eigvals(W_r)))) for W_r in raw_W_list]

    norm_Lambda = np.zeros((len(SENSITIVITY_RHOS), n_sc))
    norm_rho1   = np.zeros((len(SENSITIVITY_RHOS), n_sc))
    timescale   = np.zeros((len(SENSITIVITY_RHOS), n_sc))
    budget_0102 = np.zeros(len(SENSITIVITY_RHOS))

    for ri, rho_target in enumerate(SENSITIVITY_RHOS):
        for si, (W_r, rho_r) in enumerate(zip(raw_W_list, rho_raw_list)):
            alpha_i  = rho_target / rho_r
            W_sc_i   = W_r * alpha_i
            M_sc_i   = W_sc_i - np.eye(N)
            stab_i   = _stability_report(W_sc_i, M_sc_i)
            Lam_i    = np.linalg.solve(M_sc_i.T, -np.eye(N))
            rho1_i   = susceptibility_matrix(M_sc_i, 1.0)
            norm_Lambda[ri, si] = np.linalg.norm(Lam_i, "fro")
            norm_rho1[ri, si]   = np.linalg.norm(rho1_i, "fro")
            timescale[ri, si]   = 1.0 / max(stab_i.slowest_mode, 1e-12)
        W_raw_sc01      = stable_by_tag["sc01"]["W_raw"]
        rho_raw_sc01    = float(np.max(np.abs(np.linalg.eigvals(W_raw_sc01))))
        alpha_ref       = rho_target / rho_raw_sc01
        M_ref           = W_raw_sc01 * alpha_ref - np.eye(N)
        eps_ref         = implied_perturbation(M_ref, dz_01_02, T1_IRF)
        budget_0102[ri] = float(np.linalg.norm(eps_ref))

    rhos_arr = np.array(SENSITIVITY_RHOS)

    fig4, axes4 = plt.subplots(2, 2)
    panel_letters4 = ("a", "b", "c", "d")
    ax_lam, ax_rho1, ax_bud, ax_ts = axes4.flatten()

    for si in range(n_sc):
        _col = SCENARIO_COLOURS[si]
        ax_lam.plot(rhos_arr, norm_Lambda[:, si],
                    color=_col, ls=SCENARIO_LINESTYLES[si],
                    lw=LINE_WIDTH_DATA, alpha=LINE_ALPHA,
                    label=f"sc{si+1:02d}",
                    **_scenario_marker_kw(si, len(rhos_arr), _col))
    ax_lam.axvline(IO3_TARGET, color="0.5", ls="--", lw=0.5)
    ax_lam.set_xlabel(r"Target $\rho(W)$", fontsize=FONT_SIZE_BODY)
    ax_lam.set_ylabel(r"Multiplier magnitude  $\|\Lambda\|_F$", fontsize=FONT_SIZE_BODY)
    ax_lam.legend(fontsize=FONT_SIZE_BODY, frameon=False, handlelength=1.2)
    _panel_label(ax_lam, "a")
    _style_ax(ax_lam)

    for si in range(n_sc):
        _col = SCENARIO_COLOURS[si]
        ax_rho1.plot(rhos_arr, norm_rho1[:, si],
                     color=_col, ls=SCENARIO_LINESTYLES[si],
                     lw=LINE_WIDTH_DATA, alpha=LINE_ALPHA,
                     label=f"sc{si+1:02d}",
                     **_scenario_marker_kw(si, len(rhos_arr), _col))
    ax_rho1.axvline(IO3_TARGET, color="0.5", ls="--", lw=0.5)
    ax_rho1.set_xlabel(r"Target $\rho(W)$", fontsize=FONT_SIZE_BODY)
    ax_rho1.set_ylabel(r"Short-horizon susceptibility  $\|\rho(t_1{=}1)\|_F$", fontsize=FONT_SIZE_BODY)
    ax_rho1.legend(fontsize=FONT_SIZE_BODY, frameon=False, handlelength=1.2)
    _panel_label(ax_rho1, "b")
    _style_ax(ax_rho1)

    ax_bud.plot(rhos_arr, budget_0102,
                color=SCENARIO_COLOURS[0], lw=LINE_WIDTH_DATA, alpha=LINE_ALPHA)
    ax_bud.axvline(IO3_TARGET, color="0.5", ls="--", lw=0.5)
    ax_bud.set_xlabel(r"Target $\rho(W)$", fontsize=FONT_SIZE_BODY)
    ax_bud.set_ylabel(r"Perturbation budget  $\|\tilde{\varepsilon}\|_2$  (sc01$\to$sc02)", fontsize=FONT_SIZE_BODY)
    _panel_label(ax_bud, "c")
    _style_ax(ax_bud)

    for si in range(n_sc):
        _col = SCENARIO_COLOURS[si]
        ax_ts.plot(rhos_arr, timescale[:, si],
                   color=_col, ls=SCENARIO_LINESTYLES[si],
                   lw=LINE_WIDTH_DATA, alpha=LINE_ALPHA,
                   label=f"sc{si+1:02d}",
                   **_scenario_marker_kw(si, len(rhos_arr), _col))
    ax_ts.axvline(IO3_TARGET, color="0.5", ls="--", lw=0.5)
    ax_ts.set_xlabel(r"Target $\rho(W)$", fontsize=FONT_SIZE_BODY)
    ax_ts.set_ylabel(r"Slowest timescale  $1/|\mathrm{Re}(\lambda_{\mathrm{dom}})|$", fontsize=FONT_SIZE_BODY)
    ax_ts.legend(fontsize=FONT_SIZE_BODY, frameon=False, handlelength=1.2)
    _panel_label(ax_ts, "d")
    _style_ax(ax_ts)

    _set_layout(fig4, 2, 2, FIG_WIDTH_FULL_IN)
    out4 = OUTPUT_DIR / "FIG_S1.pdf"
    fig4.savefig(out4, dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved → {out4.relative_to(_ROOT)}")

    # -----------------------------------------------------------------------
    # CSV 1 — IRF curves for all directed pairs
    # -----------------------------------------------------------------------
    out_irf_csv = OUTPUT_DIR / "irf_all_pairs.csv"
    if irf_records:
        fieldnames = (["from_sc", "to_sc", "hamming", "norm_dz", "budget", "tau"]
                      + desc_names)
        with open(out_irf_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(irf_records)
        print(f"\n[CSV] IRF curves for all {n_sc*(n_sc-1)} pairs → "
              f"{out_irf_csv.relative_to(_ROOT)}")

    # -----------------------------------------------------------------------
    # Fig 5 + CSV 2 — calibration-horizon sweep  t₁ ∈ [0.5, 10.0]
    # -----------------------------------------------------------------------
    print(f"\n[FIG_S2] Calibration-horizon sweep  t₁ ∈ [{T1_SWEEP[0]}, {T1_SWEEP[-1]}] "
          f"({len(T1_SWEEP)} points) ...")

    sweep_budgets  = np.zeros((len(T1_SWEEP), n_sc, n_sc))
    sweep_susc_nrm = np.zeros((len(T1_SWEEP), n_sc))

    for ti, t1_val in enumerate(T1_SWEEP):
        for si, sd in enumerate(stable_data):
            rho_t1 = susceptibility_matrix(sd["M"], t1_val)
            sweep_susc_nrm[ti, si] = float(np.linalg.norm(rho_t1, "fro"))
        for i, sd_f in enumerate(stable_data):
            for j, sd_t in enumerate(stable_data):
                if i == j:
                    continue
                dz_ij = delta_z_from_scenarios(
                    desc_names, STATE_ORDER_BY_DESCRIPTOR,
                    sd_f["scenario"], sd_t["scenario"]
                )
                try:
                    eps_ij = implied_perturbation(sd_f["M"], dz_ij, t1_val)
                    sweep_budgets[ti, i, j] = float(np.linalg.norm(eps_ij))
                except np.linalg.LinAlgError:
                    sweep_budgets[ti, i, j] = np.nan

    t1_arr = np.array(T1_SWEEP)

    # Fig 5: calibration-horizon sweep — two-panel (a: budget, b: susceptibility)
    fig5, (ax5a, ax5b) = plt.subplots(1, 2)
    pidx = 0
    for i in range(n_sc):
        for j in range(n_sc):
            if i == j:
                continue
            ham_ij = int(np.sum(delta_z_from_scenarios(
                desc_names, STATE_ORDER_BY_DESCRIPTOR,
                stable_data[i]["scenario"], stable_data[j]["scenario"]
            ) != 0))
            _col5 = PAIR_COLORS[pidx % len(PAIR_COLORS)]
            _mkw5 = _pair_marker_kw(pidx, len(t1_arr))
            if "markeredgecolor" in _mkw5 and _mkw5["markeredgecolor"] is None:
                _mkw5["markeredgecolor"] = _col5
            ax5a.plot(
                t1_arr, sweep_budgets[:, i, j],
                color = _col5,
                ls    = PAIR_LINESTYLES[(i * n_sc + j) % len(PAIR_LINESTYLES)],
                lw    = LINE_WIDTH_DATA,
                alpha = LINE_ALPHA,
                label = rf"sc{i+1:02d}$\to$sc{j+1:02d}  (H={ham_ij})",
                **_mkw5,
            )
            pidx += 1
    ax5a.axvline(T1_IRF, color="0.4", ls="--", lw=0.5,
                 label=rf"nominal $t_1$ = {T1_IRF}")
    ax5a.set_xlabel(r"Calibration horizon  $t_1$ (dimensionless)", fontsize=FONT_SIZE_BODY)
    ax5a.set_ylabel(r"Perturbation budget  $\|\tilde{\varepsilon}(t_1)\|$", fontsize=FONT_SIZE_BODY)
    ax5a.legend(ncol=2, fontsize=FONT_SIZE_TICK, loc="upper right",
                frameon=False, handlelength=1.0, labelspacing=0.2, columnspacing=0.8)
    _panel_label(ax5a, "a")
    _style_ax(ax5a)

    for si in range(n_sc):
        _col = SCENARIO_COLOURS[si]
        ax5b.plot(t1_arr, sweep_susc_nrm[:, si],
                  color=_col, ls=SCENARIO_LINESTYLES[si],
                  lw=LINE_WIDTH_DATA, alpha=LINE_ALPHA,
                  label=f"sc{si+1:02d}",
                  **_scenario_marker_kw(si, len(t1_arr), _col))
    ax5b.axvline(T1_IRF, color="0.4", ls="--", lw=0.5,
                 label=rf"nominal $t_1$ = {T1_IRF}")
    ax5b.set_xlabel(r"Calibration horizon  $t_1$ (dimensionless)", fontsize=FONT_SIZE_BODY)
    ax5b.set_ylabel(r"Susceptibility magnitude  $\|\rho(t_1)\|_F$", fontsize=FONT_SIZE_BODY)
    ax5b.legend(fontsize=FONT_SIZE_BODY, frameon=False, handlelength=1.2)
    _panel_label(ax5b, "b")
    _style_ax(ax5b)

    _set_layout(fig5, 1, 2, FIG_WIDTH_FULL_IN)
    out5 = OUTPUT_DIR / "FIG_S2.pdf"
    fig5.savefig(out5, dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig5)
    print(f"  Saved → {out5.relative_to(_ROOT)}")

    # CSV for t1 sweep
    out_t1_csv = OUTPUT_DIR / "t1_sweep.csv"
    dz_pairs = {}
    for i in range(n_sc):
        for j in range(n_sc):
            if i != j:
                dz_pairs[(i, j)] = delta_z_from_scenarios(
                    desc_names, STATE_ORDER_BY_DESCRIPTOR,
                    stable_data[i]["scenario"], stable_data[j]["scenario"]
                )
    t1_rows: List[dict] = []
    for ti, t1_val in enumerate(T1_SWEEP):
        for i in range(n_sc):
            for j in range(n_sc):
                if i == j:
                    continue
                dz_ij = dz_pairs[(i, j)]
                t1_rows.append({
                    "t1":            round(t1_val, 4),
                    "from_sc":       f"sc{i+1:02d}",
                    "to_sc":         f"sc{j+1:02d}",
                    "hamming":       int(np.sum(dz_ij != 0)),
                    "norm_dz":       round(float(np.linalg.norm(dz_ij)), 6),
                    "budget":        round(float(sweep_budgets[ti, i, j]), 6),
                    "susc_norm_from": round(float(sweep_susc_nrm[ti, i]), 6),
                })
    with open(out_t1_csv, "w", newline="") as f:
        writer = _csv.DictWriter(
            f, fieldnames=["t1","from_sc","to_sc","hamming","norm_dz",
                           "budget","susc_norm_from"]
        )
        writer.writeheader()
        writer.writerows(t1_rows)
    print(f"[CSV] t₁ sweep ({len(T1_SWEEP)} points, {n_sc*(n_sc-1)} pairs) → "
          f"{out_t1_csv.relative_to(_ROOT)}")

    # -----------------------------------------------------------------------
    # Fig 6 + CSV 3 — MC_SIGMA sweep (noise-scale robustness)
    # -----------------------------------------------------------------------
    print(f"\n[FIG_S3] MC_SIGMA sweep  σ ∈ [{MC_SIGMA_SWEEP[0]}, {MC_SIGMA_SWEEP[-1]}] "
          f"({len(MC_SIGMA_SWEEP)} values, {MC_SIGMA_N} samples each) ...")

    sd_sig_from = _find_scenario(stable_data,
                                 Policy_Stringency="High",
                                 Hydrogen_Role="Major")
    sd_sig_to   = _find_scenario(stable_data,
                                 Policy_Stringency="Medium",
                                 Permitting_Pace="Medium")
    dz_sig = delta_z_from_scenarios(
        desc_names, STATE_ORDER_BY_DESCRIPTOR,
        sd_sig_from["scenario"], sd_sig_to["scenario"]
    )
    W_raw_sig = sd_sig_from["W_raw"]

    sigma_rows:    List[dict] = []
    sigma_bud_mean = []
    sigma_bud_p25  = []
    sigma_bud_p75  = []

    N_dim = W_raw_sig.shape[0]
    for sig_idx, sigma_val in enumerate(MC_SIGMA_SWEEP):
        rng = np.random.default_rng(MC_RNG_SEED_SIGMA_BASE + sig_idx)
        budgets_this_sigma: List[float] = []
        rej_zero_rho = rej_unstable = rej_singular = 0
        for _ in range(MC_SIGMA_N):
            noise = rng.normal(0.0, sigma_val, (N_dim, N_dim))
            np.fill_diagonal(noise, 0.0)
            W_n   = W_raw_sig + noise
            rho_n = float(np.max(np.abs(np.linalg.eigvals(W_n))))
            if rho_n < 1e-10:
                rej_zero_rho += 1
                continue
            W_sc = W_n * (IO3_TARGET / rho_n)
            M_sc = W_sc - np.eye(N_dim)
            if np.any(np.linalg.eigvals(M_sc).real >= -STABILITY_TOLERANCE):
                rej_unstable += 1
                continue
            try:
                eps_n = implied_perturbation(M_sc, dz_sig, T1_IRF)
                budgets_this_sigma.append(float(np.linalg.norm(eps_n)))
            except np.linalg.LinAlgError:
                rej_singular += 1
                continue

        bud_arr  = np.array(budgets_this_sigma) if budgets_this_sigma else np.array([np.nan])
        bud_mean = float(np.nanmean(bud_arr))
        bud_p25  = float(np.nanpercentile(bud_arr, 25))
        bud_p75  = float(np.nanpercentile(bud_arr, 75))
        n_acc    = len(budgets_this_sigma)

        sigma_bud_mean.append(bud_mean)
        sigma_bud_p25.append(bud_p25)
        sigma_bud_p75.append(bud_p75)
        sigma_rows.append({
            "sigma":            sigma_val,
            "n_accepted":       n_acc,
            "accept_rate":      round(n_acc / MC_SIGMA_N, 4),
            "rej_zero_rho":     rej_zero_rho,
            "rej_unstable":     rej_unstable,
            "rej_singular":     rej_singular,
            "budget_mean":      round(bud_mean, 6),
            "budget_p25":       round(bud_p25, 6),
            "budget_p75":       round(bud_p75, 6),
            "budget_iqr":       round(bud_p75 - bud_p25, 6),
        })
        if rej_zero_rho + rej_unstable + rej_singular > 0:
            print(f"  σ = {sigma_val}: {n_acc}/{MC_SIGMA_N} accepted "
                  f"({rej_zero_rho} zero_rho; {rej_unstable} unstable; "
                  f"{rej_singular} singular_solve)")

    print(f"  Done ({len(MC_SIGMA_SWEEP)} sigma values).")
    sig_arr      = np.array(MC_SIGMA_SWEEP)
    bud_mean_arr = np.array(sigma_bud_mean)
    bud_p25_arr  = np.array(sigma_bud_p25)
    bud_p75_arr  = np.array(sigma_bud_p75)

    fig6, (ax6a, ax6b) = plt.subplots(1, 2)

    ax6a.fill_between(sig_arr, bud_p25_arr, bud_p75_arr,
                      alpha=0.20, color=SCENARIO_COLOURS[0], label="IQR (25–75%)")
    ax6a.plot(sig_arr, bud_mean_arr, color=SCENARIO_COLOURS[0],
              lw=LINE_WIDTH_DATA, alpha=LINE_ALPHA, label="Mean budget")
    ax6a.axvline(MC_SIGMA, color="0.4", ls="--", lw=0.5,
                 label=rf"nominal $\sigma$ = {MC_SIGMA}")
    ax6a.set_xlabel(r"Noise scale  $\sigma$  (per CIM entry)", fontsize=FONT_SIZE_BODY)
    ax6a.set_ylabel(r"Perturbation budget  $\|\tilde{\varepsilon}\|$", fontsize=FONT_SIZE_BODY)
    ax6a.legend(fontsize=FONT_SIZE_BODY, frameon=False, handlelength=1.2)
    _panel_label(ax6a, "a")
    _style_ax(ax6a)

    rel_unc = (bud_p75_arr - bud_p25_arr) / np.maximum(bud_mean_arr, 1e-12)
    ax6b.plot(sig_arr, rel_unc, color=SCENARIO_COLOURS[1],
              lw=LINE_WIDTH_DATA, alpha=LINE_ALPHA)
    ax6b.axvline(MC_SIGMA, color="0.4", ls="--", lw=0.5,
                 label=rf"nominal $\sigma$ = {MC_SIGMA}")
    ax6b.set_xlabel(r"Noise scale  $\sigma$  (per CIM entry)", fontsize=FONT_SIZE_BODY)
    ax6b.set_ylabel("IQR / Mean", fontsize=FONT_SIZE_BODY)
    ax6b.legend(fontsize=FONT_SIZE_BODY, frameon=False, handlelength=1.2)
    _panel_label(ax6b, "b")
    _style_ax(ax6b)

    _set_layout(fig6, 1, 2, FIG_WIDTH_FULL_IN)
    out6 = OUTPUT_DIR / "FIG_S3.pdf"
    fig6.savefig(out6, dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig6)
    print(f"  Saved → {out6.relative_to(_ROOT)}")

    # CSV for sigma sweep
    out_sig_csv = OUTPUT_DIR / "sigma_sweep.csv"
    with open(out_sig_csv, "w", newline="") as f:
        writer = _csv.DictWriter(
            f, fieldnames=["sigma","n_accepted","accept_rate",
                           "rej_zero_rho","rej_unstable","rej_singular",
                           "budget_mean","budget_p25","budget_p75","budget_iqr"]
        )
        writer.writeheader()
        writer.writerows(sigma_rows)
    print(f"[CSV] σ sweep → {out_sig_csv.relative_to(_ROOT)}")

    print(f"\nAll figures and CSVs written to workings/outputs/")


if __name__ == "__main__":
    main()
