"""shock_descriptor.py — unit-impulse shock analysis for a single descriptor.

Applies a unit impulse ε̃ = ±eᵢ at one attractor and traces how descriptor i
responds to a push on each other descriptor via the sensitivity profile
R⁽ⁱ⁾(τ) = exp(Mᵀτ)eᵢ, analogous to Klimek (2019) Fig 1b.  Configuration
(scenario, descriptor, sign) is at the top of the file.

Requires outputs/cib_lrt_results.pkl from run_analysis.py.
Output: outputs/"FIG_02.pdf"
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# scipy.linalg.expm intermediate squaring can overflow for fast-decaying modes;
# the final result is numerically valid — suppress.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"scipy\.linalg\._matfuncs")

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

if str(_HERE) not in sys.path:
    sys.path.append(str(_HERE))

from lrt import impulse_response_curves, _expm

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCENARIO_TAG       = "sc05"           # sc01 ... sc05
SHOCKED_DESCRIPTOR = "Policy_Stringency"
SHOCK_SIGN         = +1               # +1 or -1; any other value is rejected at runtime

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
    LINE_ALPHA, MARKER_SIZE, MARKER_EDGEWIDTH,
    DESCRIPTOR_LINESTYLES, SHORT_LABEL,
)

# Descriptor colours — one per descriptor (tab20, 15 total)
_CMAP  = matplotlib.colormaps["tab20"]
COLORS = [_CMAP(i) for i in range(15)]

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------

from config import (
    TAU_N_TIMESCALES, TAU_MIN, DT, IO3_TARGET,
    N_MC, MC_SIGMA, CI_LO, CI_HI,
    MC_RNG_SEED_UNIT_IMPULSE,
)
from config import DT_CSV as DT_MC

OUTPUT_DIR = _HERE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_FILE = OUTPUT_DIR / "cib_lrt_results.pkl"

_desc_slug  = SHOCKED_DESCRIPTOR.lower().replace("_", "-")
OUTPUT_FILE = OUTPUT_DIR / "FIG_02.pdf"

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

from figure_style import (
    apply_rcparams as _apply_rcparams,
    style_ax       as _style_ax,
    set_layout     as _set_layout,
)


# ---------------------------------------------------------------------------
# Monte Carlo helper for unit-impulse IRF
#
# Identical to _monte_carlo_irf in plot_irf.py except ε̃ = epsilon is held
# fixed across realisations (the unit-impulse vector is not re-derived from
# a scenario pair); only M varies with each noisy CIM draw.
# ---------------------------------------------------------------------------

def _monte_carlo_unit_impulse(
    W_raw:      np.ndarray,
    epsilon:    np.ndarray,
    rng:        np.random.Generator,
    n_mc:       int   = N_MC,
    sigma:      float = MC_SIGMA,
    tau_max:    float = 50.0,
    dt:         float = DT_MC,
    target_rho: float = IO3_TARGET,
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, int]]:
    """MC unit-impulse IRF: noise → IO-3 rescale → R(τ).  Returns (taus, curves, rejections)."""
    N    = W_raw.shape[0]
    taus = np.arange(0.0, tau_max + dt / 2.0, dt)
    T    = len(taus)
    mc_curves: List[np.ndarray] = []
    rejections: Dict[str, int] = {
        "zero_rho":        0,
        "unstable":        0,
        "non_finite_expm": 0,
    }

    from lrt import STABILITY_TOLERANCE

    for _ in range(n_mc):
        noise = rng.normal(0.0, sigma, (N, N))
        np.fill_diagonal(noise, 0.0)
        W_n      = W_raw + noise
        eigs_Wn  = np.linalg.eigvals(W_n)
        rho_n    = float(np.max(np.abs(eigs_Wn)))
        if rho_n < 1e-10:
            rejections["zero_rho"] += 1
            continue
        alpha_n  = target_rho / rho_n
        W_sc     = W_n * alpha_n
        eigs_Msc = eigs_Wn * alpha_n - 1.0     # eigenvalues of αW_n − I
        if not np.all(eigs_Msc.real < -STABILITY_TOLERANCE):
            rejections["unstable"] += 1
            continue
        M_sc = W_sc - np.eye(N)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            exp_dt = _expm(M_sc.T * dt)
        if np.any(~np.isfinite(exp_dt)):
            rejections["non_finite_expm"] += 1
            continue
        c      = np.zeros((T, N))
        c[0]   = epsilon
        for k in range(1, T):
            c[k] = exp_dt @ c[k - 1]
        mc_curves.append(c)

    return taus, mc_curves, rejections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    if SHOCK_SIGN not in (+1, -1):
        raise ValueError(f"SHOCK_SIGN must be +1 or -1; got {SHOCK_SIGN!r}.")

    _apply_rcparams()

    # ------------------------------------------------------------------
    # Load cache produced by run_analysis.py
    # ------------------------------------------------------------------
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache not found at {CACHE_FILE}.\n"
            "Run  python run_analysis.py  first."
        )

    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)

    desc_names: List[str] = cache["desc_names"]
    N = len(desc_names)

    sc_data = cache["lrt"].get(SCENARIO_TAG)
    if sc_data is None:
        raise KeyError(f"Scenario {SCENARIO_TAG!r} not found in cache.")
    if not sc_data["stable"]:
        raise RuntimeError(
            f"Scenario {SCENARIO_TAG!r} is not stable; cannot compute IRF."
        )

    M:       np.ndarray = sc_data["M"]
    W_raw:   np.ndarray = sc_data["W_raw"]
    alpha:   float      = sc_data["alpha"]
    tau_max: float      = max(sc_data["tau_max"], TAU_MIN)

    # ------------------------------------------------------------------
    # Unit-impulse vector  ε̃ = SHOCK_SIGN · eᵢ
    # ------------------------------------------------------------------
    if SHOCKED_DESCRIPTOR not in desc_names:
        raise ValueError(
            f"Descriptor {SHOCKED_DESCRIPTOR!r} not found.\n"
            f"Available: {desc_names}"
        )

    shock_idx          = desc_names.index(SHOCKED_DESCRIPTOR)
    epsilon            = np.zeros(N)
    epsilon[shock_idx] = float(SHOCK_SIGN)

    # ------------------------------------------------------------------
    # Deterministic IRF  R(τ) = exp(Mᵀτ) ε̃
    # ------------------------------------------------------------------
    taus_det, curves_det = impulse_response_curves(M, epsilon, tau_max, DT)

    # ------------------------------------------------------------------
    # Monte Carlo ribbon
    # ------------------------------------------------------------------
    print(f"Running {N_MC}-sample MC ribbon ...")
    rng                            = np.random.default_rng(MC_RNG_SEED_UNIT_IMPULSE)
    taus_mc, mc_list, mc_rejects   = _monte_carlo_unit_impulse(
        W_raw, epsilon, rng, tau_max=tau_max,
    )
    n_accepted = len(mc_list)
    mc_arr = np.array(mc_list)           # shape (n_accepted, T_mc, N)
    mc_p05 = np.percentile(mc_arr, CI_LO, axis=0)
    mc_p95 = np.percentile(mc_arr, CI_HI, axis=0)
    print(f"  {n_accepted}/{N_MC} samples accepted.")

    # ------------------------------------------------------------------
    # Figure — single panel, FIG_WIDTH_SINGLE_IN; style identical to Fig 1
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()

    _step           = max(1, len(taus_det) // 4)   # ~4 markers per curve
    _cycle2_markers = ["+", "x", "s", "|", "."]
    _cycle3_markers = ["^", "v", "D", "1", "*"]

    for j in range(N):
        if j == shock_idx:
            continue   # omit the self-response

        col   = COLORS[j]
        ls    = DESCRIPTOR_LINESTYLES[j % len(DESCRIPTOR_LINESTYLES)]
        short = SHORT_LABEL.get(desc_names[j], desc_names[j])

        ax.fill_between(
            taus_mc, mc_p05[:, j], mc_p95[:, j],
            color=col, alpha=0.12, linewidth=0,
        )
        plot_kw = dict(color=col, linestyle=ls, lw=LINE_WIDTH_DATA,
                       alpha=LINE_ALPHA, label=short)
        if 5 <= j < 10:
            mk      = _cycle2_markers[j - 5]
            _offset = ((j - 5) * _step // 5) % _step
            plot_kw.update(marker=mk, markerfacecolor="none",
                           markeredgecolor=col, markersize=MARKER_SIZE,
                           markeredgewidth=MARKER_EDGEWIDTH,
                           markevery=(_offset, _step))
        elif j >= 10:
            mk      = _cycle3_markers[j - 10]
            _offset = ((j - 10) * _step // 5) % _step
            plot_kw.update(marker=mk, markerfacecolor="none",
                           markeredgecolor=col, markersize=MARKER_SIZE,
                           markeredgewidth=MARKER_EDGEWIDTH,
                           markevery=(_offset, _step))
        ax.plot(taus_det, curves_det[:, j], **plot_kw)

    ax.axhline(0, color="0.5", lw=0.3, ls=":")
    ax.set_xlabel(r"$\tau$ (dimensionless model time)", fontsize=FONT_SIZE_BODY)
    ax.set_ylabel(r"IRF  $R(\tau)$", fontsize=FONT_SIZE_BODY)
    ax.legend(
        ncol=2, fontsize=FONT_SIZE_TICK, loc="upper right", frameon=False,
        handlelength=1.2, labelspacing=0.15, columnspacing=0.6, borderpad=0.3,
    )
    _style_ax(ax)
    _ylo, _ = ax.get_ylim()
    ax.set_ylim(_ylo, 0.15)

    _set_layout(fig, 1, 1, FIG_WIDTH_SINGLE_IN)
    fig.savefig(OUTPUT_FILE, dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUTPUT_FILE.relative_to(_ROOT)}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    shock_sign_str = "+" if SHOCK_SIGN > 0 else "-"
    other_idx      = [i for i in range(N) if i != shock_idx]
    peak_abs       = np.max(np.abs(curves_det[:, other_idx]), axis=0)
    sorted_order   = np.argsort(-peak_abs)

    print(f"\nSensitivity profile of {SHOCKED_DESCRIPTOR} to a {shock_sign_str}1 unit impulse on each other descriptor")
    print(f"Scenario: {SCENARIO_TAG}   IO-3 α = {alpha:.4f}   τ_max = {tau_max:.1f}")
    print(f"\nPushes ranked by peak |R(τ)| of {SHOCKED_DESCRIPTOR} (excluding self):")
    print(f"  {'Rank':<5} {'Push on':<35} {'Peak R(τ)':<12} {'τ at peak':<10}")
    print("  " + "-" * 62)
    for rank, k in enumerate(sorted_order):
        di   = other_idx[k]
        t_pk = float(taus_det[np.argmax(np.abs(curves_det[:, di]))])
        peak = float(curves_det[np.argmax(np.abs(curves_det[:, di])), di])
        print(f"  {rank+1:<5} {desc_names[di]:<35} {peak:<+12.4f} {t_pk:<10.2f}")

    self_peak = float(curves_det[np.argmax(np.abs(curves_det[:, shock_idx])), shock_idx])
    t_self    = float(taus_det[np.argmax(np.abs(curves_det[:, shock_idx]))])
    print(f"\n  Self-response:")
    print(f"  {SHOCKED_DESCRIPTOR:<35} {self_peak:<+12.4f} at τ = {t_self:.2f}")
    rej_parts = [f"{cnt} {cat}" for cat, cnt in mc_rejects.items() if cnt > 0]
    rej_str   = "" if not rej_parts else "  [" + "; ".join(rej_parts) + "]"
    print(f"\n  MC ribbon: {n_accepted}/{N_MC} accepted  "
          f"(σ = {MC_SIGMA}, seed = {MC_RNG_SEED_UNIT_IMPULSE}){rej_str}")


if __name__ == "__main__":
    main()
