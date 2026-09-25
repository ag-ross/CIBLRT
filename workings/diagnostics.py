"""
diagnostics.py -- paper diagnostics not produced by the other stages.

Reads outputs/cib_lrt_results.pkl and writes outputs/table_diagnostics.csv,
one row (name, scenario, descriptor, value) per quantity:

  forward_budget,                       forward implied-perturbation norm ‖ε‖ for
  budget_forward_backward_max_rel_gap   every directed stable pair, and the largest
                                        relative gap to the backward budget ‖ε̃‖
  min_energy_rms_backward / _forward    minimum-energy RMS forcing (Van Loan Gramian)
                                        for IRF_PAIR in both forms
  self_response_first_negative_tau,     for every Λ_jj < 0: first lag at which the
  self_response_cumulative_zero_tau     self-response exp(Mτ)_jj turns negative, and
                                        at which its cumulative ρ(τ)_jj crosses zero
  lambda_diag_sign_crossover_rho,       ρ_target at which a Λ_jj changes sign, and
  lambda_diag_ranking_min_spearman,     the invariance of the Λ_jj ranking and of
  lambda_diag_smallest_entry_invariant  its smallest entry over SENSITIVITY_RHOS
  profile_first_order_term,             first-order term W[i,k]/e of the sensitivity
  profile_peak_to_first_order_ratio     profile of SHOCKED_DESCRIPTOR (i) at
                                        SCENARIO_TAG, and the curve peak / term
  most_exposed_at_t1=...                descriptor with the largest off-diagonal column
                                        sum of |ρ(t₁)| at sc01, for each t₁ in T1_SWEEP
  polar_pair_budget,                    the N_POLAR largest budgets at IO3_TARGET, and
  polar_pairs_largest_at_every_rho      whether the same pairs lead at every ρ_target
  two_hop_second_leg_at_origin          budget of the second leg C->B of each polar
                                        two-hop path, evaluated at the origin anchor

and two Monte Carlo tables, from the same draws as FIG_01 and FIG_02:

  mc_sign_robustness.csv   fraction of the FIG_01 realisations (IRF_PAIR) whose
                           curve changes sign, per descriptor
  mc_peak_band.csv         5-95 % band of each other descriptor's FIG_02 curve at
                           its deterministic peak lag

Usage
-----
  cd workings
  python diagnostics.py
"""

from __future__ import annotations

import csv
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List

# scipy.linalg.expm intermediate squaring can overflow for fast-decaying modes;
# the final result is numerically valid -- suppress.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"scipy\.linalg\._matfuncs")

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

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

from config import (
    T1_IRF, T1_SWEEP, SENSITIVITY_RHOS, IO3_TARGET, DT, DT_CSV,
    CI_LO, CI_HI, MC_RNG_SEED_IRF, MC_RNG_SEED_UNIT_IMPULSE,
)
from data import STATE_ORDER_BY_DESCRIPTOR
from lrt import (
    susceptibility_matrix, perturbation_budget, delta_z_from_scenarios,
    io3_rescale_for_stability, impulse_response_curves,
)
from plot_irf import _monte_carlo_irf
from shock_descriptor import (
    SCENARIO_TAG, SHOCKED_DESCRIPTOR, SHOCK_SIGN, _monte_carlo_unit_impulse,
)

OUTPUT_DIR = _HERE / "outputs"
CACHE_FILE = OUTPUT_DIR / "cib_lrt_results.pkl"
OUT_FILE   = OUTPUT_DIR / "table_diagnostics.csv"
SIGN_FILE  = OUTPUT_DIR / "mc_sign_robustness.csv"
BAND_FILE  = OUTPUT_DIR / "mc_peak_band.csv"

IRF_PAIR = ("sc05", "sc03")   # the FIG_01 pair
N_POLAR  = 4
TWO_HOP_ANCHORS = ("sc03", "sc04")


def forward_budget(M: np.ndarray, dz: np.ndarray, t1: float) -> float:
    """‖ε‖₂ of the forward implied perturbation, (M^-1 (exp(M t1) - I))^-1 Δz."""
    N = M.shape[0]
    return float(np.linalg.norm(np.linalg.solve(expm(M * t1) - np.eye(N), M @ dz)))


def gramian(A: np.ndarray, t: float) -> np.ndarray:
    """∫_0^t exp(A^T s) exp(A s) ds by Van Loan's block exponential."""
    n = A.shape[0]
    F = expm(np.block([[-A.T, np.eye(n)], [np.zeros((n, n)), A]]) * t)
    return F[n:, n:].T @ F[:n, n:]


def min_energy_rms(G: np.ndarray, dz: np.ndarray, t1: float) -> float:
    """Minimum-energy RMS forcing sqrt(Δz^T G^-1 Δz / t1)."""
    return float(np.sqrt(dz @ np.linalg.solve(G, dz) / t1))


def lambda_diag(W_raw: np.ndarray, rho: float) -> np.ndarray:
    """Diagonal of Λ after rescaling W_raw to spectral radius rho."""
    W, _ = io3_rescale_for_stability(W_raw, rho)
    M = W - np.eye(W.shape[0])
    return np.diag(np.linalg.solve(M.T, -np.eye(W.shape[0])))


def _write_csv(path: Path, header: List[str], rows: List[list], label: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[CSV] {label} -> {path.relative_to(_ROOT)}")


def main() -> None:
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    names: List[str] = cache["desc_names"]
    N     = len(names)
    lrt   = cache["lrt"]
    tags  = [t for t in sorted(lrt) if lrt[t]["stable"]]
    rows: List[list] = []

    def add(name: str, value, scenario: str = "", descriptor: str = "") -> None:
        rows.append([name, scenario, descriptor, f"{value:.6f}" if isinstance(value, float) else value])

    def dz(a: str, b: str) -> np.ndarray:
        return delta_z_from_scenarios(names, STATE_ORDER_BY_DESCRIPTOR, lrt[a]["scenario"], lrt[b]["scenario"])

    gaps: List[float] = []
    for a in tags:
        for b in tags:
            if a == b:
                continue
            fwd = forward_budget(lrt[a]["M"], dz(a, b), T1_IRF)
            bwd = perturbation_budget(lrt[a]["M"], dz(a, b), T1_IRF)
            gaps.append(abs(fwd - bwd) / bwd)
            add("forward_budget", fwd, f"{a}->{b}")
    add("budget_forward_backward_max_rel_gap", max(gaps))

    a, b   = IRF_PAIR
    dz_ab  = dz(a, b)
    add("min_energy_rms_backward", min_energy_rms(gramian(lrt[a]["M"], T1_IRF), dz_ab, T1_IRF), f"{a}->{b}")
    add("min_energy_rms_forward",  min_energy_rms(gramian(lrt[a]["M"].T, T1_IRF), dz_ab, T1_IRF), f"{a}->{b}")

    for t in tags:
        M, Lam = lrt[t]["M"], lrt[t]["Lambda"]
        taus = np.arange(0.0, lrt[t]["tau_max"] + DT / 2.0, DT)
        for j in range(N):
            if Lam[j, j] >= 0:
                continue
            self_resp = np.array([expm(M * tau)[j, j] for tau in taus])
            cumul     = np.array([susceptibility_matrix(M, tau)[j, j] if tau > 0 else 0.0 for tau in taus])
            neg       = np.flatnonzero(self_resp < 0)
            cross     = np.flatnonzero((cumul[1:] < 0) & (cumul[:-1] >= 0))
            add("self_response_first_negative_tau", float(taus[neg[0]]) if len(neg) else "none", t, names[j])
            add("self_response_cumulative_zero_tau", float(taus[cross[0] + 1]) if len(cross) else "none", t, names[j])

    min_spearman  = 1.0
    argmin_invariant = True
    for t in tags:
        W_raw = lrt[t]["W_raw"]
        grid  = np.array([lambda_diag(W_raw, rho) for rho in SENSITIVITY_RHOS])
        ref   = lambda_diag(W_raw, IO3_TARGET)
        for j in range(N):
            for k in np.flatnonzero(np.sign(grid[1:, j]) != np.sign(grid[:-1, j])):
                root = brentq(lambda rho: lambda_diag(W_raw, rho)[j], SENSITIVITY_RHOS[k], SENSITIVITY_RHOS[k + 1])
                add("lambda_diag_sign_crossover_rho", float(root), t, names[j])
        for row in grid:
            min_spearman     = min(min_spearman, float(spearmanr(row, ref).correlation))
            argmin_invariant = argmin_invariant and int(np.argmin(row)) == int(np.argmin(ref))
    add("lambda_diag_ranking_min_spearman", min_spearman)
    add("lambda_diag_smallest_entry_invariant", argmin_invariant)

    sc   = lrt[SCENARIO_TAG]
    W    = sc["W"]
    i    = names.index(SHOCKED_DESCRIPTOR)
    taus = np.arange(0.0, sc["tau_max"] + DT / 2.0, DT)
    R    = np.array([expm(sc["M"].T * tau)[:, i] for tau in taus])
    for k in range(N):
        if k == i:
            continue
        first_order = W[i, k] / np.e
        peak        = R[np.argmax(np.abs(R[:, k])), k]
        add("profile_first_order_term", float(first_order), SCENARIO_TAG, names[k])
        add("profile_peak_to_first_order_ratio", float(peak / first_order) if first_order != 0 else "none", SCENARIO_TAG, names[k])

    for t1 in T1_SWEEP:
        rho = susceptibility_matrix(lrt["sc01"]["M"], t1)
        exposure = np.abs(rho).sum(axis=0) - np.abs(np.diag(rho))
        add(f"most_exposed_at_t1={t1:g}", float(exposure.max()), "sc01", names[int(np.argmax(exposure))])

    def budgets_at(rho: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for a in tags:
            W_a, _ = io3_rescale_for_stability(lrt[a]["W_raw"], rho)
            M_a = W_a - np.eye(N)
            for b in tags:
                if a != b:
                    out[f"{a}->{b}"] = perturbation_budget(M_a, dz(a, b), T1_IRF)
        return out

    ref_budgets = budgets_at(IO3_TARGET)
    polar = sorted(ref_budgets, key=ref_budgets.get, reverse=True)[:N_POLAR]
    for pair in polar:
        add("polar_pair_budget", ref_budgets[pair], pair)
    add("polar_pairs_largest_at_every_rho", all(
        set(sorted(bg, key=bg.get, reverse=True)[:N_POLAR]) == set(polar)
        for bg in (budgets_at(rho) for rho in SENSITIVITY_RHOS)
    ))
    for pair in polar:
        o, d = pair.split("->")
        for c in TWO_HOP_ANCHORS:
            add("two_hop_second_leg_at_origin", perturbation_budget(lrt[o]["M"], dz(c, d), T1_IRF), f"{pair} via {c}")

    _write_csv(OUT_FILE, ["name", "scenario", "descriptor", "value"], rows, f"{len(rows)} diagnostics")
    for row in rows:
        if not row[0].startswith(("forward_budget", "profile_first_order_term")):
            print("  " + "  ".join(str(x) for x in row if x != ""))

    rng = np.random.default_rng(MC_RNG_SEED_IRF)
    _, mc_list, _ = _monte_carlo_irf(lrt[a]["W_raw"], dz_ab, rng, tau_max=lrt[a]["tau_max"], dt=DT_CSV)
    sign_change = np.mean([np.any(c[1:] * c[:-1] < 0, axis=0) for c in mc_list], axis=0)
    _write_csv(SIGN_FILE, ["descriptor", "fraction_sign_change"],
               [[d, round(float(f), 6)] for d, f in zip(names, sign_change)],
               f"fraction of FIG_01 realisations changing sign ({len(mc_list)} accepted)")

    epsilon    = np.zeros(N)
    epsilon[i] = float(SHOCK_SIGN)
    taus_det, curves_det = impulse_response_curves(sc["M"], epsilon, sc["tau_max"], DT)
    rng = np.random.default_rng(MC_RNG_SEED_UNIT_IMPULSE)
    taus_mc, mc_list, _ = _monte_carlo_unit_impulse(sc["W_raw"], epsilon, rng, tau_max=sc["tau_max"])
    mc_arr    = np.array(mc_list)
    mc_p05    = np.percentile(mc_arr, CI_LO, axis=0)
    mc_p95    = np.percentile(mc_arr, CI_HI, axis=0)
    other_idx = [k for k in range(N) if k != i]
    band_rows: List[list] = []
    for k in np.argsort(-np.max(np.abs(curves_det[:, other_idx]), axis=0)):
        di   = other_idx[k]
        t_pk = float(taus_det[int(np.argmax(np.abs(curves_det[:, di])))])
        k_mc = int(np.argmin(np.abs(taus_mc - t_pk)))
        band_rows.append([names[di], round(t_pk, 4),
                          round(float(mc_p05[k_mc, di]), 6), round(float(mc_p95[k_mc, di]), 6)])
    _write_csv(BAND_FILE, ["descriptor", "peak_tau", "band_lo", "band_hi"], band_rows,
               f"5-95 % band at the deterministic peak lag ({len(mc_list)} accepted)")


if __name__ == "__main__":
    main()
