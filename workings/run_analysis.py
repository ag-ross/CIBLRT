"""
run_analysis.py — CIB-LRT analysis of the Phase_D_CIM energy-transition dataset.

All results are cached in a single pickle file (outputs/cib_lrt_results.pkl).
On subsequent runs the cache is loaded directly, skipping the PyCIB attractor
search and the LRT computation.  Pass --force to recompute LRT objects from
cached scenarios (parameters can be changed freely).

When ATTRACTOR_SEARCH_COMPLETE = True (set after the final production run),
--force recomputes all LRT objects using the attractor set already stored in
the cache — PyCIB is not invoked.  This locks the attractor set (which was
determined by exhaustive enumeration) whilst still allowing LRT parameters
such as IO3_TARGET, T1_VALUES, and SENSITIVITY_RHOS to be adjusted.

Pipeline: load CSV → exhaustive attractor search → per-scenario LRT objects
(W, M, Σ, Λ, ρ(t₁)) → cross-scenario perturbation budgets → IO-3 sensitivity
analysis → save outputs/cib_lrt_results.pkl.

Usage
-----
  cd workings
  python run_analysis.py           # use cached results if available
  python run_analysis.py --force   # recompute and overwrite cache

Output
------
  outputs/cib_lrt_results.pkl  — all LRT results; read by plot_irf.py

Dependencies
------------
See workings/README.md.  PyCIB, numpy, and scipy are all required.
"""

from __future__ import annotations

import argparse
import pickle
import platform
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
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
            "(see workings/README.md), or place the unpacked source at "
            f"{_PYCIB}."
        ) from None
    sys.path.insert(0, str(_PYCIB))

from data import build_matrix, STATE_ORDER_BY_DESCRIPTOR, scenario_to_indices
from lrt import (
    cib_lrt,
    cib_lrt_rescaled,
    build_effective_matrix,
    CIBLRTResult,
    susceptibility_matrix,
    implied_perturbation,
    perturbation_budget,
    delta_z_from_scenarios,
    io3_rescale_for_stability,
    _stability_report,
)
from cib.analysis import ScenarioAnalyzer
from cib.core import Scenario

# ---------------------------------------------------------------------------
# Output directory and cache file
# ---------------------------------------------------------------------------
OUTPUT_DIR = _HERE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_FILE = OUTPUT_DIR / "cib_lrt_results.pkl"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
# When True, --force recomputes LRT objects from cached W_raw matrices but
# never re-invokes PyCIB.  Set to True after the final production run.
ATTRACTOR_SEARCH_COMPLETE = True   # ← locked after exhaustive run (5 scenarios, 2026-05-09)

from config import (
    T1_VALUES, T1_IRF, DT, TAU_N_TIMESCALES, TAU_MIN, IO3_TARGET,
    SENSITIVITY_RHOS,
)

SEP = "=" * 72


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _save_cache(cache: dict) -> None:
    """Serialise the full results dictionary to a single pickle file."""
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  Saved results -> {CACHE_FILE.relative_to(_ROOT)}")


def _load_cache() -> Optional[dict]:
    """Return the cached results dict, or None if no cache exists."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return None


def _print_cached_summary(cache: dict) -> None:
    """Print a concise summary of previously cached results."""
    desc_names  = cache["desc_names"]
    stable_tags = [tag for tag, d in cache["lrt"].items() if d["stable"]]
    print(f"\nLoaded from cache: {CACHE_FILE.relative_to(_ROOT)}")
    print(f"  Descriptors        : {len(desc_names)}")
    print(f"  Unique scenarios   : {cache['n_all_scenarios']}")
    print(f"  Stable scenarios   : {len(stable_tags)}")
    print(f"\n  Parameters used:")
    for k, v in cache["params"].items():
        print(f"    {k:<22s} = {v}")
    print(f"\n  Perturbation budgets (t1 = {cache['params']['T1_IRF']}):")
    for row in cache["perturbation_budgets"]:
        print(f"    {row['from']}->{row['to']}  "
              f"H={row['hamming']:2d}  "
              f"‖Δz‖={row['norm_dz']:.4f}  "
              f"‖ε̃‖={row['budget']:.4f}")
    locked = "  [ATTRACTOR SET LOCKED — PyCIB will not be re-invoked on --force]" \
             if ATTRACTOR_SEARCH_COMPLETE else ""
    print(f"\n  Rerun with --force to recompute from scratch.{locked}")


# ---------------------------------------------------------------------------
# Console-output helpers
# ---------------------------------------------------------------------------

def _print_matrix_summary(
    name:       str,
    mat:        np.ndarray,
    desc_names: List[str],
    top_k:      int = 5,
) -> None:
    """Print the top-k largest-magnitude off-diagonal entries of a matrix."""
    N       = len(desc_names)
    entries = [
        (abs(mat[i, j]), mat[i, j], desc_names[i], desc_names[j])
        for i in range(N) for j in range(N) if i != j
    ]
    entries.sort(key=lambda x: -x[0])
    print(f"  {name} -- top {top_k} off-diagonal entries by |magnitude|:")
    for _, val, row, col in entries[:top_k]:
        print(f"    [{row[:28]:28s} <- {col[:28]:28s}]  {val:+.4f}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(description="CIB-LRT analysis pipeline")
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore any existing cache and recompute from scratch",
    )
    args = parser.parse_args()

    print(SEP)
    print("CIB-LRT Analysis  --  Phase_D_CIM energy-transition dataset")
    print(SEP)

    # ------------------------------------------------------------------
    # Cache check — skip the full pipeline if results already exist.
    # ------------------------------------------------------------------
    if not args.force:
        cache = _load_cache()
        if cache is not None:
            _print_cached_summary(cache)
            return

    # Record software environment for reproducibility.
    try:
        import scipy as _scipy
        _scipy_ver: Optional[str] = _scipy.__version__
        print(f"scipy           : {_scipy_ver}")
    except ImportError:
        _scipy_ver = None
        print("scipy           : not installed (using _lrt_shim fallback)")

    # Initialise the cache structure; populated incrementally below.
    from lrt import _USING_SCIPY_FALLBACK as _shim_in_use
    cache: dict = {
        "env": {
            "python":               platform.python_version(),
            "numpy":                np.__version__,
            "scipy":                _scipy_ver,
            "using_scipy_fallback": _shim_in_use,
            "platform":             platform.platform(),
        },
        "params": {
            "ATTRACTOR_SEARCH_METHOD": "exhaustive",
            "T1_VALUES":        T1_VALUES,
            "T1_IRF":           T1_IRF,
            "DT":               DT,
            "TAU_N_TIMESCALES": TAU_N_TIMESCALES,
            "TAU_MIN":          TAU_MIN,
            "IO3_TARGET":       IO3_TARGET,
            "SENSITIVITY_RHOS": SENSITIVITY_RHOS,
        },
        "desc_names":         [],
        "n_all_scenarios":    0,
        "lrt":                {},
        "perturbation_budgets": [],
        "sensitivity":        [],
    }

    # ------------------------------------------------------------------
    # 1. Load CIM matrix
    # ------------------------------------------------------------------
    print("\n[1] Loading Phase_D_CIM.csv ...")
    matrix     = build_matrix()
    N          = matrix.n_descriptors
    desc_names = list(matrix.descriptors.keys())
    cache["desc_names"] = desc_names
    print(f"    {N} descriptors x 3 states  ->  {3**N:,} scenario space")

    # ------------------------------------------------------------------
    # 2. Find consistent scenarios
    # ------------------------------------------------------------------
    if ATTRACTOR_SEARCH_COMPLETE and args.force:
        # Locked mode: attractor set was already determined by exhaustive
        # enumeration in a previous run.  Load scenarios from the existing
        # cache so we never re-invoke PyCIB.
        existing = _load_cache()
        if existing is None:
            print(
                "ERROR: ATTRACTOR_SEARCH_COMPLETE=True but no cache found.\n"
                "  Set ATTRACTOR_SEARCH_COMPLETE=False to run a fresh attractor search,\n"
                "  or restore outputs/cib_lrt_results.pkl from version control."
            )
            return
        print("\n[2] ATTRACTOR_SEARCH_COMPLETE -- loading attractor set from cache.")
        # Reconstruct Scenario objects from the cached state dicts.
        analyzer = ScenarioAnalyzer(matrix)
        unique_scenarios: List[Scenario] = []
        for tag in sorted(existing["lrt"].keys()):
            sc_dict = existing["lrt"][tag]["scenario"]
            unique_scenarios.append(Scenario(sc_dict, matrix))
        print(f"    Loaded {len(unique_scenarios)} scenario(s) from cache (PyCIB not invoked).")
    else:
        print("\n[2] Exhaustive attractor enumeration -- find_all_consistent_exact() ...")
        analyzer = ScenarioAnalyzer(matrix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result_exact = analyzer.find_all_consistent_exact()
        unique_scenarios = result_exact.scenarios
        print(f"    Solver status   : {result_exact.status}")
        print(f"    Complete        : {result_exact.is_complete}")
        print(f"    Scenarios found : {len(unique_scenarios)}")
        if hasattr(result_exact, "runtime_s") and result_exact.runtime_s:
            print(f"    Runtime         : {result_exact.runtime_s:.2f} s")
        if not unique_scenarios:
            print("\nNo consistent scenarios found.")
            return

    cache["n_all_scenarios"] = len(unique_scenarios)

    # ------------------------------------------------------------------
    # 3. Per-scenario CIB-LRT analysis
    # ------------------------------------------------------------------
    print(f"\n[3] Per-scenario CIB-LRT analysis ...")

    lrt_results:    List[Tuple[Scenario, CIBLRTResult]] = []
    stable_results: List[Tuple[Scenario, CIBLRTResult]] = []

    for sc_idx, sc in enumerate(unique_scenarios):
        sc_tag = f"sc{sc_idx + 1:02d}"
        print(f"\n  {'-' * 64}")
        print(f"  Scenario {sc_idx + 1} of {len(unique_scenarios)}  ({sc_tag})")
        for d, s in sc.to_dict().items():
            print(f"    {d[:38]:38s}: {s}")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result_raw = cib_lrt(matrix, sc)

        W_raw = result_raw.W   # unscaled; MC procedures rescale per draw

        print(f"\n  Unscaled stability:")
        print(result_raw.stability.summary())

        alpha  = 1.0
        result = result_raw
        if not result_raw.stability.is_stable:
            result, alpha = cib_lrt_rescaled(matrix, sc, target_rho=IO3_TARGET)
            print(
                f"\n  IO-3 rescaling applied:  alpha = {alpha:.6f}  "
                f"(ρ(W) was {result_raw.stability.spectral_radius_W:.4f} -> "
                f"ρ(αW) = {result.stability.spectral_radius_W:.4f})"
            )
            print("  Note: all LRT objects below refer to the rescaled matrix αW.")
            print(f"\n  Scaled stability:")
            print(result.stability.summary())

        # Cache entry populated for all scenarios (stable or not).
        tau_max = 0.0
        if result.stability.is_stable:
            tau_max = max(result.stability.recommended_tau_max(TAU_N_TIMESCALES), TAU_MIN)

        cache["lrt"][sc_tag] = {
            "scenario":                      sc.to_dict(),
            "stable":                        result.stability.is_stable,
            "alpha":                         alpha,
            "spectral_radius_W_raw":         result_raw.stability.spectral_radius_W,
            "spectral_radius_W_scaled":      result.stability.spectral_radius_W,
            "slowest_mode":                  result.stability.slowest_mode,
            "eigenvalues_M":                 np.linalg.eigvals(result.M),
            "W":                             result.W,
            "W_raw":                         W_raw,
            "M":                             result.M,
            "Lambda":                        result.Lambda,
            "Sigma":                         result.Sigma,
            "susceptibility":                {},
            "negative_multiplier_descriptors": result.negative_multiplier_descriptors,
            "tau_max":                       tau_max,
        }

        if not result.stability.is_stable:
            print("  WARNING: stability not achieved after IO-3 rescaling; skipping.")
            lrt_results.append((sc, result))
            continue

        print(f"\n  Adaptive τ_max: {tau_max:.1f}  "
              f"(5 x slowest timescale = 5 x {1.0/result.stability.slowest_mode:.1f})")

        # Negative self-multiplier: flag descriptors where indirect feedback
        # reverses the direct effect of a perturbation.
        if result.has_negative_self_multiplier:
            print(
                f"\n  NOTE: negative self-multiplier Λ[j,j] < 0 "
                f"in {len(result.negative_multiplier_descriptors)} descriptor(s):"
            )
            for d in result.negative_multiplier_descriptors:
                j = desc_names.index(d)
                print(f"    {d[:44]:44s}: Λ[j,j] = {result.Lambda[j, j]:+.4f}")

        _print_matrix_summary("Lambda (cross-impact multiplier)", result.Lambda, desc_names)

        for t1 in T1_VALUES:
            rho = susceptibility_matrix(result.M, t1)
            cache["lrt"][sc_tag]["susceptibility"][t1] = rho

        lrt_results.append((sc, result))
        stable_results.append((sc, result))

    # ------------------------------------------------------------------
    # 4. Cross-scenario perturbation budgets and IRF
    # ------------------------------------------------------------------
    if len(stable_results) < 2:
        print("\n[4] Fewer than two stable scenarios -- skipping cross-scenario analysis.")
        _save_cache(cache)
        return

    print(f"\n[4] Cross-scenario perturbation budgets and IRF "
          f"(t1={T1_IRF}) ...")

    budget_rows: List[dict] = []

    for i, (sc_from, res_from) in enumerate(stable_results):
        for j, (sc_to, _) in enumerate(stable_results):
            if i == j:
                continue

            dz = delta_z_from_scenarios(
                desc_names, STATE_ORDER_BY_DESCRIPTOR,
                sc_from.to_dict(), sc_to.to_dict(),
            )
            if np.all(dz == 0):
                continue  # identical state vectors; should not occur post-dedup

            hamming = int(np.sum(dz != 0))
            budget  = perturbation_budget(res_from.M, dz, T1_IRF)
            budget_rows.append({
                "from":    f"sc{i+1:02d}",
                "to":      f"sc{j+1:02d}",
                "hamming": hamming,
                "norm_dz": float(np.linalg.norm(dz)),
                "budget":  float(budget),
            })

    cache["perturbation_budgets"] = budget_rows
    print(f"\n  Computed {len(budget_rows)} directed-pair perturbation budgets.")

    # ------------------------------------------------------------------
    # 5. IO-3 target-ρ sensitivity analysis
    # ------------------------------------------------------------------
    print("\n[5] IO-3 target-ρ sensitivity analysis ...")
    print(f"    Evaluating target_rho ∈ {SENSITIVITY_RHOS} ...")

    # Look up tags from the cache (stable only) to preserve correct ordering.
    stable_tags_ordered = [
        tag for tag in sorted(cache["lrt"].keys())
        if cache["lrt"][tag]["stable"]
    ]
    raw_W_list = [cache["lrt"][tag]["W_raw"] for tag in stable_tags_ordered]

    dz_01_02 = delta_z_from_scenarios(
        desc_names, STATE_ORDER_BY_DESCRIPTOR,
        stable_results[0][0].to_dict(),
        stable_results[1][0].to_dict(),
    )

    sens_rows: List[dict] = []
    for rho_target in SENSITIVITY_RHOS:
        row: dict = {"target_rho": rho_target}
        for si, W_raw_i in enumerate(raw_W_list):
            W_sc_i, alpha_i = io3_rescale_for_stability(W_raw_i, target_rho=rho_target)
            M_sc_i          = W_sc_i - np.eye(N)
            stab_i          = _stability_report(W_sc_i, M_sc_i)
            Lambda_i        = np.linalg.solve(M_sc_i.T, -np.eye(N))
            rho1_i          = susceptibility_matrix(M_sc_i, 1.0)
            sc_tag_i        = f"sc{si+1:02d}"
            row[f"{sc_tag_i}_alpha"]       = float(alpha_i)
            row[f"{sc_tag_i}_norm_Lambda"] = float(np.linalg.norm(Lambda_i, "fro"))
            row[f"{sc_tag_i}_norm_rho1"]   = float(np.linalg.norm(rho1_i, "fro"))
            row[f"{sc_tag_i}_timescale"]   = float(1.0 / max(stab_i.slowest_mode, 1e-12))

        rho_raw_sc01    = float(np.max(np.abs(np.linalg.eigvals(raw_W_list[0]))))
        W_sc_ref        = raw_W_list[0] * (rho_target / rho_raw_sc01)
        M_sc_ref        = W_sc_ref - np.eye(N)
        row["sc01_to_sc02_budget"] = float(perturbation_budget(M_sc_ref, dz_01_02, T1_IRF))
        sens_rows.append(row)

    cache["sensitivity"] = sens_rows
    print(f"    Computed {len(sens_rows)} sensitivity rows "
          f"({len(stable_results)} stable scenarios each).")

    # ------------------------------------------------------------------
    # 6. Save all results to the cache file
    # ------------------------------------------------------------------
    _save_cache(cache)

    print(f"\n{SEP}")
    print("Analysis complete.  Results cached in outputs/cib_lrt_results.pkl")
    print(SEP)


if __name__ == "__main__":
    main()
