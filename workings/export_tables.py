"""export_tables.py — Export tabular results from cib_lrt_results.pkl to CSV.

One CSV per paper table, written to workings/outputs/:
  table_scenarios.csv         Table 1  (tab:scenarios)
  table_rescaling_params.csv  IO-3 rescaling parameters (prose, Sec. 3)
  table_budgets.csv           Table 2  (tab:budgets)
  table_lambda_diag.csv       Table 3  (tab:lambda_diag)

Usage:
    python workings/export_tables.py   (from repository root)
    python export_tables.py            (from workings/)
"""

import csv
import pickle
import pathlib
import numpy as np

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
PKL  = HERE / "outputs" / "cib_lrt_results.pkl"
OUT  = HERE / "outputs"

with open(PKL, "rb") as fh:
    data = pickle.load(fh)

lrt     = data["lrt"]
names   = data["desc_names"]
budgets = data["perturbation_budgets"]

SCENARIOS = ["sc01", "sc02", "sc03", "sc04", "sc05"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def write_csv(path, rows, header=None):
    """Write *rows* (list of lists) to *path*; prepend *header* if given."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if header:
            w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path.relative_to(HERE.parent)}")


# ---------------------------------------------------------------------------
# Table 1 — Scenarios (tab:scenarios)
# Source: data["lrt"][sc]["scenario"]
# ---------------------------------------------------------------------------
print("Exporting table_scenarios.csv …")

header = ["Descriptor"] + SCENARIOS
rows = []
for name in names:
    row = [name] + [lrt[sc]["scenario"][name] for sc in SCENARIOS]
    rows.append(row)

write_csv(OUT / "table_scenarios.csv", rows, header=header)


# ---------------------------------------------------------------------------
# Rescaling parameters (IO-3; supports prose in Sec. 3)
# Source: data["lrt"][sc]["alpha"], "spectral_radius_W_*", "slowest_mode"
# ---------------------------------------------------------------------------
print("Exporting table_rescaling_params.csv …")

header = [
    "scenario",
    "alpha",
    "spectral_radius_W_raw",
    "spectral_radius_W_scaled",
    "stable",
    "slowest_mode_abs_Re_lambda",
    "slowest_timescale_model_units",
]
rows = []
for sc in SCENARIOS:
    s = lrt[sc]
    alpha   = s["alpha"]
    rho_raw = s["spectral_radius_W_raw"]
    rho_sc  = s["spectral_radius_W_scaled"]
    stable  = s["stable"]
    slow    = s["slowest_mode"]           # |Re(lambda_min)|
    ts      = 1.0 / slow if slow > 0 else float("inf")
    rows.append([sc, f"{alpha:.6f}", f"{rho_raw:.4f}", f"{rho_sc:.4f}",
                 stable, f"{slow:.6f}", f"{ts:.4f}"])

write_csv(OUT / "table_rescaling_params.csv", rows, header=header)


# ---------------------------------------------------------------------------
# Table 2 — Perturbation budgets (tab:budgets)
# Source: data["perturbation_budgets"]; rounded to 2 d.p. as in the paper.
# ---------------------------------------------------------------------------
print("Exporting table_budgets.csv …")

# Build lookup for easy access
budget_lookup = {(b["from"], b["to"]): b for b in budgets}

# Paper order: sc01→*, sc02→*, sc03→*, sc04→*, sc05→*  (targets sorted)
TARGETS_BY_ORIGIN = {
    "sc01": ["sc02", "sc03", "sc04", "sc05"],
    "sc02": ["sc01", "sc03", "sc04", "sc05"],
    "sc03": ["sc01", "sc02", "sc04", "sc05"],
    "sc04": ["sc01", "sc02", "sc03", "sc05"],
    "sc05": ["sc01", "sc02", "sc03", "sc04"],
}

header = [
    "from_scenario",
    "to_scenario",
    "hamming_distance",
    "norm_delta_z",
    "perturbation_budget",
    "note",
]
rows = []
for origin in SCENARIOS:
    for target in TARGETS_BY_ORIGIN[origin]:
        b   = budget_lookup[(origin, target)]
        H   = b["hamming"]
        ndz = b["norm_dz"]
        bud = b["budget"]
        # Flag large-displacement entries as in the paper footnote.
        flag = "linearisation extrapolation (H>=12)" if H >= 12 else ""
        rows.append([
            origin, target, H,
            f"{ndz:.2f}",
            f"{bud:.2f}",
            flag,
        ])

write_csv(OUT / "table_budgets.csv", rows, header=header)


# ---------------------------------------------------------------------------
# Table 3 — Lambda diagonal (tab:lambda_diag)
# Source: data["lrt"][sc]["Lambda"][j, j]; rounded to 3 d.p. as in the paper.
# Negative entries indicate self-reversing feedback (see paper Sec. 4).
# ---------------------------------------------------------------------------
print("Exporting table_lambda_diag.csv …")

header = ["descriptor"] + SCENARIOS + ["has_negative_entry"]
rows = []
for j, name in enumerate(names):
    vals = [lrt[sc]["Lambda"][j, j] for sc in SCENARIOS]
    has_neg = any(v < 0 for v in vals)
    rows.append(
        [name] + [f"{v:.3f}" for v in vals] + [has_neg]
    )

write_csv(OUT / "table_lambda_diag.csv", rows, header=header)


print()
print(f"Done.  Four CSV files written to {OUT.relative_to(HERE.parent)}")
print("  table_scenarios.csv        Table 1  (tab:scenarios)")
print("  table_rescaling_params.csv IO-3 rescaling params (Sec. 3 prose)")
print("  table_budgets.csv          Table 2  (tab:budgets)")
print("  table_lambda_diag.csv      Table 3  (tab:lambda_diag)")
