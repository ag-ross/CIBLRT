"""
CIB-LRT Practitioner Example: Unit-Impulse Shock Analysis
==========================================================

WHO IS THIS FOR
---------------
CIB practitioners who have completed a Cross-Impact Balance study and want to
go beyond the standard list of consistent scenarios.  No knowledge of the
underlying mathematics is needed to run this example or interpret its output.

WHAT THIS EXAMPLE DOES
-----------------------
It takes a hardcoded Cross-Impact Matrix (CIM) (an energy-transition study
with 10 descriptors and 3 states each) finds the consistent scenarios
(structural equilibria), then asks a simple question at the high-ambition
attractor (high policy stringency, fast renewables, low fossil prices).
The setting FORM selects one of two questions:

    FORM = "backward" (the default, and the form reported in the paper):
    "If each other descriptor in turn receives a single unit push, which
     pushes move Policy Stringency most strongly, in which direction, and in
     what sequence?"

    FORM = "forward":
    "If Policy Stringency receives a single unit push, which other
     descriptors does it move most strongly, in which direction, and in
     what sequence?"

The answer is the Impulse Response Function (IRF): a curve for each other
descriptor.  In the backward form (the sensitivity profile) the curve shows
the activation pressure on Policy Stringency over time after a unit push on
that descriptor.  In the forward form (the forward profile) it shows the
activation pressure on that descriptor over time after a unit push on Policy
Stringency.

HOW TO USE AS A TEMPLATE
-------------------------
To adapt this to your own CIB study, replace the DESCRIPTORS and IMPACTS
dictionaries with your own CIM data.  Then change SHOCK_DESCRIPTOR to
whichever descriptor you want to profile, set FORM to the question you want
answered, and adjust ATTRACTOR_PROFILE to point to the scenario of interest.

DEPENDENCIES
------------
  pip install numpy scipy matplotlib
  pip install git+https://github.com/ag-ross/PyCIB.git

OUTPUT
------
  - Console: ranked list of descriptor responses with directions
  - Console: Type I cross-impact multiplier diagonal with sign-reversal flags
  - Plot:    impulse response curves for all descriptors
  - File:    output_policy_shock_summary.txt: run parameters, ranked response
             table, multiplier diagonal, and any MC warnings in plain text
"""

from __future__ import annotations

import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE     = Path(__file__).resolve().parent
_ROOT     = _HERE.parent

# Add local development paths only if they exist (not present in standard installs).
for _p in [_ROOT / "PyCIB-main", _ROOT / "workings"]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.append(str(_p))

try:
    from scipy.linalg import expm
except ImportError:                                                # pragma: no cover
    warnings.warn(
        "scipy not installed; using numpy-only _lrt_shim fallback.",
        RuntimeWarning, stacklevel=1,
    )
    from _lrt_shim.linalg import expm

from cib.core import CIBMatrix
from cib.analysis import ScenarioAnalyzer


# ===========================================================================
# SECTION 1: HARDCODED CROSS-IMPACT MATRIX
#
# This is the full CIM for a 10-descriptor energy-transition study, taken
# from PyCIB's built-in example dataset (DATASET_C10).
#
# This is hardcoded for simplicity. The model code of the paper provides
# code where the CIM is imported from a csv file.
#
# Descriptors are listed in the order they will appear in the output.
# Each descriptor has three ordered states: index 0 (low/unfavourable),
# index 1 (medium/transitional), index 2 (high/favourable).
#
# Note: Technology_Costs is the exception, its states run High → Moderate → Low
# because lower costs are more favourable; the ordinal scale is inverted
# relative to the other descriptors.
# ===========================================================================

DESCRIPTORS: dict[str, list[str]] = {
    "Policy_Stringency":    ["Low",     "Medium",   "High"],
    "Regulatory_Framework": ["Weak",    "Moderate", "Strong"],
    "Renewables_Deployment":["Slow",    "Moderate", "Fast"],
    "Grid_Flexibility":     ["Low",     "Medium",   "High"],
    "Storage_Capacity":     ["Limited", "Moderate", "Extensive"],
    "Fossil_Price_Level":   ["Low",     "Medium",   "High"],
    "Investment_Level":     ["Low",     "Medium",   "High"],
    "Technology_Costs":     ["High",    "Moderate", "Low"],   # note: inverted scale
    "Electrification_Demand":["Low",    "Medium",   "High"],
    "Public_Acceptance":    ["Low",     "Medium",   "High"],
}

# ---------------------------------------------------------------------------
# Cross-impact scores.
#
# Format: (source_descriptor, source_state, target_descriptor, target_state): score
#
# Score interpretation:
#   +2 = strongly supportive     +1 = weakly supportive
#    0 = no direct influence      (entries with 0 are omitted)
#   -1 = weakly suppressive      -2 = strongly suppressive
#
# Only non-zero entries are listed.  All unspecified pairs have score = 0.
# ---------------------------------------------------------------------------

IMPACTS: dict[tuple[str, str, str, str], int] = {
    # --- Policy_Stringency influences ---
    ("Policy_Stringency", "High", "Electrification_Demand", "High"):    1,
    ("Policy_Stringency", "High", "Electrification_Demand", "Low"):    -1,
    ("Policy_Stringency", "High", "Fossil_Price_Level",     "High"):   -1,
    ("Policy_Stringency", "High", "Fossil_Price_Level",     "Low"):     1,
    ("Policy_Stringency", "High", "Grid_Flexibility",       "High"):    2,
    ("Policy_Stringency", "High", "Grid_Flexibility",       "Low"):    -2,
    ("Policy_Stringency", "High", "Regulatory_Framework",   "Strong"):  2,
    ("Policy_Stringency", "High", "Regulatory_Framework",   "Weak"):   -2,
    ("Policy_Stringency", "High", "Renewables_Deployment",  "Fast"):    2,
    ("Policy_Stringency", "High", "Renewables_Deployment",  "Slow"):   -2,
    ("Policy_Stringency", "Low",  "Electrification_Demand", "High"):   -1,
    ("Policy_Stringency", "Low",  "Electrification_Demand", "Low"):     1,
    ("Policy_Stringency", "Low",  "Fossil_Price_Level",     "High"):    1,
    ("Policy_Stringency", "Low",  "Fossil_Price_Level",     "Low"):    -1,
    ("Policy_Stringency", "Low",  "Grid_Flexibility",       "High"):   -2,
    ("Policy_Stringency", "Low",  "Grid_Flexibility",       "Low"):     2,
    ("Policy_Stringency", "Low",  "Regulatory_Framework",   "Strong"): -2,
    ("Policy_Stringency", "Low",  "Regulatory_Framework",   "Weak"):    2,
    ("Policy_Stringency", "Low",  "Renewables_Deployment",  "Fast"):   -2,
    ("Policy_Stringency", "Low",  "Renewables_Deployment",  "Slow"):    2,
    ("Policy_Stringency", "Medium", "Electrification_Demand", "High"): -1,
    ("Policy_Stringency", "Medium", "Electrification_Demand", "Low"):  -1,
    ("Policy_Stringency", "Medium", "Electrification_Demand", "Medium"): 1,
    ("Policy_Stringency", "Medium", "Fossil_Price_Level",   "High"):   -1,
    ("Policy_Stringency", "Medium", "Fossil_Price_Level",   "Low"):    -1,
    ("Policy_Stringency", "Medium", "Fossil_Price_Level",   "Medium"):  1,
    ("Policy_Stringency", "Medium", "Grid_Flexibility",     "High"):   -1,
    ("Policy_Stringency", "Medium", "Grid_Flexibility",     "Low"):    -1,
    ("Policy_Stringency", "Medium", "Grid_Flexibility",     "Medium"):  2,
    ("Policy_Stringency", "Medium", "Regulatory_Framework", "Moderate"): 2,
    ("Policy_Stringency", "Medium", "Regulatory_Framework", "Strong"): -1,
    ("Policy_Stringency", "Medium", "Regulatory_Framework", "Weak"):   -1,
    ("Policy_Stringency", "Medium", "Renewables_Deployment", "Fast"):  -1,
    ("Policy_Stringency", "Medium", "Renewables_Deployment", "Moderate"): 2,
    ("Policy_Stringency", "Medium", "Renewables_Deployment", "Slow"):  -1,

    # --- Regulatory_Framework influences ---
    ("Regulatory_Framework", "Strong",   "Investment_Level",      "High"):  1,
    ("Regulatory_Framework", "Strong",   "Investment_Level",      "Low"):  -1,
    ("Regulatory_Framework", "Strong",   "Renewables_Deployment", "Fast"):  2,
    ("Regulatory_Framework", "Strong",   "Renewables_Deployment", "Slow"): -2,
    ("Regulatory_Framework", "Moderate", "Investment_Level",      "High"): -1,
    ("Regulatory_Framework", "Moderate", "Investment_Level",      "Low"):  -1,
    ("Regulatory_Framework", "Moderate", "Investment_Level",      "Medium"): 1,
    ("Regulatory_Framework", "Moderate", "Renewables_Deployment", "Fast"): -1,
    ("Regulatory_Framework", "Moderate", "Renewables_Deployment", "Moderate"): 2,
    ("Regulatory_Framework", "Moderate", "Renewables_Deployment", "Slow"): -1,
    ("Regulatory_Framework", "Weak",     "Investment_Level",      "High"): -1,
    ("Regulatory_Framework", "Weak",     "Investment_Level",      "Low"):   1,
    ("Regulatory_Framework", "Weak",     "Renewables_Deployment", "Fast"): -2,
    ("Regulatory_Framework", "Weak",     "Renewables_Deployment", "Slow"):  2,

    # --- Renewables_Deployment influences ---
    ("Renewables_Deployment", "Fast",     "Electrification_Demand", "High"):  2,
    ("Renewables_Deployment", "Fast",     "Electrification_Demand", "Low"):  -2,
    ("Renewables_Deployment", "Fast",     "Fossil_Price_Level",     "High"): -1,
    ("Renewables_Deployment", "Fast",     "Fossil_Price_Level",     "Low"):   1,
    ("Renewables_Deployment", "Fast",     "Grid_Flexibility",       "High"):  2,
    ("Renewables_Deployment", "Fast",     "Grid_Flexibility",       "Low"):  -2,
    ("Renewables_Deployment", "Fast",     "Storage_Capacity",       "Extensive"): 1,
    ("Renewables_Deployment", "Fast",     "Storage_Capacity",       "Limited"):  -1,
    ("Renewables_Deployment", "Fast",     "Technology_Costs",       "High"):  1,
    ("Renewables_Deployment", "Fast",     "Technology_Costs",       "Low"):  -1,
    ("Renewables_Deployment", "Moderate", "Electrification_Demand", "High"): -1,
    ("Renewables_Deployment", "Moderate", "Electrification_Demand", "Low"):  -1,
    ("Renewables_Deployment", "Moderate", "Electrification_Demand", "Medium"): 2,
    ("Renewables_Deployment", "Moderate", "Fossil_Price_Level",     "High"): -1,
    ("Renewables_Deployment", "Moderate", "Fossil_Price_Level",     "Low"):  -1,
    ("Renewables_Deployment", "Moderate", "Fossil_Price_Level",     "Medium"): 1,
    ("Renewables_Deployment", "Moderate", "Grid_Flexibility",       "High"): -1,
    ("Renewables_Deployment", "Moderate", "Grid_Flexibility",       "Low"):  -1,
    ("Renewables_Deployment", "Moderate", "Grid_Flexibility",       "Medium"): 2,
    ("Renewables_Deployment", "Moderate", "Storage_Capacity",       "Extensive"): -1,
    ("Renewables_Deployment", "Moderate", "Storage_Capacity",       "Limited"):  -1,
    ("Renewables_Deployment", "Moderate", "Storage_Capacity",       "Moderate"):  1,
    ("Renewables_Deployment", "Moderate", "Technology_Costs",       "High"): -1,
    ("Renewables_Deployment", "Moderate", "Technology_Costs",       "Low"):  -1,
    ("Renewables_Deployment", "Moderate", "Technology_Costs",       "Moderate"): 1,
    ("Renewables_Deployment", "Slow",     "Electrification_Demand", "High"): -2,
    ("Renewables_Deployment", "Slow",     "Electrification_Demand", "Low"):   2,
    ("Renewables_Deployment", "Slow",     "Fossil_Price_Level",     "High"):  1,
    ("Renewables_Deployment", "Slow",     "Fossil_Price_Level",     "Low"):  -1,
    ("Renewables_Deployment", "Slow",     "Grid_Flexibility",       "High"): -2,
    ("Renewables_Deployment", "Slow",     "Grid_Flexibility",       "Low"):   2,
    ("Renewables_Deployment", "Slow",     "Storage_Capacity",       "Extensive"): -1,
    ("Renewables_Deployment", "Slow",     "Storage_Capacity",       "Limited"):   1,
    ("Renewables_Deployment", "Slow",     "Technology_Costs",       "High"):  -1,
    ("Renewables_Deployment", "Slow",     "Technology_Costs",       "Low"):    1,

    # --- Grid_Flexibility influences ---
    ("Grid_Flexibility", "High",   "Electrification_Demand", "High"):    2,
    ("Grid_Flexibility", "High",   "Electrification_Demand", "Low"):    -2,
    ("Grid_Flexibility", "High",   "Renewables_Deployment",  "Fast"):    1,
    ("Grid_Flexibility", "High",   "Renewables_Deployment",  "Slow"):   -1,
    ("Grid_Flexibility", "High",   "Storage_Capacity",       "Extensive"): 1,
    ("Grid_Flexibility", "High",   "Storage_Capacity",       "Limited"): -1,
    ("Grid_Flexibility", "Low",    "Electrification_Demand", "High"):   -2,
    ("Grid_Flexibility", "Low",    "Electrification_Demand", "Low"):     2,
    ("Grid_Flexibility", "Low",    "Renewables_Deployment",  "Fast"):   -1,
    ("Grid_Flexibility", "Low",    "Renewables_Deployment",  "Slow"):    1,
    ("Grid_Flexibility", "Low",    "Storage_Capacity",       "Extensive"): -1,
    ("Grid_Flexibility", "Low",    "Storage_Capacity",       "Limited"):   1,
    ("Grid_Flexibility", "Medium", "Electrification_Demand", "High"):   -1,
    ("Grid_Flexibility", "Medium", "Electrification_Demand", "Low"):    -1,
    ("Grid_Flexibility", "Medium", "Electrification_Demand", "Medium"):  2,
    ("Grid_Flexibility", "Medium", "Renewables_Deployment",  "Fast"):   -1,
    ("Grid_Flexibility", "Medium", "Renewables_Deployment",  "Moderate"): 1,
    ("Grid_Flexibility", "Medium", "Renewables_Deployment",  "Slow"):   -1,
    ("Grid_Flexibility", "Medium", "Storage_Capacity",       "Extensive"): -1,
    ("Grid_Flexibility", "Medium", "Storage_Capacity",       "Limited"):  -1,
    ("Grid_Flexibility", "Medium", "Storage_Capacity",       "Moderate"):  1,

    # --- Storage_Capacity influences ---
    ("Storage_Capacity", "Extensive", "Grid_Flexibility",      "High"):  2,
    ("Storage_Capacity", "Extensive", "Grid_Flexibility",      "Low"):  -2,
    ("Storage_Capacity", "Extensive", "Renewables_Deployment", "Fast"):  1,
    ("Storage_Capacity", "Extensive", "Renewables_Deployment", "Slow"): -1,
    ("Storage_Capacity", "Limited",   "Grid_Flexibility",      "High"): -2,
    ("Storage_Capacity", "Limited",   "Grid_Flexibility",      "Low"):   2,
    ("Storage_Capacity", "Limited",   "Renewables_Deployment", "Fast"): -1,
    ("Storage_Capacity", "Limited",   "Renewables_Deployment", "Slow"):  1,
    ("Storage_Capacity", "Moderate",  "Grid_Flexibility",      "High"): -1,
    ("Storage_Capacity", "Moderate",  "Grid_Flexibility",      "Low"):  -1,
    ("Storage_Capacity", "Moderate",  "Grid_Flexibility",      "Medium"): 2,
    ("Storage_Capacity", "Moderate",  "Renewables_Deployment", "Fast"): -1,
    ("Storage_Capacity", "Moderate",  "Renewables_Deployment", "Moderate"): 1,
    ("Storage_Capacity", "Moderate",  "Renewables_Deployment", "Slow"): -1,

    # --- Fossil_Price_Level influences ---
    ("Fossil_Price_Level", "High",   "Electrification_Demand", "High"):  2,
    ("Fossil_Price_Level", "High",   "Electrification_Demand", "Low"):  -2,
    ("Fossil_Price_Level", "High",   "Policy_Stringency",      "High"):  1,
    ("Fossil_Price_Level", "High",   "Policy_Stringency",      "Low"):  -1,
    ("Fossil_Price_Level", "High",   "Renewables_Deployment",  "Fast"):  2,
    ("Fossil_Price_Level", "High",   "Renewables_Deployment",  "Slow"): -2,
    ("Fossil_Price_Level", "Low",    "Electrification_Demand", "High"): -2,
    ("Fossil_Price_Level", "Low",    "Electrification_Demand", "Low"):   2,
    ("Fossil_Price_Level", "Low",    "Policy_Stringency",      "High"): -1,
    ("Fossil_Price_Level", "Low",    "Policy_Stringency",      "Low"):   1,
    ("Fossil_Price_Level", "Low",    "Renewables_Deployment",  "Fast"): -2,
    ("Fossil_Price_Level", "Low",    "Renewables_Deployment",  "Slow"):  2,
    ("Fossil_Price_Level", "Medium", "Electrification_Demand", "High"): -1,
    ("Fossil_Price_Level", "Medium", "Electrification_Demand", "Low"):  -1,
    ("Fossil_Price_Level", "Medium", "Electrification_Demand", "Medium"): 2,
    ("Fossil_Price_Level", "Medium", "Policy_Stringency",      "High"): -1,
    ("Fossil_Price_Level", "Medium", "Policy_Stringency",      "Low"):  -1,
    ("Fossil_Price_Level", "Medium", "Policy_Stringency",      "Medium"): 1,
    ("Fossil_Price_Level", "Medium", "Renewables_Deployment",  "Fast"): -1,
    ("Fossil_Price_Level", "Medium", "Renewables_Deployment",  "Moderate"): 2,
    ("Fossil_Price_Level", "Medium", "Renewables_Deployment",  "Slow"): -1,

    # --- Investment_Level influences ---
    ("Investment_Level", "High",   "Grid_Flexibility",      "High"):  1,
    ("Investment_Level", "High",   "Grid_Flexibility",      "Low"):  -1,
    ("Investment_Level", "High",   "Renewables_Deployment", "Fast"):  2,
    ("Investment_Level", "High",   "Renewables_Deployment", "Slow"): -2,
    ("Investment_Level", "High",   "Storage_Capacity",      "Extensive"): 2,
    ("Investment_Level", "High",   "Storage_Capacity",      "Limited"):  -2,
    ("Investment_Level", "High",   "Technology_Costs",      "High"):  1,
    ("Investment_Level", "High",   "Technology_Costs",      "Low"):  -1,
    ("Investment_Level", "Low",    "Grid_Flexibility",      "High"): -1,
    ("Investment_Level", "Low",    "Grid_Flexibility",      "Low"):   1,
    ("Investment_Level", "Low",    "Renewables_Deployment", "Fast"): -2,
    ("Investment_Level", "Low",    "Renewables_Deployment", "Slow"):  2,
    ("Investment_Level", "Low",    "Storage_Capacity",      "Extensive"): -2,
    ("Investment_Level", "Low",    "Storage_Capacity",      "Limited"):   2,
    ("Investment_Level", "Low",    "Technology_Costs",      "High"): -1,
    ("Investment_Level", "Low",    "Technology_Costs",      "Low"):   1,
    ("Investment_Level", "Medium", "Grid_Flexibility",      "High"): -1,
    ("Investment_Level", "Medium", "Grid_Flexibility",      "Low"):  -1,
    ("Investment_Level", "Medium", "Grid_Flexibility",      "Medium"): 1,
    ("Investment_Level", "Medium", "Renewables_Deployment", "Fast"): -1,
    ("Investment_Level", "Medium", "Renewables_Deployment", "Moderate"): 2,
    ("Investment_Level", "Medium", "Renewables_Deployment", "Slow"): -1,
    ("Investment_Level", "Medium", "Storage_Capacity",      "Extensive"): -1,
    ("Investment_Level", "Medium", "Storage_Capacity",      "Limited"):  -1,
    ("Investment_Level", "Medium", "Storage_Capacity",      "Moderate"):  2,
    ("Investment_Level", "Medium", "Technology_Costs",      "High"): -1,
    ("Investment_Level", "Medium", "Technology_Costs",      "Low"):  -1,
    ("Investment_Level", "Medium", "Technology_Costs",      "Moderate"): 1,

    # --- Technology_Costs influences ---
    ("Technology_Costs", "High",     "Investment_Level",      "High"):  1,
    ("Technology_Costs", "High",     "Investment_Level",      "Low"):  -1,
    ("Technology_Costs", "High",     "Renewables_Deployment", "Fast"):  2,
    ("Technology_Costs", "High",     "Renewables_Deployment", "Slow"): -2,
    ("Technology_Costs", "Low",      "Investment_Level",      "High"): -1,
    ("Technology_Costs", "Low",      "Investment_Level",      "Low"):   1,
    ("Technology_Costs", "Low",      "Renewables_Deployment", "Fast"): -2,
    ("Technology_Costs", "Low",      "Renewables_Deployment", "Slow"):  2,
    ("Technology_Costs", "Moderate", "Investment_Level",      "High"): -1,
    ("Technology_Costs", "Moderate", "Investment_Level",      "Low"):  -1,
    ("Technology_Costs", "Moderate", "Investment_Level",      "Medium"): 1,
    ("Technology_Costs", "Moderate", "Renewables_Deployment", "Fast"): -1,
    ("Technology_Costs", "Moderate", "Renewables_Deployment", "Moderate"): 2,
    ("Technology_Costs", "Moderate", "Renewables_Deployment", "Slow"): -1,

    # --- Electrification_Demand influences ---
    ("Electrification_Demand", "High",   "Grid_Flexibility",   "High"):  2,
    ("Electrification_Demand", "High",   "Grid_Flexibility",   "Low"):  -2,
    ("Electrification_Demand", "High",   "Policy_Stringency",  "High"):  1,
    ("Electrification_Demand", "High",   "Policy_Stringency",  "Low"):  -1,
    ("Electrification_Demand", "High",   "Renewables_Deployment", "Fast"):  1,
    ("Electrification_Demand", "High",   "Renewables_Deployment", "Slow"): -1,
    ("Electrification_Demand", "Low",    "Grid_Flexibility",   "High"): -2,
    ("Electrification_Demand", "Low",    "Grid_Flexibility",   "Low"):   2,
    ("Electrification_Demand", "Low",    "Policy_Stringency",  "High"): -1,
    ("Electrification_Demand", "Low",    "Policy_Stringency",  "Low"):   1,
    ("Electrification_Demand", "Low",    "Renewables_Deployment", "Fast"): -1,
    ("Electrification_Demand", "Low",    "Renewables_Deployment", "Slow"):  1,
    ("Electrification_Demand", "Medium", "Grid_Flexibility",   "High"): -1,
    ("Electrification_Demand", "Medium", "Grid_Flexibility",   "Low"):  -1,
    ("Electrification_Demand", "Medium", "Grid_Flexibility",   "Medium"): 2,
    ("Electrification_Demand", "Medium", "Policy_Stringency",  "High"): -1,
    ("Electrification_Demand", "Medium", "Policy_Stringency",  "Low"):  -1,
    ("Electrification_Demand", "Medium", "Policy_Stringency",  "Medium"): 1,
    ("Electrification_Demand", "Medium", "Renewables_Deployment", "Fast"): -1,
    ("Electrification_Demand", "Medium", "Renewables_Deployment", "Moderate"): 1,
    ("Electrification_Demand", "Medium", "Renewables_Deployment", "Slow"): -1,

    # --- Public_Acceptance influences ---
    ("Public_Acceptance", "High",   "Policy_Stringency",     "High"):  1,
    ("Public_Acceptance", "High",   "Policy_Stringency",     "Low"):  -1,
    ("Public_Acceptance", "High",   "Renewables_Deployment", "Fast"):  1,
    ("Public_Acceptance", "High",   "Renewables_Deployment", "Slow"): -1,
    ("Public_Acceptance", "Low",    "Policy_Stringency",     "High"): -1,
    ("Public_Acceptance", "Low",    "Policy_Stringency",     "Low"):   1,
    ("Public_Acceptance", "Low",    "Renewables_Deployment", "Fast"): -1,
    ("Public_Acceptance", "Low",    "Renewables_Deployment", "Slow"):  1,
    ("Public_Acceptance", "Medium", "Policy_Stringency",     "High"): -1,
    ("Public_Acceptance", "Medium", "Policy_Stringency",     "Low"):  -1,
    ("Public_Acceptance", "Medium", "Policy_Stringency",     "Medium"): 1,
    ("Public_Acceptance", "Medium", "Renewables_Deployment", "Fast"): -1,
    ("Public_Acceptance", "Medium", "Renewables_Deployment", "Moderate"): 1,
    ("Public_Acceptance", "Medium", "Renewables_Deployment", "Slow"): -1,
}


# ===========================================================================
# SECTION 2: ANALYSIS SETTINGS
#
# Change FORM to switch between the two readings of the network:
#   "backward"  response of SHOCK_DESCRIPTOR to a unit push on each other
#               descriptor (sensitivity profile, the form reported in the paper)
#   "forward"   response of each other descriptor to a unit push on
#               SHOCK_DESCRIPTOR (forward profile)
# Change SHOCK_DESCRIPTOR to profile a different descriptor.
# Change ATTRACTOR_PROFILE to target a different scenario.
# ===========================================================================

FORM = "backward"

# The profiled descriptor: the responding one (backward) or the pushed one (forward).
SHOCK_DESCRIPTOR = "Policy_Stringency"

# The attractor (consistent scenario) at which the shock is evaluated.
# The example selects the first consistent scenario where every entry in
# this dictionary matches. You need only specify the descriptors that
# uniquely identify the scenario of interest.
ATTRACTOR_PROFILE = {
    "Policy_Stringency":     "High",
    "Renewables_Deployment": "Fast",
    "Fossil_Price_Level":    "Low",
    "Storage_Capacity":      "Extensive",
    "Technology_Costs":      "Low",   # low costs = most favourable state
}

# How far into model time to trace the shock response.
# Set to None to use an automatic value based on the network's slowest
# timescale (5 × slowest relaxation time, minimum 20 units).
TAU_MAX = None

# Time step for the response curves.  Smaller = smoother, but slower.
DT = 0.05

# IO-3 rescaling target.  Empirical CIMs are almost always too strongly
# interconnected for the maths to converge without rescaling.  This factor
# (must be between 0 and 1) controls how much the network is damped.
# 0.9 is the standard value from the companion paper.
IO3_TARGET_RHO = 0.9

# ---------------------------------------------------------------------------
# Monte Carlo uncertainty bands.
#
# The shaded ribbons answer the question: "how sensitive are these curves to
# small errors in the agreed cross-impact scores?"  For each realisation,
# independent Gaussian noise with standard deviation MC_SIGMA is added to
# every off-diagonal entry of W (representing elicitation uncertainty of
# roughly half an ordinal level per score).  IO-3 rescaling is re-applied,
# and the IRF is recomputed.  The 5th–95th percentile band across all
# accepted realisations forms the ribbon for each descriptor.
#
# Narrow ribbons → the result is a robust structural property of the network.
# Wide ribbons   → the result is sensitive to the specific agreed scores.
#
# The default is 10,000 to match the companion paper.  Reduce to 1,000
# for faster exploratory runs (bands will be slightly less smooth).
# ---------------------------------------------------------------------------
N_MC     = 10_000  # number of Monte Carlo realisations (matches companion paper)
MC_SIGMA = 0.5    # noise standard deviation (≈ half an ordinal level per score)
MC_SEED  = 99     # random seed for reproducibility (matches companion paper)
DT_MC    = 0.1    # coarser time step for MC (finer deterministic DT is used above)
CI_LO    = 5      # lower percentile for the ribbon
CI_HI    = 95     # upper percentile for the ribbon

# Output plot file (None = show interactively instead of saving).
OUTPUT_FILE    = _HERE / "output_policy_shock.pdf"

# Output summary file (None = skip writing the text summary).
SUMMARY_FILE   = _HERE / "output_policy_shock_summary.txt"


# ===========================================================================
# SECTION 3: SETTINGS VALIDATION
#
# Fix the orientation (M^T backward, M forward) and the wording that FORM
# selects, then catch configuration errors before any computation begins.
# ===========================================================================

_FORMS = {
    "backward": {
        "orient":    lambda A: A.T,
        "title":     "Sensitivity profile",
        "impulse":   "response to a +1 unit impulse on each other descriptor",
        "summary":   f"response of '{SHOCK_DESCRIPTOR}' to a +1 unit impulse on each other descriptor",
        "push":      f"A unit push on this descriptor pressures '{SHOCK_DESCRIPTOR}'",
        "unreached": f"A unit push on this descriptor does not reach '{SHOCK_DESCRIPTOR}'",
        "uncertain": "the response to a push on them",
        "R":         "R",
        "ylabel":    "Activation pressure  R(τ)",
        "Lambda":    "Lambda",
        "inverse":   "(I - W)^{-T}",
    },
    "forward": {
        "orient":    lambda A: A,
        "title":     "Forward profile",
        "impulse":   f"response of each other descriptor to a +1 unit impulse on '{SHOCK_DESCRIPTOR}'",
        "summary":   f"response of each other descriptor to a +1 unit impulse on '{SHOCK_DESCRIPTOR}'",
        "push":      f"A unit push on '{SHOCK_DESCRIPTOR}' pressures this descriptor",
        "unreached": f"A unit push on '{SHOCK_DESCRIPTOR}' does not reach this descriptor",
        "uncertain": f"their response to a push on '{SHOCK_DESCRIPTOR}'",
        "R":         "R^F",
        "ylabel":    "Activation pressure  $\\mathregular{R^F}$(τ)",
        "Lambda":    "Lambda^F",
        "inverse":   "(I - W)^{-1}",
    },
}

if FORM not in _FORMS:
    raise ValueError(
        f"FORM = {FORM!r} is invalid.  "
        "It must be 'backward' or 'forward'.  "
        "Check SECTION 2: ANALYSIS SETTINGS."
    )
_F = _FORMS[FORM]

if not (0.0 < IO3_TARGET_RHO < 1.0):
    raise ValueError(
        f"IO3_TARGET_RHO = {IO3_TARGET_RHO} is invalid.  "
        "It must be strictly between 0 and 1 (typical value: 0.9).  "
        "Check SECTION 2: ANALYSIS SETTINGS."
    )

if SHOCK_DESCRIPTOR not in DESCRIPTORS:
    raise ValueError(
        f"SHOCK_DESCRIPTOR = {SHOCK_DESCRIPTOR!r} is not in the descriptor list.\n"
        f"Available descriptors: {list(DESCRIPTORS.keys())}"
    )

for d, s in ATTRACTOR_PROFILE.items():
    if d not in DESCRIPTORS:
        raise ValueError(
            f"ATTRACTOR_PROFILE contains unknown descriptor {d!r}.\n"
            f"Available descriptors: {list(DESCRIPTORS.keys())}"
        )
    if s not in DESCRIPTORS[d]:
        raise ValueError(
            f"ATTRACTOR_PROFILE: state {s!r} is not valid for descriptor {d!r}.\n"
            f"Valid states for {d!r}: {DESCRIPTORS[d]}"
        )

if DT <= 0 or DT_MC <= 0:
    raise ValueError("DT and DT_MC must both be positive.")

if N_MC < 100:
    warnings.warn(
        f"N_MC = {N_MC} is very small.  Monte Carlo bands will be noisy.  "
        "Use N_MC >= 1000 for smooth ribbons and N_MC = 10000 for publication quality.",
        stacklevel=1,
    )


# ===========================================================================
# SECTION 4: BUILD CIBMatrix AND FIND CONSISTENT SCENARIOS
#
# CIBMatrix is the PyCIB data structure that stores the cross-impact scores.
# ScenarioAnalyzer finds all consistent scenarios (structural equilibria) by
# exhaustive search; guaranteed complete for matrices of this size.
# ===========================================================================

print("=" * 60)
print("CIB-LRT Example: Unit-Impulse Shock Analysis")
print("=" * 60)

# Build the PyCIB matrix object from our hardcoded descriptors and impacts.
matrix = CIBMatrix(DESCRIPTORS)
for (src_d, src_s, tgt_d, tgt_s), score in IMPACTS.items():
    matrix.set_impact(src_d, src_s, tgt_d, tgt_s, score)

print(f"\nDescriptors : {matrix.n_descriptors}")
n_states = [len(states) for states in DESCRIPTORS.values()]
if len(set(n_states)) == 1:
    print(f"States each : {n_states[0]}")
    print(f"Scenario space: {n_states[0]}^{matrix.n_descriptors} = {n_states[0]**matrix.n_descriptors:,} candidates")
else:
    print(f"States each : {n_states}")
    print(f"Scenario space: {int(np.prod(n_states)):,} candidates")

# Find all consistent scenarios.  For a 10-descriptor × 3-state matrix this
# completes in well under a second.
print("\nSearching for consistent scenarios (structural equilibria) ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    search = ScenarioAnalyzer(matrix).find_all_consistent_exact()

scenarios = search.scenarios
print(f"Found {len(scenarios)} consistent scenarios:")
for i, sc in enumerate(scenarios, 1):
    tag = f"  sc{i:02d}"
    states = sc.to_dict()
    fields = "  ".join(f"{d}={states.get(d, '?'):<10}" for d in ATTRACTOR_PROFILE)
    print(f"{tag}  {fields}".rstrip())


# ===========================================================================
# SECTION 5: SELECT THE TARGET ATTRACTOR
#
# We identify the scenario that matches ATTRACTOR_PROFILE.  The profile
# only needs to specify enough descriptors to uniquely pick one scenario.
# If more than one scenario matches, a warning is printed and the first
# match is used, but the practitioner should extend ATTRACTOR_PROFILE
# until the match is unique.
# ===========================================================================

matching = [
    sc for sc in scenarios
    if all(sc.to_dict().get(d) == s for d, s in ATTRACTOR_PROFILE.items())
]

if len(matching) == 0:
    raise ValueError(
        f"No consistent scenario matches ATTRACTOR_PROFILE = {ATTRACTOR_PROFILE}.\n"
        "Check your profile entries or print the scenario list above."
    )

if len(matching) > 1:
    warnings.warn(
        f"ATTRACTOR_PROFILE matches {len(matching)} consistent scenarios, not just one.  "
        "The first match will be used, but the shock profile is attractor-specific.  "
        "Add more descriptors to ATTRACTOR_PROFILE to uniquely identify the scenario "
        "you intend.",
        stacklevel=1,
    )

target_sc = matching[0]

print(f"\nSelected attractor:")
for d, s in target_sc.to_dict().items():
    print(f"  {d:<30} : {s}")


# ===========================================================================
# SECTION 6: BUILD THE EFFECTIVE CROSS-IMPACT MATRIX W
#
# At a consistent scenario, each descriptor is in a specific state.
# W[j, i] picks up the cross-impact score that source descriptor i (in its
# current attractor state) exerts on target descriptor j (in its attractor
# state).  This gives an N × N matrix that captures the immediate structural
# coupling at this particular equilibrium.
# ===========================================================================

desc_names = list(DESCRIPTORS.keys())
N = len(desc_names)
sc_dict = target_sc.to_dict()

W = np.zeros((N, N))
for j, tgt_d in enumerate(desc_names):
    for i, src_d in enumerate(desc_names):
        if i == j:
            continue   # CIB has no self-impacts
        key = (src_d, sc_dict[src_d], tgt_d, sc_dict[tgt_d])
        W[j, i] = IMPACTS.get(key, 0.0)

rho_raw = float(np.max(np.abs(np.linalg.eigvals(W))))
print(f"\nRaw spectral radius ρ(W) = {rho_raw:.3f}  "
      f"(M is guaranteed stable when ρ(W) < 1; typical empirical CIMs are 3-20)")


# ===========================================================================
# SECTION 7: IO-3 RESCALING
#
# Empirical CIMs almost always produce a spectral radius > 1, meaning the
# raw network is explosive rather than stable.  IO-3 rescaling multiplies
# every entry in W by the same small factor α = target_rho / ρ(W).
#
# This preserves every relative magnitude and the full set of consistent
# scenarios, while ensuring the system can settle after a shock.  Think of
# it as tuning down the overall volume of cross-impact scores uniformly.
# ===========================================================================

# Guard against a degenerate CIM where all off-diagonal scores are zero,
# which would make rho(W) ≈ 0 and division undefined.
if rho_raw < 1e-10:
    raise RuntimeError(
        "The effective cross-impact matrix W at this attractor has spectral "
        "radius ≈ 0 (all off-diagonal entries are effectively zero).  "
        "IO-3 rescaling cannot be applied.  Check that the selected attractor "
        "has non-zero cross-impact scores between at least some descriptors."
    )

alpha = IO3_TARGET_RHO / rho_raw
W_sc  = W * alpha
# The drift matrix M = W - I governs how the system relaxes back to
# equilibrium after a perturbation.  For the maths to work (stable system),
# all eigenvalues of M must have negative real parts.
M_sc  = W_sc - np.eye(N)

# Verify stability after rescaling.
_STABILITY_TOL = 1.0e-10
eigs_M     = np.linalg.eigvals(M_sc)
is_stable  = bool(np.all(eigs_M.real < -_STABILITY_TOL))
slowest    = float(np.min(np.abs(eigs_M.real)))   # |Re(λ)| of slowest mode
timescale  = 1.0 / slowest                        # in dimensionless model units

print(f"IO-3 rescaling factor α = {alpha:.4f}  →  ρ(αW) = {IO3_TARGET_RHO:.3f}  "
      f"stable = {is_stable}")

if not is_stable:
    raise RuntimeError(
        "IO-3 rescaling did not produce a stable network: at least one eigenvalue of M "
        "has a non-negative real part.  LRT results are only valid for stable networks.  "
        "Try reducing IO3_TARGET_RHO (e.g. to 0.7 or 0.5) and re-run.  "
        "If instability persists, inspect the eigenvalues of M printed below:\n"
        f"  Re(λ) = {sorted(eigs_M.real.tolist(), reverse=True)}"
    )

print(f"Slowest relaxation timescale ≈ {timescale:.1f} model units")


# ===========================================================================
# SECTION 8: UNIT-IMPULSE SHOCK ANALYSIS
#
# The unit vector ε = e_i is a vector of zeros with a 1 in position i
# (the profiled descriptor).  FORM selects the impulse response:
#
#   backward  R(τ) = exp(M^T τ) ε, row i of exp(M τ), the response of
#             descriptor i to a unit push on each other descriptor:
#             R_j(τ) = activation pressure on descriptor i at time τ from a unit push on j
#   forward   R^F(τ) = exp(M τ) ε, column i of exp(M τ), the response of
#             each other descriptor to a unit push on descriptor i:
#             R^F_j(τ) = activation pressure on descriptor j at time τ from a unit push on i
#
# Below, R_j(τ) also denotes R^F_j(τ) in the forward form.
# Positive R_j(τ): the push moves the responding descriptor (i backward,
#                  j forward) toward a higher ordinal state (upward pressure).
# Negative R_j(τ): the push moves the responding descriptor toward a
#                  lower ordinal state (downward pressure).
#
# All pressures decay to zero as the system absorbs the shock → provided
# the network is stable (ensured by IO-3 rescaling above).
# ===========================================================================

shock_idx         = desc_names.index(SHOCK_DESCRIPTOR)
epsilon           = np.zeros(N)
epsilon[shock_idx] = 1.0   # unit weight selects the profiled descriptor

# Choose the time horizon: five slowest timescales, or at least 20 units.
tau_max = TAU_MAX if TAU_MAX is not None else max(5.0 * timescale, 20.0)
taus    = np.arange(0.0, tau_max + DT / 2.0, DT)

# Compute the response curves by repeated application of the matrix
# exponential step exp(K Δt), with K = M^T (backward) or M (forward).
# This is the semigroup property: R(τ + Δt) = exp(K Δt) · R(τ).
# .real: scipy returns real output for real input; the numpy fallback used
# when scipy is absent may return complex with negligible imaginary residual.
exp_step = expm(_F["orient"](M_sc) * DT).real
curves   = np.zeros((len(taus), N))
curves[0] = epsilon
for k in range(1, len(taus)):
    curves[k] = exp_step @ curves[k - 1]

print(f"\n{_F['title']} of '{SHOCK_DESCRIPTOR}': {_F['impulse']}")
print(f"Time horizon τ_max = {tau_max:.1f} model units  (Δt = {DT})")


# ===========================================================================
# SECTION 9: RESULTS: RANKED DESCRIPTOR RESPONSES
#
# For each other descriptor j, we report:
#   - Peak |R_j(τ)|: how strongly the responding descriptor (the profiled
#                    descriptor backward, j forward) responds to the unit push,
#                    as the maximum of |R_j(τ)| over τ
#   - Direction:     positive (upward pressure) or negative (downward pressure)
#                    on the responding descriptor at the moment of peak
#                    response, or none if the curve is identically zero
#   - τ at peak:     how quickly the response peaks
# ===========================================================================

_ZERO_TOL = 1.0e-12
peak_hdr  = f"Peak |{_F['R']}|"

print(f"\n{'Rank':<5} {'Descriptor':<30} {peak_hdr:<12} "
      f"{'Direction':<12} {'τ at peak':<10}")
print("  " + "-" * 68)

other_idx = [i for i in range(N) if i != shock_idx]
ranked = sorted(
    other_idx,
    key=lambda i: np.max(np.abs(curves[:, i])),
    reverse=True,
)

unreached = [i for i in other_idx if np.max(np.abs(curves[:, i])) < _ZERO_TOL]

for rank, i in enumerate(ranked, 1):
    peak_val  = float(curves[np.argmax(np.abs(curves[:, i])), i])
    peak_abs  = abs(peak_val)
    direction = "none" if i in unreached else "positive ↑" if peak_val > 0 else "negative ↓"
    tau_pk    = float(taus[np.argmax(np.abs(curves[:, i]))])
    tau_txt   = f"{'-':<10}" if i in unreached else f"{tau_pk:<10.2f}"
    pk_txt    = f"{'-':<12}" if i in unreached else f"{peak_abs:<12.4f}"
    print(f"  {rank:<4} {desc_names[i]:<30} {pk_txt} "
          f"{direction:<12} {tau_txt}")

print()
print("Interpretation guide:")
print(f"  positive ↑  {_F['push']}")
print("              toward a higher ordinal state (upward pressure).")
print(f"  negative ↓  {_F['push']}")
print("              toward a lower ordinal state (downward pressure).")
if unreached:
    print(f"  none        {_F['unreached']}")
    print("              in the elicited network at this attractor (the curve is identically zero).")
print()
print("Note: responses reflect ALL direct and indirect network paths,")
print("not just the immediate cross-impact scores.  Counterintuitive")
print("directions reveal indirect feedback loops in the CIM structure.")
print()
print("IMPORTANT: the τ axis is dimensionless model time, not calendar")
print("time.  The late-time ordering of responses is a structural property")
print("of the network.  The early-peak order also depends on the rescaling")
print("target.  Do NOT attach calendar dates to these curves without")
print("independent empirical calibration.")


# ===========================================================================
# SECTION 10: TYPE I CROSS-IMPACT MULTIPLIER
#
# The Type I cross-impact multiplier Lambda = -M^{-T} = (I - W)^{-T}
# (backward) or Lambda^F = -M^{-1} = (I - W)^{-1} = Lambda^T (forward)
# aggregates all direct and indirect influence chains through the network
# into a single matrix.  It is the CIB analogue of the Leontief inverse
# in input-output economics.
#
# The diagonal entry Lambda_jj, identical in both forms, gives the total
# cumulative activation that descriptor j accumulates from sustained pressure
# on itself, propagated through all indirect feedback chains.  A NEGATIVE
# diagonal entry is a structural warning: the indirect feedback ultimately
# reverses the direction of the descriptor's own response.  It does not
# predict sign changes in the curves above, which are off-diagonal responses
# between the profiled descriptor and the others.  (See paper Section 2.6 and
# Supplementary Table S2.)
# ===========================================================================

print("\n" + "=" * 60)
print(f"Type I Cross-Impact Multiplier  {_F['Lambda']} = {_F['inverse']}")
print("=" * 60)

lambda_diag = None   # set here; used again in Section 13 export
lambda_hdr  = f"diag({_F['Lambda']})"
try:
    Lambda      = _F["orient"](-np.linalg.inv(M_sc))   # -M^{-T} backward, -M^{-1} forward
    lambda_diag = np.diag(Lambda)

    print(f"\n{'Descriptor':<30} {lambda_hdr:<18} {'Note'}")
    print("  " + "-" * 68)
    for j, name in enumerate(desc_names):
        val  = lambda_diag[j]
        note = "negative: cumulative self-response reverses sign" if val < 0 else ""
        print(f"  {name:<30} {val:<18.4f} {note}")

    n_negative = int(np.sum(lambda_diag < 0))
    print()
    if n_negative > 0:
        print(f"  {n_negative} descriptor(s) have negative diagonal entries.")
        print("  Their cumulative self-response reverses sign before settling.")
        print("  This is a structural property of the network at this attractor")
        print("  and rescaling target, not an artefact of the noise level.")
    else:
        print("  All diagonal entries are positive: no cumulative self-response reverses sign.")

except np.linalg.LinAlgError:
    print("  WARNING: M is singular. Type I multiplier could not be computed.")
    print("  This should not occur after IO-3 rescaling; check the CIM entries.")

print()

# ===========================================================================
# SECTION 11: MONTE CARLO UNCERTAINTY BANDS
#
# We perturb the agreed cross-impact scores with small random noise and
# recompute the IRF for each realisation.  This shows how much the curves
# would shift if the expert panel had scored each relationship slightly
# differently, quantifying the robustness of the structural result.
#
# The unit-impulse vector ε is kept fixed across all realisations: only
# the network (W) varies, not the profiled descriptor.
# ===========================================================================

print(f"\nRunning {N_MC}-sample Monte Carlo ribbon "
      f"(σ = {MC_SIGMA}, seed = {MC_SEED}) ...")

rng        = np.random.default_rng(MC_SEED)
taus_mc    = np.arange(0.0, tau_max + DT_MC / 2.0, DT_MC)
T_mc       = len(taus_mc)
mc_curves  = []   # accepted realisations: list of (T_mc × N) arrays

for _ in range(N_MC):
    # Add independent Gaussian noise to every off-diagonal entry of W.
    noise = rng.normal(0.0, MC_SIGMA, (N, N))
    np.fill_diagonal(noise, 0.0)          # self-impacts remain zero
    W_noisy = W + noise

    # Re-apply IO-3 rescaling so the noisy network is also stable.
    rho_n = float(np.max(np.abs(np.linalg.eigvals(W_noisy))))
    if rho_n < 1e-10:
        continue
    W_n_sc = W_noisy * (IO3_TARGET_RHO / rho_n)
    M_n_sc = W_n_sc - np.eye(N)

    # Skip realisations where the rescaled matrix is not stable.
    if not np.all(np.linalg.eigvals(M_n_sc).real < -_STABILITY_TOL):
        continue

    # Compute the IRF at the coarser MC time step.
    exp_mc = expm(_F["orient"](M_n_sc) * DT_MC).real  # see note on exp_step above
    c      = np.zeros((T_mc, N))
    c[0]   = epsilon
    for k in range(1, T_mc):
        c[k] = exp_mc @ c[k - 1]

    # Discard realisations that produce non-finite values (rare edge cases).
    if np.all(np.isfinite(c)):
        mc_curves.append(c)

n_accepted = len(mc_curves)
rejection_rate = (N_MC - n_accepted) / N_MC
print(f"  {n_accepted}/{N_MC} realisations accepted  "
      f"(rejection rate {100 * rejection_rate:.1f}%).")

if rejection_rate > 0.05:
    warnings.warn(
        f"Monte Carlo rejection rate is {100 * rejection_rate:.1f}%, substantially above "
        "zero.  This suggests the rescaled network is close to the stability boundary.  "
        "The uncertainty bands may be unreliable.  Consider reducing IO3_TARGET_RHO "
        "(e.g. to 0.7) and re-running.",
        stacklevel=1,
    )

if n_accepted < 2:
    raise RuntimeError(
        "Fewer than 2 Monte Carlo realisations were accepted; cannot compute "
        "percentile bands.  Reduce IO3_TARGET_RHO or MC_SIGMA and re-run."
    )

# Stack into a (n_accepted × T_mc × N) array and compute percentile bands.
mc_arr  = np.array(mc_curves)
mc_lo   = np.percentile(mc_arr, CI_LO, axis=0)   # shape (T_mc, N)
mc_hi   = np.percentile(mc_arr, CI_HI, axis=0)

# Warn if the band interpolated at the deterministic peak time straddles zero.
uncertain_dirs = []
for i in other_idx:
    if i in unreached:
        continue   # an identically zero curve has no peak
    tau_pk = taus[np.argmax(np.abs(curves[:, i]))]
    if np.interp(tau_pk, taus_mc, mc_lo[:, i]) < 0 < np.interp(tau_pk, taus_mc, mc_hi[:, i]):
        uncertain_dirs.append(desc_names[i])

if uncertain_dirs:
    print()
    print("WARNING: the following descriptor(s) have MC bands that straddle zero at the peaks of their curves,")
    print(f"meaning the direction of {_F['uncertain']} is uncertain at the elicitation")
    print("noise level used.  Do not report their direction as a firm conclusion:")
    for d in uncertain_dirs:
        print(f"  • {d}")
    if unreached:
        print("Curves with direction none have no peak and are not tested.")


# ===========================================================================
# SECTION 12: PLOT
#
# Each coloured line is the activation pressure R_j(τ) over time on the
# profiled descriptor from a unit push on descriptor j (backward), or on
# descriptor j from a unit push on the profiled descriptor (forward).  All
# curves start at zero, peak, and decay to zero as the network absorbs the
# push, except that a curve whose push never reaches the responding
# descriptor stays at zero.
# ===========================================================================

cmap   = matplotlib.colormaps["tab10"]
colors = [cmap(i % 10) for i in range(N)]

fig, ax = plt.subplots(figsize=(8, 4.5))

for i in range(N):
    if i == shock_idx:
        continue   # the self-response of the profiled descriptor is not plotted
    col   = colors[i]
    label = desc_names[i].replace("_", " ")

    # Shaded MC band (5th–95th percentile across realisations).
    ax.fill_between(taus_mc, mc_lo[:, i], mc_hi[:, i],
                    color=col, alpha=0.15, linewidth=0)

    # Deterministic IRF curve on top of the band.
    ax.plot(taus, curves[:, i], color=col, lw=1.2,
            alpha=0.85, label=label)

ax.axhline(0, color="0.6", lw=0.6, ls="--")
ax.set_xlabel("Model time τ (dimensionless)", fontsize=10)
ax.set_ylabel(_F["ylabel"], fontsize=10)
attractor_label = textwrap.fill(", ".join(f"{d.replace('_', ' ')}={sc_dict[d]}" for d in ATTRACTOR_PROFILE), 100)
ax.set_title(
    f"{_F['title']} of '{SHOCK_DESCRIPTOR.replace('_', ' ')}'\n"
    f"at attractor: {attractor_label}\n"
    f"[shaded: {CI_LO}–{CI_HI}th pct, N={n_accepted} MC draws, σ={MC_SIGMA}]",
    fontsize=8,
)
ax.legend(ncol=2, fontsize=7, loc="upper right", frameon=False,
          handlelength=1.2, labelspacing=0.2)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.grid(True, alpha=0.15)

plt.tight_layout()

if OUTPUT_FILE is not None:
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"Plot saved → {OUTPUT_FILE.name}")
else:
    plt.show()

plt.close(fig)


# ===========================================================================
# SECTION 13: EXPORT TEXT SUMMARY
#
# Writes the key analytical results to a plain-text file alongside the plot.
# This captures everything a practitioner would want to take away: the run
# parameters, the selected attractor, the ranked response table, the Type I
# multiplier diagonal, and any MC warnings.  The visual story is in the PDF;
# the numbers are here.
# ===========================================================================

if SUMMARY_FILE is not None:
    with open(SUMMARY_FILE, "w", encoding="utf-8") as _f:

        def _w(line=""):
            _f.write(line + "\n")

        _w("=" * 60)
        _w("CIB-LRT Unit-Impulse Shock Analysis: Results Summary")
        _w("=" * 60)

        _w()
        _w("RUN PARAMETERS")
        _w("-" * 40)
        _w(f"  Profiled descriptor    : {SHOCK_DESCRIPTOR}")
        _w(f"  IO-3 target rho        : {IO3_TARGET_RHO}")
        _w(f"  Raw spectral radius    : {rho_raw:.3f}")
        _w(f"  Rescaling factor alpha : {alpha:.4f}")
        _w(f"  Network stable         : {is_stable}")
        _w(f"  Slowest timescale      : {timescale:.1f} model units")
        _w(f"  Time horizon tau_max   : {tau_max:.1f} model units")
        _w(f"  MC realisations        : {n_accepted}/{N_MC} accepted "
           f"(rejection rate {100 * rejection_rate:.1f}%)")
        _w(f"  MC noise sigma         : {MC_SIGMA}")
        _w(f"  MC seed                : {MC_SEED}")
        _w(f"  MC percentile band     : {CI_LO}th-{CI_HI}th")

        _w()
        _w("SELECTED ATTRACTOR")
        _w("-" * 40)
        for d, s in target_sc.to_dict().items():
            _w(f"  {d:<30} : {s}")

        _w()
        _w("RANKED DESCRIPTOR RESPONSES")
        _w(f"({_F['summary']})")
        _w("-" * 68)
        _w(f"  {'Rank':<5} {'Descriptor':<30} {peak_hdr:<12} "
           f"{'Direction':<14} {'t at peak':<10}")
        _w("  " + "-" * 64)
        for rank, i in enumerate(ranked, 1):
            peak_val  = float(curves[np.argmax(np.abs(curves[:, i])), i])
            peak_abs  = abs(peak_val)
            direction = "none" if i in unreached else "positive" if peak_val > 0 else "negative"
            tau_pk    = float(taus[np.argmax(np.abs(curves[:, i]))])
            tau_txt   = f"{'-':<10}" if i in unreached else f"{tau_pk:<10.2f}"
            pk_txt    = f"{'-':<12}" if i in unreached else f"{peak_abs:<12.4f}"
            _w(f"  {rank:<5} {desc_names[i]:<30} {pk_txt} "
               f"{direction:<14} {tau_txt}")

        _w()
        _w(f"TYPE I CROSS-IMPACT MULTIPLIER  {lambda_hdr}")
        _w(f"{_F['Lambda']} = {_F['inverse']};  negative diagonal = cumulative self-response reverses sign")
        _w("-" * 68)
        _w(f"  {'Descriptor':<30} {lambda_hdr:<18} Note")
        _w("  " + "-" * 64)
        if lambda_diag is not None:
            for j, name in enumerate(desc_names):
                val  = float(lambda_diag[j])
                note = "cumulative self-response reverses sign" if val < 0 else ""
                _w(f"  {name:<30} {val:<18.4f} {note}")
        else:
            _w("  (could not compute: M is singular)")

        if uncertain_dirs:
            _w()
            _w("MC WARNING: uncertain response direction")
            _w("-" * 40)
            _w("  The following descriptors have MC bands that straddle zero at the peaks of their curves.")
            _w(f"  The direction of {_F['uncertain']} should not be reported as a firm conclusion:")
            for d in uncertain_dirs:
                _w(f"    - {d}")
            if unreached:
                _w("  Curves with direction none have no peak and are not tested.")
        elif len(unreached) == len(other_idx):
            _w()
            _w("MC: every curve has direction none, so no direction is tested.")
        else:
            _w()
            _w("MC: no band straddles zero at the peak of its curve, so every tested direction is robust.")
            if unreached:
                _w("  Curves with direction none have no peak and are not tested.")

        _w()
        _w("NOTE: the t axis is dimensionless model time, not calendar time.")
        _w("      Late-time ordering of responses is a structural property.")
        _w("      Early-peak order and absolute magnitudes depend on IO3_TARGET_RHO; do not compare")
        _w("      across studies that used a different rescaling target.")
        _w()
        _w(f"Plot saved to : {OUTPUT_FILE.name if OUTPUT_FILE else '(interactive)'}")
        _w("=" * 60)

    del _w
    print(f"Summary saved → {SUMMARY_FILE.name}")

print("\nDone.")
