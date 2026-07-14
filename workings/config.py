"""config.py — shared parameters for the CIB-LRT pipeline."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Core LRT parameters
# ---------------------------------------------------------------------------
IO3_TARGET:       float = 0.9
T1_IRF:           float = 1.0
DT:               float = 0.01    # fine step for deterministic curves
DT_CSV:           float = 0.1     # coarser step for MC ribbon / CSV
TAU_N_TIMESCALES: float = 5.0
TAU_MIN:          float = 20.0

# Susceptibility horizons cached by run_analysis.py.
T1_VALUES: list[float] = [1.0, 5.0]


# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------
SENSITIVITY_RHOS: list[float] = [round(v, 4) for v in np.linspace(0.50, 0.99, 50).tolist()]
T1_SWEEP:         list[float] = list(np.round(np.linspace(0.5, 10.0, 20), 4))
MC_SIGMA_SWEEP:   list[float] = [round(s, 2) for s in np.linspace(0.1, 1.0, 10)]


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
N_MC:          int = 10_000   # samples per Fig 1 / Fig 7 / Fig 6 σ-point
MC_SIGMA:      float = 0.5    # half an ordinal level per CIM entry
MC_SIGMA_N:    int = 10_000   # samples per σ in Fig 6 sweep
MC_RNG_SEED:   int = 99

# Per-figure seeds.  MC_RNG_SEED_SIGMA_BASE + sig_idx gives each σ index an
# independent seed across the Fig 6 sweep.
MC_RNG_SEED_IRF:          int = MC_RNG_SEED
MC_RNG_SEED_UNIT_IMPULSE: int = MC_RNG_SEED
MC_RNG_SEED_SIGMA_BASE:   int = MC_RNG_SEED + 100

CI_LO: int = 5
CI_HI: int = 95
