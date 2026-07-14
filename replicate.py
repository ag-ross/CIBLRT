"""
replicate.py -- one-command replication of all CIB-LRT paper results.

Runs the four workings/ pipeline scripts in dependency order and writes
all outputs to workings/outputs/.

  Stage 1  run_analysis.py     -- attractor enumeration + LRT objects (cache)
  Stage 2  plot_irf.py         -- figures FIG_01, FIG_03, FIG_04, FIG_S1-S3
  Stage 3  shock_descriptor.py -- unit-impulse shock figure FIG_02
  Stage 4  export_tables.py    -- paper tables as CSV

Usage
-----
  python replicate.py           # use cached attractor set if present
  python replicate.py --force   # recompute LRT objects from cached scenarios
                                #   (passes --force to run_analysis.py only;
                                #    attractor set itself is preserved when
                                #    ATTRACTOR_SEARCH_COMPLETE = True)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
WORKINGS = ROOT / "workings"

# Stages in dependency order.  (script, short description)
STAGES: list[tuple[str, str]] = [
    ("run_analysis.py",     "Attractor enumeration + LRT objects"),
    ("plot_irf.py",         "Figures FIG_01, FIG_03, FIG_04, FIG_S1-S3"),
    ("shock_descriptor.py", "Unit-impulse shock figure FIG_02"),
    ("export_tables.py",    "Paper tables -> CSV"),
]

STAGE_SEP   = "-" * 64
SECTION_SEP = "=" * 64


def _run_stage(
    script:     str,
    label:      str,
    stage_num:  int,
    n_stages:   int,
    extra_args: list[str],
) -> None:
    """Run one pipeline stage as a subprocess; exit on failure."""
    print(f"\n{STAGE_SEP}")
    print(f"  [{stage_num}/{n_stages}]  {label}")
    print(f"           {script}")
    print(STAGE_SEP)

    t0     = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(WORKINGS / script)] + extra_args,
        cwd=WORKINGS,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  FAILED (exit {result.returncode}) after {elapsed:.1f}s")
        print("  Fix the error above, then re-run replicate.py.")
        sys.exit(result.returncode)

    print(f"\n  Done in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full CIB-LRT paper replication pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "All outputs are written to workings/outputs/.\n"
            "Run  python workings/run_analysis.py  alone to inspect cached results."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Recompute LRT objects from the cached attractor set "
            "(passes --force to run_analysis.py; skips the PyCIB attractor search "
            "when ATTRACTOR_SEARCH_COMPLETE = True)."
        ),
    )
    args = parser.parse_args()

    print(SECTION_SEP)
    print("  CIB-LRT replication pipeline")
    print(f"  {len(STAGES)} stages -> workings/outputs/")
    print(SECTION_SEP)

    t_start = time.perf_counter()

    for i, (script, label) in enumerate(STAGES, 1):
        extra = ["--force"] if args.force and script == "run_analysis.py" else []
        _run_stage(script, label, i, len(STAGES), extra)

    total = time.perf_counter() - t_start
    print(f"\n{SECTION_SEP}")
    print(f"  All {len(STAGES)} stages complete in {total:.1f}s")
    print("  Outputs written to workings/outputs/")
    print(SECTION_SEP)


if __name__ == "__main__":
    main()
