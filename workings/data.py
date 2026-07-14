"""
data.py — Load Phase_D_CIM.csv and build a CIBMatrix for CIB-LRT analysis.

Dataset-specific: STATE_ORDER_BY_DESCRIPTOR is bound to the Phase_D CIM and
must be updated for any other elicitation.  lrt.py itself is dataset-agnostic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cib.core import CIBMatrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid integer impact range (enforced at CSV load only; MC may exceed it).
IMPACT_CLIP: Tuple[float, float] = (-3.0, 3.0)

#: Default CSV path (Phase_D_CIM.csv beside this module).
DEFAULT_CIM_CSV: Path = Path(__file__).resolve().parent / "Phase_D_CIM.csv"

#: Canonical ordinal state ordering for each descriptor.
#: States are listed low → high (index 0, 1, 2).
STATE_ORDER_BY_DESCRIPTOR: Dict[str, List[str]] = {
    "Decarbonisation_Outcome": ["High emissions", "Medium", "Net-zero aligned"],
    "EU_Policy_Alignment":     ["Low", "Medium", "High"],
    "Electrification_Pace":    ["Slow", "Medium", "High"],
    "Energy_Security_Pressure":["Low", "Medium", "High"],
    "Fossil_Price_Pressure":   ["Low", "Medium", "High"],
    "Grid_Development":        ["Lagging", "Moderate", "Strong"],
    "Hydrogen_Role":           ["Limited", "Moderate", "Major"],
    "Industrial_Energy_Demand":["Low", "Medium", "High"],
    "Investment_Availability": ["Low", "Medium", "High"],
    "Land_Use_Conflict":       ["Low", "Medium", "High"],
    "Permitting_Pace":         ["Slow", "Medium", "Fast"],
    "Policy_Stringency":       ["Low", "Medium", "High"],
    "Public_Acceptance":       ["Low", "Medium", "High"],
    "Renewables_Deployment":   ["Low", "Medium", "High"],
    "Technology_Costs":        ["Low", "Medium", "High"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_score_agreed(raw: str, *, row_hint: str) -> float:
    """Parse CIM score as an integer within IMPACT_CLIP."""
    lo, hi = IMPACT_CLIP
    try:
        x = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Score_agreed ({raw!r}) {row_hint}") from exc
    rounded = round(x)
    if abs(x - rounded) > 1e-6:
        raise ValueError(f"Score_agreed must be integral ({raw!r}) {row_hint}")
    if rounded < lo or rounded > hi:
        raise ValueError(
            f"Score_agreed must be in [{lo:g}, {hi:g}] ({raw!r} -> {int(rounded)}) {row_hint}"
        )
    return float(rounded)


def _load_csv(
    csv_path: Path,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str, str, str], float]]:
    """
    Load descriptors and impact scores from a CIM CSV file.

    Returns
    -------
    descriptors : dict mapping descriptor name -> ordered list of state labels
    impacts     : dict mapping (src_desc, src_state, tgt_desc, tgt_state) -> score
    """
    src_states: Dict[str, set] = {}
    tgt_states: Dict[str, set] = {}
    impacts: Dict[Tuple[str, str, str, str], float] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        for lineno, row in enumerate(csv.DictReader(f), start=2):
            sd = row["Source descriptor"]
            td = row["Target descriptor"]
            if sd == td:
                continue
            ss, ts = row["Source_state"], row["Target_state"]
            src_states.setdefault(sd, set()).add(ss)
            tgt_states.setdefault(td, set()).add(ts)
            key = (sd, ss, td, ts)
            row_hint = f"at data row {lineno}"
            score = _parse_score_agreed(row["Score_agreed"], row_hint=row_hint)
            if key in impacts:
                if impacts[key] != score:
                    raise ValueError(
                        f"Duplicate cross-impact row {key!r} with conflicting score {row_hint}"
                    )
                continue
            impacts[key] = score

    csv_descriptors = sorted(set(src_states) | set(tgt_states))
    csv_set = set(csv_descriptors)
    mapped_set = set(STATE_ORDER_BY_DESCRIPTOR)
    missing = sorted(csv_set - mapped_set)
    if missing:
        raise ValueError(f"Descriptors missing from STATE_ORDER_BY_DESCRIPTOR: {missing!r}")
    extra = sorted(mapped_set - csv_set)
    if extra:
        raise ValueError(f"Descriptors in STATE_ORDER_BY_DESCRIPTOR but absent from CSV: {extra!r}")

    descriptors: Dict[str, List[str]] = {}
    for name in csv_descriptors:
        observed = src_states.get(name, set()) | tgt_states.get(name, set())
        ordered = STATE_ORDER_BY_DESCRIPTOR[name]
        if set(ordered) != observed:
            raise ValueError(
                f"State mismatch for {name!r}: expected {ordered!r}, "
                f"observed {sorted(observed)!r}"
            )
        descriptors[name] = list(ordered)

    # Completeness check: every ordered (src_desc, src_state, tgt_desc, tgt_state)
    # combination must be present.  A missing entry would silently produce a zero
    # in W, indistinguishable from a true "no influence" score of 0.
    missing_entries: List[str] = []
    for sd, sd_states in descriptors.items():
        for ss in sd_states:
            for td, td_states in descriptors.items():
                if td == sd:
                    continue
                for ts in td_states:
                    if (sd, ss, td, ts) not in impacts:
                        missing_entries.append(
                            f"  ({sd!r}, {ss!r}) → ({td!r}, {ts!r})"
                        )
    if missing_entries:
        raise ValueError(
            f"CIM CSV is incomplete: {len(missing_entries)} cross-impact "
            f"entries missing:\n" + "\n".join(missing_entries[:20])
            + ("\n  ... (truncated)" if len(missing_entries) > 20 else "")
        )

    return descriptors, impacts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_matrix(*, csv_path: Optional[Path] = None) -> CIBMatrix:
    """
    Build and return a deterministic CIBMatrix from the CIM CSV.

    Parameters
    ----------
    csv_path : path to the CSV file; defaults to Phase_D_CIM.csv
               beside this module.

    Returns
    -------
    CIBMatrix with all cross-impact entries loaded.
    """
    path = csv_path if csv_path is not None else DEFAULT_CIM_CSV
    descriptors, impacts = _load_csv(path)
    matrix = CIBMatrix(descriptors)
    matrix.set_impacts(impacts)
    return matrix


def state_index(descriptor: str, state_label: str) -> int:
    """
    Return the ordinal index (0, 1, 2) of a state label for a descriptor.

    This is the integer encoding used in the LRT: z_j = index of the
    active state of descriptor j.
    """
    ordered = STATE_ORDER_BY_DESCRIPTOR[descriptor]
    try:
        return ordered.index(state_label)
    except ValueError as exc:
        raise ValueError(
            f"State {state_label!r} not in ordering for descriptor {descriptor!r}: {ordered!r}"
        ) from exc


def scenario_to_indices(scenario_dict: Dict[str, str]) -> Dict[str, int]:
    """
    Convert a {descriptor: state_label} scenario dict to {descriptor: index}.

    Uses STATE_ORDER_BY_DESCRIPTOR for the ordinal encoding.
    """
    return {d: state_index(d, s) for d, s in scenario_dict.items()}
