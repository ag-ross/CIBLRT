"""
lrt.py — CIB Linear Response Theory implementation.

Models CIB descriptor dynamics near a consistent scenario as an OU SDE:

    dx(t) = M x(t) dt + dB(t),   M = W − I_N

yielding: W (effective cross-impact), M (drift), Σ (stationary covariance),
ρ(t₁) (susceptibility), ε̃ (implied perturbation), R(τ) (IRF), Λ (multiplier).

Stability: all Re(λ(M)) < 0 required; use `cib_lrt_rescaled` for IO-3 rescaling.

Dependencies: numpy, scipy (both required).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.linalg import expm as _expm
    from scipy.linalg import solve_continuous_lyapunov as _solve_lyapunov
    _USING_SCIPY_FALLBACK = False
except ImportError:                                                # pragma: no cover
    warnings.warn(
        "scipy not installed; using numpy-only _lrt_shim fallback.",
        RuntimeWarning, stacklevel=2,
    )
    from _lrt_shim.linalg import expm as _expm
    from _lrt_shim.linalg import solve_continuous_lyapunov as _solve_lyapunov
    _USING_SCIPY_FALLBACK = True


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StabilityReport:
    """Spectral diagnostics for the drift matrix M = W − I_N."""

    eigenvalues_M:     np.ndarray
    spectral_radius_W: float
    is_stable:         bool
    is_neumann:        bool
    slowest_mode:      float
    fastest_mode:      float

    def summary(self) -> str:
        """Return a formatted multi-line stability summary string."""
        slow_ts = 1.0 / max(self.slowest_mode, 1e-12)
        fast_ts = 1.0 / max(self.fastest_mode, 1e-12)
        lines = [
            f"  Stable  (all Re(λ(M)) < 0)      : {self.is_stable}",
            f"  Neumann (ρ(W) < 1)              : {self.is_neumann}  "
            f"[ρ(W) = {self.spectral_radius_W:.4f}]",
            f"  Slowest |Re(λ)|                 : {self.slowest_mode:.4f}  "
            f"(timescale ≈ {slow_ts:.2f})",
            f"  Fastest |Re(λ)|                 : {self.fastest_mode:.4f}  "
            f"(timescale ≈ {fast_ts:.2f})",
            "  Eigenvalues of M:",
        ]
        for ev in sorted(self.eigenvalues_M, key=lambda x: x.real):
            if abs(ev.imag) > 1e-10:
                lines.append(f"    {ev.real:+.4f} {ev.imag:+.4f}j")
            else:
                lines.append(f"    {ev.real:+.4f}")
        return "\n".join(lines)

    def recommended_tau_max(self, n_timescales: float = 5.0) -> float:
        """Return n_timescales / |Re(λ_slowest)| as a suggested τ_max."""
        if self.slowest_mode < 1e-12:
            return float("inf")
        return n_timescales / self.slowest_mode


@dataclass
class CIBLRTResult:
    """Full CIB-LRT output at one consistent scenario."""

    descriptor_names:                 List[str]
    scenario_states:                  Dict[str, str]
    W:                                np.ndarray
    M:                                np.ndarray
    stability:                        StabilityReport
    Sigma:                            Optional[np.ndarray]
    Lambda:                           Optional[np.ndarray]
    has_negative_self_multiplier:     bool = field(default=False, repr=False)
    negative_multiplier_descriptors:  List[str] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Core construction
# ---------------------------------------------------------------------------

def build_effective_matrix(
    matrix,   # CIBMatrix — type hint omitted to avoid circular import
    scenario, # Scenario
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract the effective cross-impact matrix W at a consistent scenario.

    W[j, i] = C_{ij}(z*_i, z*_j) for i ≠ j, W[j, j] = 0.  Convention:
    column i influences row j, so M = W − I acts from the left on the state vector.
    """
    desc_names = list(matrix.descriptors.keys())
    N = len(desc_names)
    W = np.zeros((N, N))

    for j, tgt_desc in enumerate(desc_names):
        tgt_state = scenario.get_state(tgt_desc)
        for i, src_desc in enumerate(desc_names):
            if i == j:
                continue  # CIB has no self-impacts; diagonal stays zero
            src_state = scenario.get_state(src_desc)
            W[j, i] = matrix.get_impact(src_desc, src_state, tgt_desc, tgt_state)

    return W, desc_names


#: Stability tolerance: Re(λ(M)) < -STABILITY_TOLERANCE is required.
STABILITY_TOLERANCE: float = 1.0e-10


def _stability_report(W: np.ndarray, M: np.ndarray) -> StabilityReport:
    """Compute spectral diagnostics for W and M = W − I."""
    eig_M  = np.linalg.eigvals(M)
    eig_W  = np.linalg.eigvals(W)
    real_M = eig_M.real
    decay  = np.abs(real_M)
    rho_W  = float(np.max(np.abs(eig_W)))

    return StabilityReport(
        eigenvalues_M     = eig_M,
        spectral_radius_W = rho_W,
        is_stable         = bool(np.all(real_M < -STABILITY_TOLERANCE)),
        is_neumann        = rho_W < 1.0,
        slowest_mode      = float(np.min(decay)),
        fastest_mode      = float(np.max(decay)),
    )


def _assemble_lrt_result(
    W:          np.ndarray,
    desc_names: List[str],
    scenario,
    *,
    warn_unstable: bool = True,
) -> CIBLRTResult:
    """Build Σ, Λ, and stability flags from (W, desc_names, scenario)."""
    N = len(desc_names)
    M = W - np.eye(N)

    stab = _stability_report(W, M)

    Sigma:    Optional[np.ndarray] = None
    Lambda:   Optional[np.ndarray] = None
    has_neg:  bool      = False
    neg_desc: List[str] = []

    if stab.is_stable:
        Sigma    = _solve_lyapunov(M, -np.eye(N))
        Lambda   = np.linalg.solve(M.T, -np.eye(N))
        neg_desc = [desc_names[j] for j in range(N) if Lambda[j, j] < 0]
        has_neg  = len(neg_desc) > 0
    elif warn_unstable:
        warnings.warn(
            f"Drift matrix M is not stable: not all Re(λ(M)) < 0.  "
            f"ρ(W) = {stab.spectral_radius_W:.4f}.  "
            "Σ, ρ(t₁), IRF, and Λ are undefined for an unstable scenario.  "
            "Use cib_lrt_rescaled() to apply IO-3 global rescaling.",
            RuntimeWarning,
            stacklevel=3,
        )

    return CIBLRTResult(
        descriptor_names                = desc_names,
        scenario_states                 = scenario.to_dict(),
        W                               = W,
        M                               = M,
        stability                       = stab,
        Sigma                           = Sigma,
        Lambda                          = Lambda,
        has_negative_self_multiplier    = has_neg,
        negative_multiplier_descriptors = neg_desc,
    )


def cib_lrt(matrix, scenario) -> CIBLRTResult:
    """
    Compute core CIB-LRT objects (W, M, Σ, Λ) at a consistent scenario.

    If M is not stable, issues a RuntimeWarning and returns Σ = Λ = None.
    Use `cib_lrt_rescaled` for automatic IO-3 rescaling.
    """
    W, desc_names = build_effective_matrix(matrix, scenario)
    return _assemble_lrt_result(W, desc_names, scenario, warn_unstable=True)


# ---------------------------------------------------------------------------
# IO-3 rescaling for stability
# ---------------------------------------------------------------------------

#: Recommended IO-3 rescaling band; warn outside.  None disables the warning.
TARGET_RHO_RECOMMENDED_RANGE: Optional[Tuple[float, float]] = (0.5, 0.99)


def _check_target_rho(target_rho: float) -> None:
    """Validate target_rho ∈ (0, 1) and warn outside the recommended band."""
    if not (0.0 < target_rho < 1.0):
        raise ValueError(
            f"target_rho must be in the open interval (0, 1); got {target_rho}."
        )
    if TARGET_RHO_RECOMMENDED_RANGE is not None:
        lo, hi = TARGET_RHO_RECOMMENDED_RANGE
        if not (lo <= target_rho <= hi):
            warnings.warn(
                f"target_rho={target_rho} outside recommended [{lo}, {hi}].",
                RuntimeWarning, stacklevel=3,
            )


def io3_rescale_for_stability(
    W: np.ndarray,
    target_rho: float = 0.9,
) -> Tuple[np.ndarray, float]:
    """
    Apply IO-3 global rescaling W → αW, with α = target_rho / ρ(W).

    Ensures ρ(αW) = target_rho < 1, restoring stability while preserving
    all relative magnitudes and consistent scenarios.  Raises ValueError if
    target_rho ∉ (0, 1); warns if outside the recommended band.
    """
    _check_target_rho(target_rho)
    rho_W = float(np.max(np.abs(np.linalg.eigvals(W))))
    if rho_W < 1e-12:
        return W.copy(), 1.0
    alpha = target_rho / rho_W
    return W * alpha, alpha


def cib_lrt_rescaled(
    matrix,
    scenario,
    target_rho: float = 0.9,
) -> Tuple[CIBLRTResult, float]:
    """
    CIB-LRT with automatic IO-3 rescaling if W is not stable.

    Returns (result, alpha); alpha = 1.0 if no rescaling was needed.
    """
    _check_target_rho(target_rho)
    W_raw, desc_names = build_effective_matrix(matrix, scenario)
    N = len(desc_names)

    stab_raw = _stability_report(W_raw, W_raw - np.eye(N))
    if stab_raw.is_stable:
        return cib_lrt(matrix, scenario), 1.0

    alpha    = target_rho / stab_raw.spectral_radius_W
    W_scaled = W_raw * alpha
    result   = _assemble_lrt_result(
        W_scaled, desc_names, scenario, warn_unstable=False,
    )
    return result, alpha

# ---------------------------------------------------------------------------
# LRT analytical functions
# ---------------------------------------------------------------------------

def susceptibility_matrix(M: np.ndarray, t1: float) -> np.ndarray:
    """
    CIB susceptibility matrix ρ(t₁) = M⁻ᵀ(exp(Mᵀt₁) − I_N).

    Solved as Mᵀ X = exp(Mᵀt₁) − I_N.  Asymptotic: ρ → t₁ I as t₁ → 0,
    ρ → Λ as t₁ → ∞.  Raises RuntimeError if M is not stable.
    """
    if not np.all(np.linalg.eigvals(M).real < 0):
        raise RuntimeError(
            "susceptibility_matrix requires a stable drift matrix "
            "(all Re(λ(M)) < 0).  Apply IO-3 rescaling first via "
            "cib_lrt_rescaled()."
        )
    N   = M.shape[0]
    Mt  = M.T
    return np.linalg.solve(Mt, _expm(Mt * t1) - np.eye(N))


def implied_perturbation(
    M:       np.ndarray,
    delta_z: np.ndarray,
    t1:      float,
) -> np.ndarray:
    """
    Implied constant perturbation ε̃ = (exp(Mᵀt₁) − I)⁻¹ Mᵀ Δz.

    Unique forcing over [0, t₁] that produces displacement Δz under linearised
    CIB dynamics.  Raises RuntimeError if M is not stable.
    """
    if not np.all(np.linalg.eigvals(M).real < 0):
        raise RuntimeError(
            "implied_perturbation requires a stable drift matrix "
            "(all Re(λ(M)) < 0).  Apply IO-3 rescaling first via "
            "cib_lrt_rescaled()."
        )
    N  = M.shape[0]
    Mt = M.T
    return np.linalg.solve(_expm(Mt * t1) - np.eye(N), Mt @ delta_z)


def impulse_response_curves(
    M:       np.ndarray,
    epsilon: np.ndarray,
    tau_max: float,
    dt:      float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IRF R(τ) = exp(Mᵀτ) ε̃ on [0, τ_max] via the semigroup property.

    Returns (taus, curves) where curves[k] = R(taus[k]); curves[0] = ε̃.

    Issues a RuntimeWarning if M is not stable (any Re(λ(M)) ≥ 0): the IRF
    will diverge rather than decay, and the output will not be meaningful.
    """
    if not np.all(np.linalg.eigvals(M).real < 0):
        warnings.warn(
            "impulse_response_curves: drift matrix M is not stable "
            "(not all Re(λ(M)) < 0).  The IRF will diverge; output is not "
            "meaningful.  Apply IO-3 rescaling via cib_lrt_rescaled() first.",
            RuntimeWarning,
            stacklevel=2,
        )
    taus    = np.arange(0.0, tau_max + dt / 2.0, dt)
    T       = len(taus)
    N       = M.shape[0]
    exp_dt  = _expm(M.T * dt)
    curves  = np.zeros((T, N))
    curves[0] = epsilon
    for k in range(1, T):
        curves[k] = exp_dt @ curves[k - 1]
    return taus, curves


def scenario_prediction(
    M:       np.ndarray,
    delta_z: np.ndarray,
    t1:      float,
    t2:      float,
) -> np.ndarray:
    """
    Predicted displacement at t₂: M⁻ᵀ(exp(Mᵀt₂) − I)(exp(Mᵀt₁) − I)⁻¹ Mᵀ Δz.

    Calibrates ε̃ at t₁ then propagates to the forecast horizon t₂.
    Raises RuntimeError if M is not stable.
    """
    if not np.all(np.linalg.eigvals(M).real < 0):
        raise RuntimeError(
            "scenario_prediction requires a stable drift matrix "
            "(all Re(λ(M)) < 0).  Apply IO-3 rescaling first via "
            "cib_lrt_rescaled()."
        )
    N   = M.shape[0]
    Mt  = M.T
    eps = implied_perturbation(M, delta_z, t1)
    return np.linalg.solve(Mt, (_expm(Mt * t2) - np.eye(N)) @ eps)


def perturbation_budget(
    M:       np.ndarray,
    delta_z: np.ndarray,
    t1:      float,
) -> float:
    """‖ε̃‖₂ — network-weighted scalar distance (asymmetric: M is the origin's drift).

    Raises RuntimeError if M is not stable.
    """
    if not np.all(np.linalg.eigvals(M).real < 0):
        raise RuntimeError(
            "perturbation_budget requires a stable drift matrix "
            "(all Re(λ(M)) < 0).  Apply IO-3 rescaling first via "
            "cib_lrt_rescaled()."
        )
    return float(np.linalg.norm(implied_perturbation(M, delta_z, t1)))


# ---------------------------------------------------------------------------
# Convenience: Δz construction from two scenario dicts
# ---------------------------------------------------------------------------

def delta_z_from_scenarios(
    desc_names:    List[str],
    state_order:   Dict[str, List[str]],
    scenario_from: Dict[str, str],
    scenario_to:   Dict[str, str],
) -> np.ndarray:
    """Activation displacement Δz = z_to − z_from in ordinal state-index space."""
    delta = np.zeros(len(desc_names))
    for k, d in enumerate(desc_names):
        if d not in state_order:
            raise KeyError(
                f"delta_z_from_scenarios: descriptor {d!r} not found in state_order."
            )
        order = state_order[d]
        for label, sc_name in ((scenario_from.get(d), "scenario_from"),
                               (scenario_to.get(d),   "scenario_to")):
            if label is None:
                raise KeyError(
                    f"delta_z_from_scenarios: descriptor {d!r} missing from {sc_name}."
                )
            if label not in order:
                raise ValueError(
                    f"delta_z_from_scenarios: state {label!r} for descriptor {d!r} "
                    f"not in state_order[{d!r}] = {order!r} (from {sc_name})."
                )
        delta[k] = order.index(scenario_to[d]) - order.index(scenario_from[d])
    return delta
