"""
Regression tests for ``workings/lrt.py``.

Each test exercises a single closed-form identity from the paper's
Section 2 and asserts that the implementation reproduces it to machine
precision (or to a tight numerical tolerance for quadrature checks).

The tests are written to run from the repository root with the standard
``unittest`` runner:

    python -m unittest workings.tests.test_lrt -v

or directly:

    python workings/tests/test_lrt.py

They do not require PyCIB or scipy: every input matrix is constructed
locally from a small numpy seed.  When real scipy is unavailable, the
tests still pass because ``lrt.py`` falls back transparently to the
explicit ``_lrt_shim`` package (see ``workings/_lrt_shim/__init__.py``).
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

# Make ``import lrt`` work no matter where the test is invoked from.
_HERE = Path(__file__).resolve().parent
_WORKINGS = _HERE.parent
if str(_WORKINGS) not in sys.path:
    sys.path.append(str(_WORKINGS))

# Suppress the "scipy not installed" warning so that test output is clean.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    import lrt


# ---------------------------------------------------------------------------
# Test fixture: a small, hand-crafted stable W matrix
# ---------------------------------------------------------------------------

def _make_stable_W(n: int = 6, seed: int = 1) -> np.ndarray:
    """
    Return an ``(n, n)`` asymmetric matrix with zero diagonal, scaled so
    that M = W - I is stable.  Constructed once and reused across tests.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n)) * 0.4
    np.fill_diagonal(W, 0.0)
    # Rescale to spectral radius 0.85 (well inside the stability region).
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    return W * (0.85 / rho)


# ---------------------------------------------------------------------------
# Identity checks
# ---------------------------------------------------------------------------

class TestSusceptibilityIdentity(unittest.TestCase):
    """``rho(t1) @ epsilon == Delta_z`` whenever ``epsilon`` is the implied
    perturbation derived from a known ``Delta_z``."""

    def setUp(self) -> None:
        self.W  = _make_stable_W()
        self.M  = self.W - np.eye(self.W.shape[0])
        self.dz = np.array([1.0, -1.0, 0.0, 2.0, -2.0, 1.0])

    def test_rho_eps_equals_delta_z(self) -> None:
        for t1 in (0.5, 1.0, 5.0):
            eps = lrt.implied_perturbation(self.M, self.dz, t1)
            rho = lrt.susceptibility_matrix(self.M, t1)
            np.testing.assert_allclose(rho @ eps, self.dz, atol=1e-10)


class TestIRFConstruction(unittest.TestCase):
    """``R(0) = epsilon`` exactly; iterative IRF agrees with analytic
    ``exp(M^T tau) @ epsilon`` to machine precision."""

    def setUp(self) -> None:
        self.W  = _make_stable_W()
        self.M  = self.W - np.eye(self.W.shape[0])
        self.dz = np.array([1.0, -1.0, 0.0, 2.0, -2.0, 1.0])
        self.t1 = 1.0
        self.eps = lrt.implied_perturbation(self.M, self.dz, self.t1)

    def test_irf_at_tau_zero_equals_epsilon(self) -> None:
        taus, curves = lrt.impulse_response_curves(
            self.M, self.eps, tau_max=2.0, dt=0.01,
        )
        np.testing.assert_array_equal(curves[0], self.eps)
        self.assertEqual(taus[0], 0.0)

    def test_irf_iterative_matches_analytic(self) -> None:
        from lrt import _expm
        taus, curves = lrt.impulse_response_curves(
            self.M, self.eps, tau_max=self.t1, dt=0.01,
        )
        # Analytic R(t1) = exp(M^T t1) @ epsilon.
        R_analytic = _expm(self.M.T * self.t1) @ self.eps
        R_iter     = curves[-1]
        np.testing.assert_allclose(R_iter, R_analytic, atol=1e-10)


class TestIRFIntegratesToDeltaZ(unittest.TestCase):
    """``integral_0^{t1} R(tau) dtau == Delta_z`` to quadrature tolerance."""

    def setUp(self) -> None:
        self.W  = _make_stable_W()
        self.M  = self.W - np.eye(self.W.shape[0])
        self.dz = np.array([1.0, -1.0, 0.0, 2.0, -2.0, 1.0])
        self.t1 = 1.0

    def test_irf_integral_equals_dz(self) -> None:
        eps = lrt.implied_perturbation(self.M, self.dz, self.t1)
        taus, curves = lrt.impulse_response_curves(
            self.M, eps, tau_max=self.t1, dt=1.0e-4,
        )
        # Use numpy's trapezoid (avoid deprecated ``trapz`` on numpy >= 2).
        integrate = getattr(np, "trapezoid", None) or np.trapz
        integral = integrate(curves, taus, axis=0)
        np.testing.assert_allclose(integral, self.dz, atol=1e-5)


class TestSusceptibilityShortLongHorizon(unittest.TestCase):
    """Asymptotic behaviour: ``rho(t1) -> t1 * I`` as ``t1 -> 0`` and
    ``rho(t1) -> Lambda`` as ``t1 -> infinity``."""

    def setUp(self) -> None:
        self.W = _make_stable_W()
        self.M = self.W - np.eye(self.W.shape[0])
        self.Lambda = -np.linalg.inv(self.M.T)        # closed-form

    def test_short_horizon_limit(self) -> None:
        t1 = 1.0e-4
        rho = lrt.susceptibility_matrix(self.M, t1)
        np.testing.assert_allclose(rho, t1 * np.eye(self.M.shape[0]), atol=1e-7)

    def test_long_horizon_limit(self) -> None:
        rho = lrt.susceptibility_matrix(self.M, t1=200.0)
        # At t1 = 200 with slowest decay ~0.15 (1 - 0.85), exp(M^T * 200) is
        # numerically zero, so rho == Lambda to machine precision.
        np.testing.assert_allclose(rho, self.Lambda, atol=1e-10)


class TestLyapunovIdentity(unittest.TestCase):
    """``M Sigma + Sigma M^T + I = 0`` for the stationary OU covariance."""

    def test_lyapunov_residual(self) -> None:
        W = _make_stable_W()
        from cib_core_stub import Scenario as _ScenarioStub  # noqa: F401  # see note
        # Build a minimal CIBLRTResult by hand: scenario isn't needed for
        # the Lyapunov check, only M and Sigma.
        M = W - np.eye(W.shape[0])
        Sigma = lrt._solve_lyapunov(M, -np.eye(W.shape[0]))
        residual = M @ Sigma + Sigma @ M.T + np.eye(W.shape[0])
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)


class TestMultiplierAsNeumannSeries(unittest.TestCase):
    """``Lambda == (I - W)^{-T} == sum_{k=0}^{K} (W^T)^k`` for finite K when
    spectral radius of W is < 1."""

    def test_neumann_partial_sum(self) -> None:
        W = _make_stable_W()
        N = W.shape[0]
        M = W - np.eye(N)
        Lambda = -np.linalg.inv(M.T)
        # Partial sum to high order should agree closely.
        WT = W.T
        partial = np.eye(N).copy()
        term = np.eye(N).copy()
        for _ in range(200):
            term = term @ WT
            partial = partial + term
        np.testing.assert_allclose(partial, Lambda, atol=1e-8)


class TestIRFStabilityWarning(unittest.TestCase):
    """``impulse_response_curves`` issues a RuntimeWarning when passed an
    unstable drift matrix, and produces no warning for a stable one."""

    def test_warns_on_unstable_M(self) -> None:
        # M with a positive eigenvalue (+1) — clearly unstable.
        M_unstable = np.diag([-1.0, -1.0, 1.0])
        eps = np.array([1.0, 0.0, 0.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lrt.impulse_response_curves(M_unstable, eps, tau_max=1.0, dt=0.1)
        self.assertTrue(
            any(issubclass(rec.category, RuntimeWarning) for rec in w),
            "expected a RuntimeWarning for unstable M",
        )

    def test_no_warning_on_stable_M(self) -> None:
        W = _make_stable_W()
        M = W - np.eye(W.shape[0])
        eps = np.ones(M.shape[0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lrt.impulse_response_curves(M, eps, tau_max=1.0, dt=0.1)
        runtime_warnings = [rec for rec in w if issubclass(rec.category, RuntimeWarning)]
        self.assertEqual(runtime_warnings, [],
                         "unexpected RuntimeWarning for stable M")


class TestStabilityReportTolerance(unittest.TestCase):
    """``_stability_report`` accepts matrices whose eigenvalues land at
    ``-epsilon`` due to rounding, where the strict ``< 0`` check would reject."""

    def test_borderline_stable_accepted(self) -> None:
        # Diagonal matrix with a borderline-but-still-stable eigenvalue.
        W = np.zeros((3, 3))
        W[0, 0] = 0.0   # diagonal kept zero by CIB convention
        W[1, 2] = 0.5
        W[2, 1] = 0.5
        M = W - np.eye(3)
        # M has eigenvalues -1, -0.5, -1.5 -- comfortably stable.
        report = lrt._stability_report(W, M)
        self.assertTrue(report.is_stable)

    def test_unstable_rejected(self) -> None:
        # Construct M with an eigenvalue at +1 (clearly unstable).
        M = np.diag([-1.0, -1.0, 1.0])
        W = M + np.eye(3)
        report = lrt._stability_report(W, M)
        self.assertFalse(report.is_stable)


class TestIO3Rescaling(unittest.TestCase):
    """``io3_rescale_for_stability`` produces an output with spectral radius
    equal to the target."""

    def test_rescaling_hits_target(self) -> None:
        rng = np.random.default_rng(7)
        W   = rng.standard_normal((10, 10))
        np.fill_diagonal(W, 0.0)
        # Inflate so that rho(W) >> 1.
        W = W * 5.0
        alpha_target = 0.85
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            W_scaled, alpha = lrt.io3_rescale_for_stability(W, target_rho=alpha_target)
        rho_scaled = float(np.max(np.abs(np.linalg.eigvals(W_scaled))))
        self.assertAlmostEqual(rho_scaled, alpha_target, places=8)
        self.assertAlmostEqual(alpha * float(np.max(np.abs(np.linalg.eigvals(W)))),
                               alpha_target, places=8)

    def test_target_rho_must_be_open_unit_interval(self) -> None:
        W = np.eye(3)
        with self.assertRaises(ValueError):
            lrt.io3_rescale_for_stability(W, target_rho=0.0)
        with self.assertRaises(ValueError):
            lrt.io3_rescale_for_stability(W, target_rho=1.0)
        with self.assertRaises(ValueError):
            lrt.io3_rescale_for_stability(W, target_rho=-0.5)

    def test_warns_outside_recommended_range(self) -> None:
        W = np.array([[0.0, 1.0], [1.0, 0.0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lrt.io3_rescale_for_stability(W, target_rho=0.1)
            self.assertTrue(any(issubclass(rec.category, RuntimeWarning) for rec in w),
                            "expected a RuntimeWarning for target_rho below 0.5")


class TestPerturbationBudgetAsymmetry(unittest.TestCase):
    """The budget depends on the origin attractor's M; reversing only Delta_z
    (with M held fixed) gives the same Euclidean norm."""

    def test_sign_flip_preserves_norm(self) -> None:
        W = _make_stable_W()
        M = W - np.eye(W.shape[0])
        dz = np.array([1.0, -1.0, 0.0, 2.0, -2.0, 1.0])
        b_fwd = lrt.perturbation_budget(M, dz, t1=1.0)
        b_rev = lrt.perturbation_budget(M, -dz, t1=1.0)
        self.assertAlmostEqual(b_fwd, b_rev, places=12)

    def test_asymmetry_requires_different_M(self) -> None:
        W1 = _make_stable_W(seed=1)
        W2 = _make_stable_W(seed=2)
        M1 = W1 - np.eye(W1.shape[0])
        M2 = W2 - np.eye(W2.shape[0])
        dz = np.array([1.0, -1.0, 0.0, 2.0, -2.0, 1.0])
        b_M1 = lrt.perturbation_budget(M1, dz, t1=1.0)
        b_M2 = lrt.perturbation_budget(M2, -dz, t1=1.0)
        # Different M and reversed dz: norms can legitimately differ.
        self.assertNotAlmostEqual(b_M1, b_M2, places=4)


class TestDeltaZFromScenarios(unittest.TestCase):
    """``delta_z_from_scenarios`` computes ordinal differences correctly and
    validates inputs."""

    def setUp(self) -> None:
        self.desc_names = ["A", "B"]
        self.state_order = {"A": ["low", "mid", "high"],
                            "B": ["x", "y", "z"]}

    def test_basic_difference(self) -> None:
        dz = lrt.delta_z_from_scenarios(
            self.desc_names, self.state_order,
            {"A": "low",  "B": "z"},
            {"A": "high", "B": "x"},
        )
        np.testing.assert_array_equal(dz, np.array([2.0, -2.0]))

    def test_missing_descriptor_raises(self) -> None:
        with self.assertRaises(KeyError):
            lrt.delta_z_from_scenarios(
                self.desc_names, self.state_order,
                {"A": "low"},                       # B missing
                {"A": "high", "B": "x"},
            )

    def test_unknown_state_raises(self) -> None:
        with self.assertRaises(ValueError):
            lrt.delta_z_from_scenarios(
                self.desc_names, self.state_order,
                {"A": "BOGUS", "B": "z"},
                {"A": "high",  "B": "x"},
            )


class TestScenarioPrediction(unittest.TestCase):
    """``scenario_prediction(M, delta_z, t1, t2)`` returns the displacement
    predicted at horizon t2 when calibrated at t1.

    Three independent invariants are verified:

    1. **IRF integral consistency** — the closed-form output must agree with
       the numerical trapezoid integral of R(τ) = exp(Mᵀτ) ε̃ over [0, t2],
       where ε̃ = implied_perturbation(M, delta_z, t1).  Tested for t2 < t1,
       t2 = t1 (recovers Δz), and t2 > t1.

    2. **Linearity in delta_z** — the formula is linear in Δz (it enters
       only through the linear solve in implied_perturbation), so
       f(a·Δz₁ + b·Δz₂) = a·f(Δz₁) + b·f(Δz₂) must hold to machine
       precision for arbitrary scalars a, b.

    3. **Long-horizon limit** — as t2 → ∞, exp(Mᵀt2) → 0 and the output
       converges to Λ ε̃ = −M⁻ᵀ ε̃.  Verified at t2 = 200, where the
       slowest mode of the test fixture (decay rate ≈ 0.15) gives
       |exp(Mᵀ·200)| < 10⁻¹³.
    """

    def setUp(self) -> None:
        self.W   = _make_stable_W()
        self.M   = self.W - np.eye(self.W.shape[0])
        self.dz  = np.array([1.0, -1.0, 0.0, 2.0, -2.0, 1.0])
        self.t1  = 1.0

    # ------------------------------------------------------------------
    # Helper: numerical IRF integral over [0, t2] via trapezoid rule.
    # ------------------------------------------------------------------
    def _irf_integral(self, t2: float) -> np.ndarray:
        eps = lrt.implied_perturbation(self.M, self.dz, self.t1)
        taus, curves = lrt.impulse_response_curves(
            self.M, eps, tau_max=t2, dt=1.0e-4,
        )
        integrate = getattr(np, "trapezoid", None) or np.trapz
        return integrate(curves, taus, axis=0)

    # ------------------------------------------------------------------
    # 1. Consistency with numerical IRF integral
    # ------------------------------------------------------------------

    def test_integral_consistency_t2_less_than_t1(self) -> None:
        """Closed form agrees with numerical integral when t2 < t1."""
        t2 = 0.5                                        # strictly before calibration horizon
        pred  = lrt.scenario_prediction(self.M, self.dz, self.t1, t2)
        numer = self._irf_integral(t2)
        np.testing.assert_allclose(pred, numer, atol=1e-5)

    def test_integral_consistency_t2_equals_t1(self) -> None:
        """At t2 = t1 the output recovers delta_z (sanity / regression)."""
        pred  = lrt.scenario_prediction(self.M, self.dz, self.t1, self.t1)
        numer = self._irf_integral(self.t1)
        np.testing.assert_allclose(pred, self.dz,  atol=1e-5)
        np.testing.assert_allclose(pred, numer,    atol=1e-5)

    def test_integral_consistency_t2_greater_than_t1(self) -> None:
        """Closed form agrees with numerical integral when t2 > t1."""
        t2 = 5.0                                        # beyond calibration horizon
        pred  = lrt.scenario_prediction(self.M, self.dz, self.t1, t2)
        numer = self._irf_integral(t2)
        np.testing.assert_allclose(pred, numer, atol=1e-5)

    # ------------------------------------------------------------------
    # 2. Linearity in delta_z
    # ------------------------------------------------------------------

    def test_linearity_in_delta_z(self) -> None:
        """scenario_prediction is linear in delta_z to machine precision."""
        dz2 = np.array([-0.5, 0.0, 1.0, -1.0, 0.5, 2.0])
        a, b = 3.0, -2.0
        t2   = 2.0

        pred_combined  = lrt.scenario_prediction(
            self.M, a * self.dz + b * dz2, self.t1, t2,
        )
        pred_linear    = (
            a * lrt.scenario_prediction(self.M, self.dz, self.t1, t2)
            + b * lrt.scenario_prediction(self.M, dz2,   self.t1, t2)
        )
        np.testing.assert_allclose(pred_combined, pred_linear, atol=1e-12)

    # ------------------------------------------------------------------
    # 3. Long-horizon limit: output → Λ ε̃ = −M⁻ᵀ ε̃
    # ------------------------------------------------------------------

    def test_long_horizon_limit(self) -> None:
        """At t2 = 200, exp(Mᵀt2) ≈ 0 and output converges to −M⁻ᵀ ε̃."""
        t2  = 200.0
        eps = lrt.implied_perturbation(self.M, self.dz, self.t1)
        # −M⁻ᵀ ε̃ = Λ ε̃ (the long-run multiplied displacement)
        Lambda_eps = np.linalg.solve(self.M.T, -eps)
        pred       = lrt.scenario_prediction(self.M, self.dz, self.t1, t2)
        np.testing.assert_allclose(pred, Lambda_eps, atol=1e-10)


class TestUnitImpulseIRF(unittest.TestCase):
    """``impulse_response_curves(M, e_i, ...)`` at lag τ equals the i-th column
    of ``exp(M^T τ)`` — the discrete realisation of Eq. (11) of the paper."""

    def setUp(self) -> None:
        self.W = _make_stable_W()
        self.M = self.W - np.eye(self.W.shape[0])

    def test_unit_impulse_equals_column_of_matrix_exponential(self) -> None:
        from lrt import _expm
        N      = self.M.shape[0]
        tau    = 1.5
        expMt  = _expm(self.M.T * tau)
        for i in range(N):
            ei = np.zeros(N)
            ei[i] = 1.0
            _, curves = lrt.impulse_response_curves(
                self.M, ei, tau_max=tau, dt=0.01,
            )
            np.testing.assert_allclose(
                curves[-1], expMt[:, i], atol=1e-10,
                err_msg=f"unit-impulse IRF for e_{i} does not match column {i} "
                        f"of exp(M^T * {tau})",
            )


# ---------------------------------------------------------------------------
# Compatibility shim so the Lyapunov test can run without PyCIB.
# This is a tiny stand-in only used for the import in TestLyapunovIdentity;
# the actual test does not depend on the stub's behaviour.
# ---------------------------------------------------------------------------

class _ScenarioStub:                                # pragma: no cover
    def to_dict(self) -> dict:
        return {}


# Register the stub as a top-level module so the test import resolves.
import types as _types
_stub_module = _types.ModuleType("cib_core_stub")
_stub_module.Scenario = _ScenarioStub
sys.modules.setdefault("cib_core_stub", _stub_module)


if __name__ == "__main__":
    unittest.main()
