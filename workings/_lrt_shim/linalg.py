"""Minimal numpy-only fallback for scipy.linalg's expm and Lyapunov solver."""

from __future__ import annotations

import numpy as np

__all__ = ["expm", "solve_continuous_lyapunov"]


def expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential via eigendecomposition, Taylor-series fallback."""
    A = np.asarray(A, dtype=complex)
    vals, vecs = np.linalg.eig(A)
    try:
        result = vecs @ np.diag(np.exp(vals)) @ np.linalg.inv(vecs)
    except np.linalg.LinAlgError:
        result = np.eye(A.shape[0], dtype=complex)
        term   = np.eye(A.shape[0], dtype=complex)
        for k in range(1, 21):
            term   = term @ A / k
            result = result + term

    real_scale = max(np.max(np.abs(result.real)), 1e-30)
    if np.max(np.abs(result.imag)) < 1e-10 * real_scale:
        return result.real
    return result


def solve_continuous_lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Solve A X + X A^T = Q via Kronecker vectorisation (O(n^4) memory)."""
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    n = A.shape[0]
    lhs   = np.kron(np.eye(n), A) + np.kron(A, np.eye(n))
    vec_X = np.linalg.solve(lhs, Q.ravel(order="F"))
    return vec_X.reshape(n, n, order="F")
