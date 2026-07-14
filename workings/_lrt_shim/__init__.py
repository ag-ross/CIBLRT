"""
Explicit numpy-only fallback for scipy.linalg.expm and solve_continuous_lyapunov.

Imported by lrt.py only when scipy is not installed.  Accurate to machine
precision for well-conditioned matrices; not a drop-in scipy replacement.
"""

from . import linalg

__all__ = ["linalg"]
