"""
cuDSS integration package for DEVSIM custom direct solver callback.
"""

from .cudss_shim import local_solver_callback

__all__ = ["local_solver_callback"]
