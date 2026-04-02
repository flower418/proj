"""Lazy exports for core modules.

Keep package import side effects minimal so submodules can be used without
pulling in optional training dependencies such as torch.
"""

from importlib import import_module


__all__ = ["ContourTracker", "ManifoldODE", "PseudoinverseSolver"]


def __getattr__(name):
    if name == "ContourTracker":
        return import_module(".contour_tracker", __name__).ContourTracker
    if name == "ManifoldODE":
        return import_module(".manifold_ode", __name__).ManifoldODE
    if name == "PseudoinverseSolver":
        return import_module(".pseudoinverse", __name__).PseudoinverseSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
