"""Lazy exports for utility modules.

This keeps optional dependencies like PyYAML out of pure numerical code paths.
"""

from importlib import import_module


__all__ = [
    "project_to_contour",
    "sigma_min_at",
    "load_yaml_config",
    "validate_config",
    "contour_closure_error",
    "residual_norm",
    "smallest_singular_triplet",
]


def __getattr__(name):
    if name in {"project_to_contour", "sigma_min_at"}:
        module = import_module(".contour_init", __name__)
        return getattr(module, name)
    if name in {"load_yaml_config", "validate_config"}:
        module = import_module(".config", __name__)
        return getattr(module, name)
    if name in {"contour_closure_error", "residual_norm"}:
        module = import_module(".metrics", __name__)
        return getattr(module, name)
    if name == "smallest_singular_triplet":
        return import_module(".svd", __name__).smallest_singular_triplet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
