"""Lazy exports for nn modules.

Feature extraction is used by pure NumPy/SciPy inference code, while the
controller and loss require torch. Avoid importing torch unless it is actually
needed.
"""

from importlib import import_module


__all__ = ["NNController", "extract_features", "ControllerLoss"]


def __getattr__(name):
    if name == "NNController":
        return import_module(".controller", __name__).NNController
    if name == "extract_features":
        return import_module(".features", __name__).extract_features
    if name == "ControllerLoss":
        return import_module(".loss", __name__).ControllerLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
