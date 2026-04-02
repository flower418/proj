"""Lazy exports for training modules."""

from importlib import import_module


__all__ = ["ExpertDataGenerator", "ExpertDataset", "DAggerAugmenter", "ExpertSolver", "TrainingLogger", "ControllerTrainer"]


def __getattr__(name):
    if name in {"ExpertDataGenerator", "ExpertDataset"}:
        module = import_module(".data_generator", __name__)
        return getattr(module, name)
    if name == "DAggerAugmenter":
        return import_module(".dagger_augmentation", __name__).DAggerAugmenter
    if name == "ExpertSolver":
        return import_module(".expert_solver", __name__).ExpertSolver
    if name == "TrainingLogger":
        return import_module(".logger", __name__).TrainingLogger
    if name == "ControllerTrainer":
        return import_module(".trainer", __name__).ControllerTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
