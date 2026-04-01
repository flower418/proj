from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str | Path, validate: bool = True) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if validate:
        validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")

    ode = config.get("ode", {})
    controller = config.get("controller", {})
    tracker = config.get("tracker", {})
    training = config.get("training", {})

    if float(ode.get("epsilon", 0.0)) <= 0:
        raise ValueError("ode.epsilon must be positive.")

    min_step = float(ode.get("min_step_size", 0.0))
    init_step = float(ode.get("initial_step_size", 0.0))
    max_step = float(ode.get("max_step_size", 0.0))
    if not (0.0 < min_step <= init_step <= max_step):
        raise ValueError("ode step sizes must satisfy 0 < min <= initial <= max.")

    ctrl_min = float(controller.get("step_size_min", 0.0))
    ctrl_max = float(controller.get("step_size_max", max_step))
    if not (0.0 < ctrl_min <= ctrl_max):
        raise ValueError("controller step size bounds must satisfy 0 < min <= max.")
    if controller.get("norm_type", "layernorm").lower() not in {"layernorm", "batchnorm"}:
        raise ValueError("controller.norm_type must be 'layernorm' or 'batchnorm'.")

    if float(tracker.get("closure_tol", 0.0)) <= 0:
        raise ValueError("tracker.closure_tol must be positive.")

    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive.")

    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be positive.")
    if float(training.get("focal_gamma", 0.0)) < 0:
        raise ValueError("training.focal_gamma must be non-negative.")
    if float(training.get("scheduler_factor", 0.0)) <= 0 or float(training.get("scheduler_factor", 0.0)) >= 1:
        raise ValueError("training.scheduler_factor must be in (0, 1).")
    if int(training.get("scheduler_patience", 0)) < 0:
        raise ValueError("training.scheduler_patience must be non-negative.")
