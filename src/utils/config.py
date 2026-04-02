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
    if controller.get("activation", "silu").lower() not in {"relu", "gelu", "silu"}:
        raise ValueError("controller.activation must be 'relu', 'gelu', or 'silu'.")
    head_hidden_dim = controller.get("head_hidden_dim")
    if head_hidden_dim is not None and int(head_hidden_dim) <= 0:
        raise ValueError("controller.head_hidden_dim must be positive when provided.")

    if float(tracker.get("closure_tol", 0.0)) <= 0:
        raise ValueError("tracker.closure_tol must be positive.")

    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive.")

    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be positive.")
    if float(training.get("weight_decay", 0.0)) < 0:
        raise ValueError("training.weight_decay must be non-negative.")
    if float(training.get("focal_gamma", 0.0)) < 0:
        raise ValueError("training.focal_gamma must be non-negative.")
    if training.get("gradient_clip_norm") is not None and float(training.get("gradient_clip_norm")) <= 0:
        raise ValueError("training.gradient_clip_norm must be positive when provided.")
    if int(training.get("early_stop_patience", 0)) < 0:
        raise ValueError("training.early_stop_patience must be non-negative.")
    if float(training.get("scheduler_factor", 0.0)) <= 0 or float(training.get("scheduler_factor", 0.0)) >= 1:
        raise ValueError("training.scheduler_factor must be in (0, 1).")
    if int(training.get("scheduler_patience", 0)) < 0:
        raise ValueError("training.scheduler_patience must be non-negative.")
