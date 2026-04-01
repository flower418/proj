import torch

from src.nn.loss import ControllerLoss
from src.utils.config import validate_config


def test_focal_loss_runs():
    loss_fn = ControllerLoss()
    ds_pred = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float32)
    ds_true = torch.tensor([0.01, 0.02, 0.01], dtype=torch.float32)
    p_restart = torch.tensor([0.1, 0.9, 0.2], dtype=torch.float32)
    y_restart = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    total, step, restart = loss_fn(ds_pred, ds_true, p_restart, y_restart)
    assert torch.isfinite(total)
    assert torch.isfinite(step)
    assert torch.isfinite(restart)


def test_config_validation_accepts_default_shape():
    config = {
        "ode": {
            "epsilon": 0.1,
            "initial_step_size": 1e-2,
            "min_step_size": 1e-6,
            "max_step_size": 1e-1,
        },
        "controller": {
            "norm_type": "layernorm",
            "step_size_min": 1e-6,
            "step_size_max": 1e-1,
        },
        "tracker": {"closure_tol": 1e-3},
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-3,
            "focal_gamma": 2.0,
            "scheduler_factor": 0.5,
            "scheduler_patience": 5,
        },
    }
    validate_config(config)
