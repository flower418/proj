from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from src.core.pseudoinverse import PseudoinverseSolver
from src.train.data_generator import ExpertDataGenerator
from src.utils.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate expert trajectory data.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--matrix-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--output", default="data/expert_records.npz")
    parser.add_argument("--z0-real", type=float, default=0.2)
    parser.add_argument("--z0-imag", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    np.random.seed(args.seed)
    A = np.random.randn(args.matrix_size, args.matrix_size) + 1j * np.random.randn(args.matrix_size, args.matrix_size)
    solver = PseudoinverseSolver(
        method=config["solver"]["method"],
        tol=config["solver"]["tol"],
        max_iter=config["solver"]["max_iter"],
    )
    generator = ExpertDataGenerator(
        A=A,
        epsilon=config["ode"]["epsilon"],
        expert_tol=1e-8,
        atol=1e-8,
        drift_threshold=config["tracker"]["restart_drift_threshold"],
        base_step_size=config["ode"]["initial_step_size"],
        max_step_size=config["ode"]["max_step_size"],
        closure_tol=config["tracker"]["closure_tol"],
        solver=solver,
    )
    z0 = complex(args.z0_real, args.z0_imag)
    records = generator.generate_trajectory(z0=z0, max_steps=args.max_steps)
    records.extend(generator.add_state_perturbations(records, noise_std=config["training"]["noise_std"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        features=np.stack([record["features"] for record in records]),
        ds_expert=np.array([record["ds_expert"] for record in records], dtype=np.float32),
        y_restart=np.array([record["y_restart"] for record in records], dtype=np.int64),
    )
    print(f"generated {len(records)} records -> {output}")


if __name__ == "__main__":
    main()
