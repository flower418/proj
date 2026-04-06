from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _complex_parts(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


def format_complex(value: complex) -> str:
    return f"{float(np.real(value)):+.5f}{float(np.imag(value)):+.5f}j"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (complex, np.complexfloating)):
        return _complex_parts(complex(value))
    return value


class RunLogger:
    def __init__(self, log_dir: str | Path, run_name: str = "run", timestamped: bool = True):
        base_dir = Path(log_dir)
        self.log_dir = base_dir / f"{run_name}_{_timestamp()}" if timestamped else base_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.log_dir / "run.log"
        self._run_handle = self.run_log_path.open("a", encoding="utf-8")
        self._jsonl_handles: dict[Path, Any] = {}

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        self._run_handle.write(line + "\n")
        self._run_handle.flush()

    def write_json(self, filename: str, payload: Any) -> Path:
        path = self.log_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(to_jsonable(payload), fh, indent=2, ensure_ascii=False)
        return path

    def append_jsonl(self, filename: str, payload: Any) -> Path:
        path = self.log_dir / filename
        handle = self._jsonl_handles.get(path)
        if handle is None:
            handle = path.open("a", encoding="utf-8")
            self._jsonl_handles[path] = handle
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "payload": to_jsonable(payload),
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        return path

    def close(self) -> None:
        for handle in self._jsonl_handles.values():
            handle.close()
        self._jsonl_handles.clear()
        self._run_handle.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class StepDiagnosticsCollector:
    label: str
    num_steps: int = 0
    total_raw_ds: float = 0.0
    total_ds: float = 0.0
    min_ds: float = float("inf")
    max_ds: float = 0.0
    num_restart_signals: int = 0
    num_applied_restarts: int = 0
    num_projections: int = 0
    num_backtracked_steps: int = 0
    total_backtracks: int = 0
    max_backtracks: int = 0
    total_sigma_error: float = 0.0
    max_sigma_error: float = 0.0
    total_restart_prob: float = 0.0
    restart_prob_count: int = 0

    def observe(self, info: dict[str, Any]) -> None:
        self.num_steps += 1

        raw_ds = float(info.get("raw_ds", info.get("ds", 0.0)))
        ds = float(info.get("ds", raw_ds))
        self.total_raw_ds += raw_ds
        self.total_ds += ds
        self.min_ds = min(self.min_ds, ds)
        self.max_ds = max(self.max_ds, ds)

        if bool(info.get("need_restart", False)):
            self.num_restart_signals += 1
        if bool(info.get("applied_restart", False)):
            self.num_applied_restarts += 1
        if bool(info.get("applied_projection", False)):
            self.num_projections += 1

        backtracks = int(info.get("backtracks", 0))
        self.total_backtracks += backtracks
        self.max_backtracks = max(self.max_backtracks, backtracks)
        if backtracks > 0:
            self.num_backtracked_steps += 1

        sigma_error = float(info.get("sigma_error", 0.0))
        self.total_sigma_error += sigma_error
        self.max_sigma_error = max(self.max_sigma_error, sigma_error)

        controller_info = info.get("controller_info")
        if isinstance(controller_info, dict) and controller_info.get("restart_prob") is not None:
            self.total_restart_prob += float(controller_info["restart_prob"])
            self.restart_prob_count += 1

    def summary(self) -> dict[str, Any]:
        if self.num_steps == 0:
            return {"label": self.label, "num_steps": 0}
        return {
            "label": self.label,
            "num_steps": int(self.num_steps),
            "mean_raw_step_size": float(self.total_raw_ds / self.num_steps),
            "mean_accepted_step_size": float(self.total_ds / self.num_steps),
            "min_accepted_step_size": float(self.min_ds),
            "max_accepted_step_size": float(self.max_ds),
            "num_restart_signals": int(self.num_restart_signals),
            "restart_signal_rate": float(self.num_restart_signals / self.num_steps),
            "num_applied_restarts": int(self.num_applied_restarts),
            "applied_restart_rate": float(self.num_applied_restarts / self.num_steps),
            "num_projections": int(self.num_projections),
            "projection_rate": float(self.num_projections / self.num_steps),
            "num_backtracked_steps": int(self.num_backtracked_steps),
            "mean_backtracks_per_step": float(self.total_backtracks / self.num_steps),
            "max_backtracks": int(self.max_backtracks),
            "mean_sigma_error": float(self.total_sigma_error / self.num_steps),
            "max_sigma_error": float(self.max_sigma_error),
            "mean_restart_prob": float(self.total_restart_prob / self.restart_prob_count) if self.restart_prob_count > 0 else None,
        }


def make_step_callback(
    run_logger: RunLogger,
    collector: StepDiagnosticsCollector,
    jsonl_filename: str,
    formatter: Callable[[dict[str, Any]], str],
    print_every: int = 1,
    info_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Callable[[dict[str, Any]], None]:
    print_every = int(print_every)

    def _callback(info: dict[str, Any]) -> None:
        payload = dict(info)
        if info_transform is not None:
            payload = info_transform(payload)
        collector.observe(payload)
        run_logger.append_jsonl(jsonl_filename, payload)
        if print_every > 0 and int(payload.get("step", 0)) % print_every == 0:
            run_logger.log(formatter(payload))

    return _callback


def format_nn_step(info: dict[str, Any], label: str = "nn") -> str:
    restart_prob = None
    controller_info = info.get("controller_info")
    if isinstance(controller_info, dict):
        restart_prob = controller_info.get("restart_prob")
    restart_prob_str = "None" if restart_prob is None else f"{float(restart_prob):.4f}"
    return (
        f"[{label}] "
        f"step={int(info.get('step', 0)):05d} "
        f"raw_ds={float(info.get('raw_ds', info.get('ds', 0.0))):.6f} "
        f"ds={float(info.get('ds', 0.0)):.6f} "
        f"p_restart={restart_prob_str} "
        f"need_restart={int(bool(info.get('need_restart', False)))} "
        f"applied_restart={int(bool(info.get('applied_restart', False)))} "
        f"projection={int(bool(info.get('applied_projection', False)))} "
        f"backtracks={int(info.get('backtracks', 0))} "
        f"|dz|={float(info.get('step_distance', 0.0)):.6f} "
        f"|z-z0|={float(info.get('distance_to_start', 0.0)):.6f} "
        f"path={float(info.get('path_length', 0.0)):.6f} "
        f"wind={float(info.get('winding_angle', 0.0)):.4f} "
        f"sigma_err={float(info.get('sigma_error', 0.0)):.6e} "
        f"z={format_complex(complex(info.get('z_next', 0.0)))}"
    )


def format_newton_step(info: dict[str, Any], label: str = "newton") -> str:
    return (
        f"[{label}] "
        f"step={int(info.get('step', 0)):05d} "
        f"ds={float(info.get('ds', 0.0)):.6f} "
        f"halvings={int(info.get('predictor_halvings', 0))} "
        f"corrector_iters={int(info.get('corrector_iterations', 0))} "
        f"line_search={int(info.get('line_search_backtracks', 0))} "
        f"|dz|={float(info.get('step_distance', 0.0)):.6f} "
        f"|z-z0|={float(info.get('distance_to_start', 0.0)):.6f} "
        f"path={float(info.get('path_length', 0.0)):.6f} "
        f"wind={float(info.get('winding_angle', 0.0)):.4f} "
        f"sigma_err={float(info.get('sigma_error', 0.0)):.6e} "
        f"z={format_complex(complex(info.get('z_next', 0.0)))}"
    )
