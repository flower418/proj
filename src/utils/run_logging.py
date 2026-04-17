from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _complex_parts(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


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
    def __init__(self, log_dir: str | Path, run_name: str = 'run', timestamped: bool = True, echo: bool = True):
        base_dir = Path(log_dir)
        self.log_dir = base_dir / f'{run_name}_{_timestamp()}' if timestamped else base_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.log_dir / 'run.log'
        self._run_handle = self.run_log_path.open('a', encoding='utf-8')
        self._jsonl_handles: dict[Path, Any] = {}
        self.echo = bool(echo)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        if self.echo:
            print(line, flush=True)
        self._run_handle.write(line + '\n')
        self._run_handle.flush()

    def write_json(self, filename: str, payload: Any) -> Path:
        path = self.log_dir / filename
        with path.open('w', encoding='utf-8') as fh:
            json.dump(to_jsonable(payload), fh, indent=2, ensure_ascii=False)
        return path

    def append_jsonl(self, filename: str, payload: Any) -> Path:
        path = self.log_dir / filename
        handle = self._jsonl_handles.get(path)
        if handle is None:
            handle = path.open('a', encoding='utf-8')
            self._jsonl_handles[path] = handle
        record = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'payload': to_jsonable(payload),
        }
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        handle.flush()
        return path

    def close(self) -> None:
        for handle in self._jsonl_handles.values():
            handle.close()
        self._jsonl_handles.clear()
        self._run_handle.close()

    def __enter__(self) -> 'RunLogger':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class StepDiagnosticsCollector:
    label: str
    num_steps: int = 0
    total_raw_ds: float = 0.0
    total_ds: float = 0.0
    min_ds: float = float('inf')
    max_ds: float = 0.0
    num_projections: int = 0
    total_sigma_error: float = 0.0
    max_sigma_error: float = 0.0
    total_raw_sigma_error: float = 0.0
    max_raw_sigma_error: float = 0.0
    num_approx_triplet_skips: int = 0

    def observe(self, info: dict[str, Any]) -> None:
        self.num_steps += 1

        raw_ds = float(info.get('raw_ds', info.get('ds', 0.0)))
        ds = float(info.get('ds', raw_ds))
        self.total_raw_ds += raw_ds
        self.total_ds += ds
        self.min_ds = min(self.min_ds, ds)
        self.max_ds = max(self.max_ds, ds)

        if bool(info.get('applied_projection', False)):
            self.num_projections += 1

        sigma_error = float(info.get('sigma_error', 0.0))
        raw_sigma_error = float(info.get('raw_sigma_error', sigma_error))
        self.total_sigma_error += sigma_error
        self.max_sigma_error = max(self.max_sigma_error, sigma_error)
        self.total_raw_sigma_error += raw_sigma_error
        self.max_raw_sigma_error = max(self.max_raw_sigma_error, raw_sigma_error)
        if str(info.get('triplet_refresh_mode', '')) == 'approx_skip':
            self.num_approx_triplet_skips += 1

    def summary(self) -> dict[str, Any]:
        if self.num_steps == 0:
            return {'label': self.label, 'num_steps': 0}
        return {
            'label': self.label,
            'num_steps': int(self.num_steps),
            'mean_raw_step_size': float(self.total_raw_ds / self.num_steps),
            'mean_accepted_step_size': float(self.total_ds / self.num_steps),
            'min_accepted_step_size': float(self.min_ds),
            'max_accepted_step_size': float(self.max_ds),
            'num_projections': int(self.num_projections),
            'projection_rate': float(self.num_projections / self.num_steps),
            'mean_sigma_error': float(self.total_sigma_error / self.num_steps),
            'max_sigma_error': float(self.max_sigma_error),
            'mean_raw_sigma_error': float(self.total_raw_sigma_error / self.num_steps),
            'max_raw_sigma_error': float(self.max_raw_sigma_error),
            'num_approx_triplet_skips': int(self.num_approx_triplet_skips),
            'approx_triplet_skip_rate': float(self.num_approx_triplet_skips / self.num_steps),
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
        if print_every > 0 and int(payload.get('step', 0)) % print_every == 0:
            run_logger.log(formatter(payload))

    return _callback


def format_nn_step(info: dict[str, Any], label: str = 'nn') -> str:
    return (
        f'[{label}] '
        f"step={int(info.get('step', 0)):05d} "
        f"ds={float(info.get('ds', 0.0)):.6f} "
        f"|z-z0|={float(info.get('distance_to_start', 0.0)):.6f} "
    )


def format_newton_step(info: dict[str, Any], label: str = 'newton') -> str:
    return (
        f'[{label}] '
        f"step={int(info.get('step', 0)):05d} "
        f"ds={float(info.get('ds', 0.0)):.6f} "
        f"|z-z0|={float(info.get('distance_to_start', 0.0)):.6f} "
    )
