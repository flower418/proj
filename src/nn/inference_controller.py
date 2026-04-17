from __future__ import annotations

from typing import Any


class AdaptiveInferenceController:
    """Stateful inference-time adapter for step-size-only rollouts.

    The learned controller outputs a raw step size, then this wrapper adds:
    1. Stable-step ramp-up.
    2. Projection-aware dynamic ceiling.
    3. Curvature-aware ceiling shrinkage.
    """

    def __init__(
        self,
        base_controller,
        min_step_size: float,
        max_step_size: float | None = None,
        stable_growth_factor: float = 1.25,
        stable_growth_interval: int = 4,
        stable_sigma_error: float = 5.0e-5,
        projection_shrink_factor: float = 0.85,
        projection_ceiling_recovery: float = 1.12,
        projection_free_recovery_steps: int = 2,
        min_ceiling_ratio: float = 0.4,
        projection_penalty_distance_ratio: float = 0.12,
        projection_penalty_sigma_error: float = 1.5e-4,
        projection_penalty_streak: int = 3,
        curvature_turn_threshold: float = 0.14,
        curvature_penalty_streak: int = 2,
        curvature_shrink_factor: float = 0.9,
    ):
        self.base_controller = base_controller
        self.min_step_size = float(min_step_size)
        self.max_step_size = None if max_step_size is None else float(max_step_size)
        self.stable_growth_factor = max(float(stable_growth_factor), 1.0)
        self.stable_growth_interval = max(int(stable_growth_interval), 1)
        self.stable_sigma_error = float(stable_sigma_error)
        self.projection_shrink_factor = min(max(float(projection_shrink_factor), 0.1), 0.99)
        self.projection_ceiling_recovery = max(float(projection_ceiling_recovery), 1.0)
        self.projection_free_recovery_steps = max(int(projection_free_recovery_steps), 1)
        self.min_ceiling_ratio = min(max(float(min_ceiling_ratio), 0.01), 1.0)
        self.projection_penalty_distance_ratio = max(float(projection_penalty_distance_ratio), 0.0)
        self.projection_penalty_sigma_error = max(float(projection_penalty_sigma_error), 0.0)
        self.projection_penalty_streak = max(int(projection_penalty_streak), 1)
        self.curvature_turn_threshold = max(float(curvature_turn_threshold), 0.0)
        self.curvature_penalty_streak = max(int(curvature_penalty_streak), 1)
        self.curvature_shrink_factor = min(max(float(curvature_shrink_factor), 0.1), 0.99)
        self.input_dim = int(getattr(base_controller, "input_dim", 8))
        self.reset()

    def reset(self) -> None:
        self._stable_steps = 0
        self._projection_free_steps = 0
        self._projection_streak = 0
        self._base_step_ceiling = None if self.max_step_size is None else float(self.max_step_size)
        self._dynamic_step_ceiling = self._base_step_ceiling
        self._curvature_streak = 0

    def _clamp_step_size(self, ds: float) -> float:
        ds_value = max(float(ds), self.min_step_size)
        if self._dynamic_step_ceiling is not None:
            ds_value = min(ds_value, self._dynamic_step_ceiling)
        elif self.max_step_size is not None:
            ds_value = min(ds_value, self.max_step_size)
        return ds_value

    def predict(self, state_np) -> float:
        ds, _ = self.predict_with_info(state_np)
        return ds

    def predict_with_info(self, state_np) -> tuple[float, dict[str, Any]]:
        if hasattr(self.base_controller, "predict_with_info"):
            ds, info = self.base_controller.predict_with_info(state_np)
        else:
            ds = self.base_controller.predict(state_np)
            info = {}

        info = dict(info)
        ds_value = self._clamp_step_size(ds)

        growth_multiplier = 1.0
        if self._stable_steps >= self.stable_growth_interval:
            growth_rounds = self._stable_steps // self.stable_growth_interval
            growth_multiplier = self.stable_growth_factor ** growth_rounds
        ds_value = self._clamp_step_size(ds_value * growth_multiplier)

        info.update(
            {
                "stable_steps": int(self._stable_steps),
                "projection_streak": int(self._projection_streak),
                "projection_free_steps": int(self._projection_free_steps),
                "adaptive_growth_multiplier": float(growth_multiplier),
                "min_step_size_applied": self.min_step_size,
                "max_step_size_applied": self.max_step_size,
                "dynamic_step_ceiling": self._dynamic_step_ceiling,
            }
        )
        return ds_value, info

    def observe_step(self, info: dict[str, Any]) -> None:
        applied_projection = bool(info.get("applied_projection", False))
        raw_sigma_error = float(info.get("raw_sigma_error", info.get("sigma_error", 0.0)))
        ds = max(float(info.get("ds", 0.0)), self.min_step_size)
        projection_distance = float(info.get("projection_distance", 0.0))
        tangent_turn = float(info.get("tangent_turn", 0.0))
        projection_distance_ratio = projection_distance / max(ds, 1e-12)
        severe_projection = bool(
            applied_projection
            and (
                projection_distance_ratio >= self.projection_penalty_distance_ratio
                or raw_sigma_error >= self.projection_penalty_sigma_error
            )
        )

        if severe_projection:
            self._projection_streak += 1
            self._projection_free_steps = 0
            if self._base_step_ceiling is not None and self._projection_streak >= self.projection_penalty_streak:
                min_ceiling = max(self.min_step_size, self._base_step_ceiling * self.min_ceiling_ratio)
                current_ceiling = self._dynamic_step_ceiling if self._dynamic_step_ceiling is not None else self._base_step_ceiling
                self._dynamic_step_ceiling = max(min_ceiling, current_ceiling * self.projection_shrink_factor)
                self._projection_streak = 0
        elif applied_projection:
            self._projection_streak = max(self._projection_streak - 1, 0)
            self._projection_free_steps += 1
        else:
            self._projection_streak = 0
            self._projection_free_steps += 1
            if (
                self._base_step_ceiling is not None
                and self._dynamic_step_ceiling is not None
                and self._projection_free_steps >= self.projection_free_recovery_steps
            ):
                self._dynamic_step_ceiling = min(
                    self._base_step_ceiling,
                    self._dynamic_step_ceiling * self.projection_ceiling_recovery,
                )

        if tangent_turn >= self.curvature_turn_threshold:
            self._curvature_streak += 1
            if self._base_step_ceiling is not None and self._curvature_streak >= self.curvature_penalty_streak:
                min_ceiling = max(self.min_step_size, self._base_step_ceiling * self.min_ceiling_ratio)
                current_ceiling = self._dynamic_step_ceiling if self._dynamic_step_ceiling is not None else self._base_step_ceiling
                self._dynamic_step_ceiling = max(min_ceiling, current_ceiling * self.curvature_shrink_factor)
                self._curvature_streak = 0
        else:
            self._curvature_streak = 0

        is_stable = (
            not applied_projection
            and int(info.get("backtracks", 0)) == 0
            and raw_sigma_error <= self.stable_sigma_error
        )
        self._stable_steps = self._stable_steps + 1 if is_stable else 0
