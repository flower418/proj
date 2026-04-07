from __future__ import annotations

from typing import Any


class AdaptiveInferenceController:
    """Stateful inference-time adapter for less conservative NN rollouts.

    The learned controller remains responsible for local decisions, but we add:
    1. Restart hysteresis: require repeated high-risk signals before restarting.
    2. Restart cooldown: avoid immediate repeated restarts.
    3. Stable-step ramp-up: if several consecutive steps are clean, expand ds.
    4. Projection-aware ceiling: repeated projections temporarily reduce the
       allowed step-size ceiling during inference.
    """

    def __init__(
        self,
        base_controller,
        min_step_size: float,
        max_step_size: float | None = None,
        restart_threshold: float = 0.9,
        restart_reset_threshold: float = 0.6,
        restart_confirm_steps: int = 2,
        restart_cooldown_steps: int = 4,
        stable_growth_factor: float = 1.25,
        stable_growth_interval: int = 4,
        stable_restart_prob: float = 0.2,
        stable_sigma_error: float = 5.0e-5,
        projection_shrink_factor: float = 0.85,
        projection_ceiling_recovery: float = 1.12,
        projection_free_recovery_steps: int = 2,
        min_ceiling_ratio: float = 0.4,
        projection_penalty_distance_ratio: float = 0.12,
        projection_penalty_sigma_error: float = 1.5e-4,
        projection_penalty_streak: int = 3,
    ):
        self.base_controller = base_controller
        self.min_step_size = float(min_step_size)
        self.max_step_size = None if max_step_size is None else float(max_step_size)
        self.restart_threshold = float(restart_threshold)
        self.restart_reset_threshold = float(restart_reset_threshold)
        self.restart_confirm_steps = max(int(restart_confirm_steps), 1)
        self.restart_cooldown_steps = max(int(restart_cooldown_steps), 0)
        self.stable_growth_factor = max(float(stable_growth_factor), 1.0)
        self.stable_growth_interval = max(int(stable_growth_interval), 1)
        self.stable_restart_prob = float(stable_restart_prob)
        self.stable_sigma_error = float(stable_sigma_error)
        self.projection_shrink_factor = min(max(float(projection_shrink_factor), 0.1), 0.99)
        self.projection_ceiling_recovery = max(float(projection_ceiling_recovery), 1.0)
        self.projection_free_recovery_steps = max(int(projection_free_recovery_steps), 1)
        self.min_ceiling_ratio = min(max(float(min_ceiling_ratio), 0.01), 1.0)
        self.projection_penalty_distance_ratio = max(float(projection_penalty_distance_ratio), 0.0)
        self.projection_penalty_sigma_error = max(float(projection_penalty_sigma_error), 0.0)
        self.projection_penalty_streak = max(int(projection_penalty_streak), 1)
        self.input_dim = int(getattr(base_controller, "input_dim", 10))
        self.reset()

    def reset(self) -> None:
        self._stable_steps = 0
        self._high_restart_streak = 0
        self._restart_cooldown = 0
        self._projection_free_steps = 0
        self._projection_streak = 0
        self._base_step_ceiling = None if self.max_step_size is None else float(self.max_step_size)
        self._dynamic_step_ceiling = self._base_step_ceiling

    def _clamp_step_size(self, ds: float) -> float:
        ds_value = max(float(ds), self.min_step_size)
        if self._dynamic_step_ceiling is not None:
            ds_value = min(ds_value, self._dynamic_step_ceiling)
        elif self.max_step_size is not None:
            ds_value = min(ds_value, self.max_step_size)
        return ds_value

    def predict(self, state_np) -> tuple[float, bool]:
        ds, need_restart, _ = self.predict_with_info(state_np)
        return ds, need_restart

    def predict_with_info(self, state_np) -> tuple[float, bool, dict[str, Any]]:
        if hasattr(self.base_controller, "predict_with_info"):
            ds, _, info = self.base_controller.predict_with_info(state_np)
        else:
            ds, _ = self.base_controller.predict(state_np)
            info = {}

        info = dict(info)
        restart_prob = float(info.get("restart_prob", 0.0))
        ds_value = self._clamp_step_size(ds)

        if restart_prob >= self.restart_threshold:
            self._high_restart_streak += 1
        elif restart_prob <= self.restart_reset_threshold:
            self._high_restart_streak = 0

        growth_multiplier = 1.0
        if self._stable_steps >= self.stable_growth_interval and restart_prob <= self.stable_restart_prob:
            growth_rounds = self._stable_steps // self.stable_growth_interval
            growth_multiplier = self.stable_growth_factor ** growth_rounds
        ds_value = self._clamp_step_size(ds_value * growth_multiplier)

        need_restart = bool(
            self._restart_cooldown == 0 and self._high_restart_streak >= self.restart_confirm_steps
        )

        info.update(
            {
                "restart_threshold": self.restart_threshold,
                "restart_reset_threshold": self.restart_reset_threshold,
                "restart_confirm_steps": self.restart_confirm_steps,
                "restart_cooldown": int(self._restart_cooldown),
                "high_restart_streak": int(self._high_restart_streak),
                "stable_steps": int(self._stable_steps),
                "projection_streak": int(self._projection_streak),
                "projection_free_steps": int(self._projection_free_steps),
                "adaptive_growth_multiplier": float(growth_multiplier),
                "min_step_size_applied": self.min_step_size,
                "max_step_size_applied": self.max_step_size,
                "dynamic_step_ceiling": self._dynamic_step_ceiling,
            }
        )
        return ds_value, need_restart, info

    def observe_step(self, info: dict[str, Any]) -> None:
        if self._restart_cooldown > 0:
            self._restart_cooldown -= 1

        if bool(info.get("applied_restart", False)):
            self._stable_steps = 0
            self._high_restart_streak = 0
            self._restart_cooldown = self.restart_cooldown_steps
            self._projection_streak = 0
            self._projection_free_steps = 0
            return

        restart_prob = float(info.get("controller_info", {}).get("restart_prob", 0.0))
        if restart_prob <= self.restart_reset_threshold and not bool(info.get("need_restart", False)):
            self._high_restart_streak = 0

        applied_projection = bool(info.get("applied_projection", False))
        sigma_error = float(info.get("sigma_error", 0.0))
        ds = max(float(info.get("ds", 0.0)), self.min_step_size)
        projection_distance = float(info.get("projection_distance", 0.0))
        projection_distance_ratio = projection_distance / max(ds, 1e-12)
        severe_projection = bool(
            applied_projection
            and (
                projection_distance_ratio >= self.projection_penalty_distance_ratio
                or sigma_error >= self.projection_penalty_sigma_error
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

        is_stable = (
            not bool(info.get("need_restart", False))
            and not applied_projection
            and int(info.get("backtracks", 0)) == 0
            and float(info.get("sigma_error", 0.0)) <= self.stable_sigma_error
        )
        self._stable_steps = self._stable_steps + 1 if is_stable else 0
