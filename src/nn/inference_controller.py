from __future__ import annotations

from typing import Any


class AdaptiveInferenceController:
    """Stateful inference-time adapter for less conservative NN rollouts.

    The learned controller remains responsible for local decisions, but we add:
    1. Restart hysteresis: require repeated high-risk signals before restarting.
    2. Restart cooldown: avoid immediate repeated restarts.
    3. Stable-step ramp-up: if several consecutive steps are clean, expand ds.
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
        self.input_dim = int(getattr(base_controller, "input_dim", 10))
        self.reset()

    def reset(self) -> None:
        self._stable_steps = 0
        self._high_restart_streak = 0
        self._restart_cooldown = 0

    def _clamp_step_size(self, ds: float) -> float:
        ds_value = max(float(ds), self.min_step_size)
        if self.max_step_size is not None:
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
                "adaptive_growth_multiplier": float(growth_multiplier),
                "min_step_size_applied": self.min_step_size,
                "max_step_size_applied": self.max_step_size,
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
            return

        restart_prob = float(info.get("controller_info", {}).get("restart_prob", 0.0))
        if restart_prob <= self.restart_reset_threshold and not bool(info.get("need_restart", False)):
            self._high_restart_streak = 0

        is_stable = (
            not bool(info.get("need_restart", False))
            and not bool(info.get("applied_projection", False))
            and int(info.get("backtracks", 0)) == 0
            and float(info.get("sigma_error", 0.0)) <= self.stable_sigma_error
        )
        self._stable_steps = self._stable_steps + 1 if is_stable else 0
