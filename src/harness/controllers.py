"""Bounded controller adapters shared by the host schema and Isaac worker."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class ControllerObservation:
    position_xyz_m: tuple[float, float, float]
    orientation_quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class ControllerCommand:
    linear_velocity_mps: float
    angular_velocity_radps: float


@dataclass(frozen=True)
class ControllerStatus:
    terminate: bool = False
    termination_reason: str | None = None
    goal_reached: bool | None = None
    position_error_m: float | None = None
    heading_error_rad: float | None = None


class ControllerAdapter(Protocol):
    controller_id: str
    max_control_steps: int

    def command(self, observation: ControllerObservation) -> ControllerCommand: ...

    def status(
        self, observation: ControllerObservation, *, control_steps_executed: int
    ) -> ControllerStatus: ...


@dataclass(frozen=True)
class FixedVelocityAdapter:
    linear_velocity_mps: float
    angular_velocity_radps: float
    max_control_steps: int
    controller_id: str = "fixed_velocity"

    def command(self, observation: ControllerObservation) -> ControllerCommand:
        del observation
        return ControllerCommand(self.linear_velocity_mps, self.angular_velocity_radps)

    def status(
        self, observation: ControllerObservation, *, control_steps_executed: int
    ) -> ControllerStatus:
        del observation
        if control_steps_executed >= self.max_control_steps:
            return ControllerStatus(
                terminate=True,
                termination_reason="control_steps_completed",
                goal_reached=None,
            )
        return ControllerStatus()


@dataclass
class GoalPoseAdapter:
    target_position_xy_m: tuple[float, float]
    target_heading_rad: float | None
    max_linear_velocity_mps: float
    max_angular_velocity_radps: float
    position_tolerance_m: float
    heading_tolerance_rad: float
    max_control_steps: int
    stagnation_steps: int
    controller_id: str = "goal_pose"
    best_progress_error: float = math.inf
    last_progress_step: int = 0

    def _errors(self, observation: ControllerObservation) -> tuple[float, float, float | None]:
        x, y, _ = observation.position_xyz_m
        dx = self.target_position_xy_m[0] - x
        dy = self.target_position_xy_m[1] - y
        distance = math.hypot(dx, dy)
        yaw = quaternion_yaw_rad(observation.orientation_quaternion_wxyz)
        bearing_error = normalize_angle_rad(math.atan2(dy, dx) - yaw)
        final_heading_error = (
            normalize_angle_rad(self.target_heading_rad - yaw)
            if self.target_heading_rad is not None
            else None
        )
        return distance, bearing_error, final_heading_error

    def command(self, observation: ControllerObservation) -> ControllerCommand:
        distance, bearing_error, final_heading_error = self._errors(observation)
        if distance <= self.position_tolerance_m:
            if (
                final_heading_error is None
                or abs(final_heading_error) <= self.heading_tolerance_rad
            ):
                return ControllerCommand(0.0, 0.0)
            return ControllerCommand(
                0.0,
                clamp(
                    1.5 * final_heading_error,
                    -self.max_angular_velocity_radps,
                    self.max_angular_velocity_radps,
                ),
            )
        alignment = max(0.0, math.cos(bearing_error))
        return ControllerCommand(
            min(self.max_linear_velocity_mps, distance) * alignment,
            clamp(
                1.5 * bearing_error,
                -self.max_angular_velocity_radps,
                self.max_angular_velocity_radps,
            ),
        )

    def status(
        self, observation: ControllerObservation, *, control_steps_executed: int
    ) -> ControllerStatus:
        distance, _, final_heading_error = self._errors(observation)
        heading_reached = (
            final_heading_error is None
            or abs(final_heading_error) <= self.heading_tolerance_rad
        )
        goal_reached = distance <= self.position_tolerance_m and heading_reached
        if goal_reached:
            return ControllerStatus(
                terminate=True,
                termination_reason="goal_reached",
                goal_reached=True,
                position_error_m=distance,
                heading_error_rad=final_heading_error,
            )
        progress_error = distance
        if distance <= self.position_tolerance_m and final_heading_error is not None:
            progress_error += 0.25 * abs(final_heading_error)
        if progress_error < self.best_progress_error - 0.005:
            self.best_progress_error = progress_error
            self.last_progress_step = control_steps_executed
        if control_steps_executed >= self.max_control_steps:
            reason = "max_control_steps_reached"
        elif control_steps_executed - self.last_progress_step >= self.stagnation_steps:
            reason = "stagnation_detected"
        else:
            return ControllerStatus(
                goal_reached=False,
                position_error_m=distance,
                heading_error_rad=final_heading_error,
            )
        return ControllerStatus(
            terminate=True,
            termination_reason=reason,
            goal_reached=False,
            position_error_m=distance,
            heading_error_rad=final_heading_error,
        )


def create_controller_adapter(spec: dict[str, object]) -> ControllerAdapter:
    """Construct only a registered adapter from a host-validated controller spec."""
    controller_id = spec.get("controller_id")
    if controller_id == "fixed_velocity":
        return FixedVelocityAdapter(
            linear_velocity_mps=float(spec["linear_velocity_mps"]),
            angular_velocity_radps=float(spec["angular_velocity_radps"]),
            max_control_steps=int(spec["control_steps"]),
        )
    if controller_id == "goal_pose":
        target = spec["target_position_xy_m"]
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            raise ValueError("target_position_xy_m must contain two values")
        target_heading = spec.get("target_heading_rad")
        return GoalPoseAdapter(
            target_position_xy_m=(float(target[0]), float(target[1])),
            target_heading_rad=float(target_heading) if target_heading is not None else None,
            max_linear_velocity_mps=float(spec["max_linear_velocity_mps"]),
            max_angular_velocity_radps=float(spec["max_angular_velocity_radps"]),
            position_tolerance_m=float(spec["position_tolerance_m"]),
            heading_tolerance_rad=float(spec["heading_tolerance_rad"]),
            max_control_steps=int(spec["max_control_steps"]),
            stagnation_steps=int(spec["stagnation_steps"]),
        )
    raise ValueError(f"controller adapter is not registered: {controller_id!r}")


def differential_wheel_targets_radps(
    command: ControllerCommand, *, wheel_radius_m: float, wheel_base_m: float
) -> tuple[float, float]:
    if wheel_radius_m <= 0 or wheel_base_m <= 0:
        raise ValueError("wheel dimensions must be positive")
    left = (
        command.linear_velocity_mps - command.angular_velocity_radps * wheel_base_m / 2
    ) / wheel_radius_m
    right = (
        command.linear_velocity_mps + command.angular_velocity_radps * wheel_base_m / 2
    ) / wheel_radius_m
    return left, right


def status_dict(status: ControllerStatus) -> dict[str, object]:
    return asdict(status)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_angle_rad(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_yaw_rad(quaternion_wxyz: tuple[float, float, float, float]) -> float:
    w, x, y, z = quaternion_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
