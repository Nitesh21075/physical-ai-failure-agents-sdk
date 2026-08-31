"""Bounded controller adapters shared by the host schema and Isaac worker."""

from __future__ import annotations

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


def create_controller_adapter(spec: dict[str, object]) -> ControllerAdapter:
    """Construct only a registered adapter from a host-validated controller spec."""
    controller_id = spec.get("controller_id")
    if controller_id == "fixed_velocity":
        return FixedVelocityAdapter(
            linear_velocity_mps=float(spec["linear_velocity_mps"]),
            angular_velocity_radps=float(spec["angular_velocity_radps"]),
            max_control_steps=int(spec["control_steps"]),
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
