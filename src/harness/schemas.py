"""Small serializable records shared by scientific persistence and Plan C."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class SchemaValidationError(ValueError):
    """Raised when data cannot represent a harness schema."""


class Severity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def _non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a non-empty string")


def _mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{field_name} must be a mapping")
    return dict(value)


@dataclass(frozen=True, slots=True)
class Scenario:
    environment: str
    task: str
    seed: int = 0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    hazards: Mapping[str, Any] = field(default_factory=dict)
    scenario_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        _non_empty_string(self.environment, "environment")
        _non_empty_string(self.task, "task")
        _non_empty_string(self.scenario_id, "scenario_id")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SchemaValidationError("seed must be an integer")
        object.__setattr__(self, "parameters", _mapping(self.parameters, "parameters"))
        object.__setattr__(self, "hazards", _mapping(self.hazards, "hazards"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Scenario:
        return cls(
            scenario_id=value.get("scenario_id", str(uuid4())),
            environment=value["environment"],
            task=value["task"],
            seed=value.get("seed", 0),
            parameters=value.get("parameters", {}),
            hazards=value.get("hazards", {}),
        )

@dataclass(frozen=True, slots=True)
class EvaluationResult:
    task_success: bool
    environmental_failure: bool
    failure_type: str | None
    severity: Severity
    robot_safety_events: tuple[Mapping[str, Any], ...] = ()
    terminal: bool = False
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.severity, Severity):
            object.__setattr__(self, "severity", Severity(self.severity))
        if self.environmental_failure and not self.failure_type:
            raise SchemaValidationError("environmental failures require a failure_type")
        if not self.environmental_failure and self.failure_type is not None:
            raise SchemaValidationError("failure_type requires environmental_failure")
        object.__setattr__(
            self,
            "robot_safety_events",
            tuple(_mapping(event, "robot safety event") for event in self.robot_safety_events),
        )
        object.__setattr__(self, "metrics", _mapping(self.metrics, "metrics"))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    run_id: str
    scenario: Scenario
    backend: str
    trajectory_ref: str
    evaluation: EvaluationResult
    created_at: str

    def __post_init__(self) -> None:
        _non_empty_string(self.run_id, "run_id")
        _non_empty_string(self.backend, "backend")
        _non_empty_string(self.trajectory_ref, "trajectory_ref")
        _non_empty_string(self.created_at, "created_at")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
