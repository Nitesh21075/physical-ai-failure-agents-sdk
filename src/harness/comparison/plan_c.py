"""Plan C: paired neural-world and physics-world experiment comparison.

Isaac evaluations remain the source of truth for physical consequences. A
Reactor result is represented as labelled visual evidence and can identify a
candidate discrepancy, never establish a physical fact on its own.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from harness.schemas import ExperimentRecord, Scenario


class PlanCConfigurationError(ValueError):
    """Raised when a proposed pair cannot be represented truthfully."""


class ActionAlignment(StrEnum):
    """How closely the two backend action traces represent the same intent."""

    EXACT = "exact"
    SEMANTIC = "semantic"
    UNAVAILABLE = "unavailable"


class ComparisonStatus(StrEnum):
    """Interpretation status for a paired experiment."""

    NOT_COMPARABLE = "not_comparable"
    INCONCLUSIVE = "inconclusive"
    CONSISTENT_VISUAL_EVIDENCE = "consistent_visual_evidence"
    CANDIDATE_DISCREPANCY = "candidate_discrepancy"


def _non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PlanCConfigurationError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class VisualEventAssessment:
    """A non-authoritative annotation of Reactor video evidence.

    ``observed`` is ``None`` when an assessor cannot decide.  This is not an
    ``EvaluationResult`` because the pixels do not provide the structured
    environmental ground truth required by the main evaluator.
    """

    event_type: str
    observed: bool | None
    confidence: float
    evidence_refs: tuple[str, ...] = ()
    assessor: str = "unspecified"

    def __post_init__(self) -> None:
        _non_empty_string(self.event_type, "visual event_type")
        _non_empty_string(self.assessor, "visual assessor")
        if self.observed is not None and not isinstance(self.observed, bool):
            raise PlanCConfigurationError("visual observed must be true, false, or null")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise PlanCConfigurationError("visual confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if self.observed is not None and not self.evidence_refs:
            raise PlanCConfigurationError("a visual decision requires at least one evidence reference")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatchedExperimentSpec:
    """The declared cross-backend equivalence of one proposed experiment.

    Current Isaac and Reactor action vocabularies differ, so v0 pairs normally
    use ``SEMANTIC`` alignment with a concise mapping note.  ``EXACT`` is
    available only for a future common action adapter and requires an artifact
    reference to the shared action sequence.
    """

    task: str
    seed: int
    isaac_scenario: Scenario
    neural_scenario: Scenario
    action_alignment: ActionAlignment
    alignment_note: str
    shared_action_sequence_ref: str | None = None
    pair_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        _non_empty_string(self.task, "task")
        _non_empty_string(self.pair_id, "pair_id")
        _non_empty_string(self.alignment_note, "alignment_note")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise PlanCConfigurationError("seed must be an integer")
        if not isinstance(self.action_alignment, ActionAlignment):
            object.__setattr__(self, "action_alignment", ActionAlignment(self.action_alignment))
        if self.isaac_scenario.environment != "isaac_sim":
            raise PlanCConfigurationError("isaac_scenario.environment must be 'isaac_sim'")
        if not self.neural_scenario.environment.startswith("reactor/"):
            raise PlanCConfigurationError(
                "neural_scenario.environment must be a qualified Reactor model"
            )
        for name, scenario in (
            ("isaac_scenario", self.isaac_scenario),
            ("neural_scenario", self.neural_scenario),
        ):
            if scenario.task != self.task:
                raise PlanCConfigurationError(f"{name}.task must match the pair task")
            if scenario.seed != self.seed:
                raise PlanCConfigurationError(f"{name}.seed must match the pair seed")
        if self.action_alignment is ActionAlignment.EXACT and not self.shared_action_sequence_ref:
            raise PlanCConfigurationError(
                "exact action alignment requires shared_action_sequence_ref"
            )
        if self.shared_action_sequence_ref is not None:
            _non_empty_string(self.shared_action_sequence_ref, "shared_action_sequence_ref")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatchedExperiment:
    """Two recorded executions bound to one :class:`MatchedExperimentSpec`."""

    specification: MatchedExperimentSpec
    isaac_record: ExperimentRecord
    neural_record: ExperimentRecord

    def __post_init__(self) -> None:
        if self.isaac_record.backend != "isaac_sim":
            raise PlanCConfigurationError("isaac_record.backend must be 'isaac_sim'")
        if not self.neural_record.backend.startswith("reactor/"):
            raise PlanCConfigurationError("neural_record.backend must be a qualified Reactor model")
        self._validate_record("isaac", self.isaac_record, self.specification.isaac_scenario)
        self._validate_record("neural", self.neural_record, self.specification.neural_scenario)

    @staticmethod
    def _validate_record(name: str, record: ExperimentRecord, scenario: Scenario) -> None:
        if record.scenario != scenario:
            raise PlanCConfigurationError(f"{name}_record scenario does not match the pair specification")

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification": self.specification.to_dict(),
            "isaac_record": self.isaac_record.to_dict(),
            "neural_record": self.neural_record.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """A capability-qualified interpretation of a matched experiment."""

    pair_id: str
    status: ComparisonStatus
    action_alignment: ActionAlignment
    physics_environmental_failure: bool
    physics_failure_type: str | None
    visual_assessment: VisualEventAssessment | None
    reason: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _non_empty_string(self.pair_id, "pair_id")
        _non_empty_string(self.reason, "comparison reason")
        if not isinstance(self.status, ComparisonStatus):
            object.__setattr__(self, "status", ComparisonStatus(self.status))
        if not isinstance(self.action_alignment, ActionAlignment):
            object.__setattr__(self, "action_alignment", ActionAlignment(self.action_alignment))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlanCComparator:
    """Compare Isaac ground truth with explicit, qualified visual evidence."""

    def __init__(self, minimum_visual_confidence: float = 0.7) -> None:
        if (
            isinstance(minimum_visual_confidence, bool)
            or not isinstance(minimum_visual_confidence, (int, float))
            or not 0.0 <= minimum_visual_confidence <= 1.0
        ):
            raise ValueError("minimum_visual_confidence must be between 0 and 1")
        self.minimum_visual_confidence = float(minimum_visual_confidence)

    def compare(
        self,
        experiment: MatchedExperiment,
        visual_assessment: VisualEventAssessment | None = None,
    ) -> ComparisonResult:
        specification = experiment.specification
        physics = experiment.isaac_record.evaluation
        evidence_refs = visual_assessment.evidence_refs if visual_assessment else ()
        common = {
            "pair_id": specification.pair_id,
            "action_alignment": specification.action_alignment,
            "physics_environmental_failure": physics.environmental_failure,
            "physics_failure_type": physics.failure_type,
            "visual_assessment": visual_assessment,
            "evidence_refs": evidence_refs,
        }
        if specification.action_alignment is ActionAlignment.UNAVAILABLE:
            return ComparisonResult(
                status=ComparisonStatus.NOT_COMPARABLE,
                reason="no cross-backend action alignment was declared",
                **common,
            )
        if visual_assessment is None:
            return ComparisonResult(
                status=ComparisonStatus.INCONCLUSIVE,
                reason="no visual assessment was supplied for the Reactor trajectory",
                **common,
            )
        if visual_assessment.observed is None:
            return ComparisonResult(
                status=ComparisonStatus.INCONCLUSIVE,
                reason="the visual assessor could not determine whether the event was visible",
                **common,
            )
        if visual_assessment.confidence < self.minimum_visual_confidence:
            return ComparisonResult(
                status=ComparisonStatus.INCONCLUSIVE,
                reason=(
                    "visual assessment confidence is below the configured comparison threshold"
                ),
                **common,
            )

        event_matches = (
            not physics.environmental_failure
            or visual_assessment.event_type == physics.failure_type
        )
        if visual_assessment.observed == physics.environmental_failure and event_matches:
            return ComparisonResult(
                status=ComparisonStatus.CONSISTENT_VISUAL_EVIDENCE,
                reason=(
                    "Reactor visual evidence is consistent with the Isaac physical outcome; "
                    "it is not independent physical ground truth"
                ),
                **common,
            )
        return ComparisonResult(
            status=ComparisonStatus.CANDIDATE_DISCREPANCY,
            reason=(
                "Reactor visual evidence differs from the Isaac physical outcome; "
                "review the linked media before treating it as a model discrepancy"
            ),
            **common,
        )


class PairedDatasetRecorder:
    """Persist small Plan C metadata while leaving trajectories/media in place."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def record(self, experiment: MatchedExperiment, result: ComparisonResult) -> Path:
        if experiment.specification.pair_id != result.pair_id:
            raise PlanCConfigurationError("comparison result belongs to a different pair")
        pair_directory = self.root / experiment.specification.pair_id
        if pair_directory.exists():
            raise FileExistsError(f"paired dataset entry already exists: {pair_directory}")
        pair_directory.mkdir(parents=True)
        comparison_path = pair_directory / "comparison.json"
        payload: Mapping[str, Any] = {
            "matched_experiment": experiment.to_dict(),
            "comparison": result.to_dict(),
        }
        comparison_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return comparison_path
