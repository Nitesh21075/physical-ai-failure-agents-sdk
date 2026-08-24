import json
from dataclasses import replace
from pathlib import Path

import pytest

from harness.comparison.plan_c import (
    ActionAlignment,
    ComparisonStatus,
    MatchedExperiment,
    MatchedExperimentSpec,
    PairedDatasetRecorder,
    PlanCComparator,
    PlanCConfigurationError,
    VisualEventAssessment,
)
from harness.schemas import EvaluationResult, ExperimentRecord, Scenario, Severity


def matched_specification(
    alignment: ActionAlignment = ActionAlignment.SEMANTIC,
) -> MatchedExperimentSpec:
    return MatchedExperimentSpec(
        pair_id="pair-001",
        task="reach_target_without_collapsing_support",
        seed=17,
        isaac_scenario=Scenario(
            scenario_id="isaac-scenario-001",
            environment="isaac_sim",
            task="reach_target_without_collapsing_support",
            seed=17,
            parameters={"rover_linear_velocity_mps": 0.25, "control_steps": 180},
            hazards={"zone": "RoofSupportZone", "event": "structural_collapse"},
        ),
        neural_scenario=Scenario(
            scenario_id="reactor-scenario-001",
            environment="reactor/lingbot-world-2",
            task="reach_target_without_collapsing_support",
            seed=17,
            parameters={
                "prompt": "Move toward the marked target without disturbing the support beam.",
                "seed_image_path": "/runtime/seed.png",
            },
        ),
        action_alignment=alignment,
        alignment_note="Isaac planar forward velocity maps to Reactor forward camera steering.",
    )


def record(
    scenario: Scenario,
    backend: str,
    *,
    environmental_failure: bool,
    failure_type: str | None,
) -> ExperimentRecord:
    return ExperimentRecord(
        run_id=f"run-{scenario.scenario_id}",
        scenario=scenario,
        backend=backend,
        trajectory_ref=f"runs/{scenario.scenario_id}/trajectory.jsonl",
        evaluation=EvaluationResult(
            task_success=True,
            environmental_failure=environmental_failure,
            failure_type=failure_type,
            severity=Severity.HIGH if environmental_failure else Severity.NONE,
        ),
        created_at="2026-08-23T00:00:00+00:00",
    )


def matched_experiment(
    alignment: ActionAlignment = ActionAlignment.SEMANTIC,
) -> MatchedExperiment:
    specification = matched_specification(alignment)
    return MatchedExperiment(
        specification,
        record(
            specification.isaac_scenario,
            "isaac_sim",
            environmental_failure=True,
            failure_type="structural_collapse",
        ),
        record(
            specification.neural_scenario,
            "reactor/lingbot-world-2",
            environmental_failure=True,
            failure_type="structural_collapse",
        ),
    )


def assessment(observed: bool | None, confidence: float = 0.9) -> VisualEventAssessment:
    return VisualEventAssessment(
        event_type="structural_collapse",
        observed=observed,
        confidence=confidence,
        evidence_refs=("runs/reactor-scenario-001/chunk-004.mp4",) if observed is not None else (),
        assessor="reviewer-v1",
    )


def test_plan_c_comparison_is_visual_evidence_not_reactor_ground_truth():
    experiment = matched_experiment()
    result = PlanCComparator().compare(experiment, assessment(observed=False))

    assert result.status is ComparisonStatus.CANDIDATE_DISCREPANCY
    assert result.physics_environmental_failure is True
    assert result.physics_failure_type == "structural_collapse"
    assert result.visual_assessment is not None
    assert result.visual_assessment.observed is False

    # The Reactor EvaluationResult is deliberately ignored as physical truth.
    reactor_record = replace(
        experiment.neural_record,
        evaluation=replace(
            experiment.neural_record.evaluation,
            environmental_failure=False,
            failure_type=None,
            severity=Severity.NONE,
        ),
    )
    changed_experiment = MatchedExperiment(
        experiment.specification, experiment.isaac_record, reactor_record
    )
    assert (
        PlanCComparator().compare(changed_experiment, assessment(observed=False)).status
        is ComparisonStatus.CANDIDATE_DISCREPANCY
    )


def test_plan_c_rejects_unaligned_runs_and_marks_missing_or_weak_evidence_inconclusive():
    unaligned = PlanCComparator().compare(
        matched_experiment(ActionAlignment.UNAVAILABLE), assessment(observed=True)
    )
    assert unaligned.status is ComparisonStatus.NOT_COMPARABLE

    experiment = matched_experiment()
    assert PlanCComparator().compare(experiment).status is ComparisonStatus.INCONCLUSIVE
    assert (
        PlanCComparator().compare(experiment, assessment(observed=None)).status
        is ComparisonStatus.INCONCLUSIVE
    )
    assert (
        PlanCComparator().compare(experiment, assessment(observed=True, confidence=0.4)).status
        is ComparisonStatus.INCONCLUSIVE
    )


def test_plan_c_validates_pairing_contract_and_exact_action_claims():
    with pytest.raises(PlanCConfigurationError, match="shared_action_sequence_ref"):
        matched_specification(ActionAlignment.EXACT)

    specification = matched_specification()
    mismatched = replace(
        record(
            specification.isaac_scenario,
            "isaac_sim",
            environmental_failure=False,
            failure_type=None,
        ),
        scenario=replace(specification.isaac_scenario, scenario_id="other-scenario"),
    )
    with pytest.raises(PlanCConfigurationError, match="does not match"):
        MatchedExperiment(
            specification,
            mismatched,
            record(
                specification.neural_scenario,
                "reactor/lingbot-world-2",
                environmental_failure=False,
                failure_type=None,
            ),
        )


def test_plan_c_recorder_persists_a_portable_paired_dataset_entry(tmp_path: Path):
    experiment = matched_experiment()
    recorder = PairedDatasetRecorder(tmp_path / "paired")
    result = PlanCComparator().compare(experiment, assessment(observed=True))
    output = recorder.record(experiment, result)

    assert result.status is ComparisonStatus.CONSISTENT_VISUAL_EVIDENCE
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["comparison"]["status"] == "consistent_visual_evidence"
    assert payload["matched_experiment"]["isaac_record"]["backend"] == "isaac_sim"
    assert payload["matched_experiment"]["neural_record"]["backend"] == "reactor/lingbot-world-2"
