"""Persist a browser-recorded Reactor session as a truthful Plan-C pair."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from harness.comparison.plan_c import (
    ActionAlignment,
    MatchedExperiment,
    MatchedExperimentSpec,
    PairedDatasetRecorder,
    PlanCComparator,
)
from harness.media.isaac_export import export_isaac_replay
from harness.persistence.store import ExperimentStore
from harness.research.world_prompt import (
    OpenAIResponsesWorldPromptModel,
    WorldPromptRequest,
    WorldPromptResult,
)
from harness.schemas import EvaluationResult, ExperimentRecord, Scenario, Severity


class PairingError(ValueError):
    """Raised for invalid or expired browser paired-capture operations."""


class PairedCaptureService:
    """Host-side authority for preparing and finalizing a browser Reactor capture.

    Reactor media is received from the browser only after its user-scoped live
    session ends. The server records it as visual evidence, never as physics.
    """

    def __init__(
        self,
        store: ExperimentStore,
        runs_root: str | Path,
        prompt_factory: Callable[[str], OpenAIResponsesWorldPromptModel] | None = None,
    ) -> None:
        self.store, self.runs_root = store, Path(runs_root)
        self.prompt_factory = prompt_factory or OpenAIResponsesWorldPromptModel

    def prepare(self, isaac_run_id: str, *, objective: str, model: str) -> dict:
        record = self.store.get_experiment(isaac_run_id)
        if record is None or record["backend"] != "isaac_sim":
            raise PairingError("select an indexed Isaac Sim recording")
        run_directory = Path(record["run_directory"] or Path(record["trajectory_path"]).parent)
        try:
            replay = export_isaac_replay(run_directory)
            initial_frame = Path(replay["preview_frame_paths"][0])
        except (FileNotFoundError, IndexError, OSError, ValueError) as error:
            raise PairingError("the Isaac recording has no usable camera frame") from error
        scenario = record["scenario"]
        generated: WorldPromptResult = self.prompt_factory(model).create_prompt(
            WorldPromptRequest(
                initial_frame, scenario["task"], scenario["seed"], scenario["parameters"],
                scenario["hazards"], objective,
            )
        )
        pair_id = str(uuid4())
        prepared = {
            "pair_id": pair_id,
            "isaac_run_id": isaac_run_id,
            "seed": scenario["seed"],
            "initial_frame_path": str(initial_frame),
            "prompt": generated.prompt,
            "prompt_response_id": generated.response_id,
            "objective": objective,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._prepared_path(pair_id).parent.mkdir(parents=True, exist_ok=True)
        self._prepared_path(pair_id).write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**prepared, "seed_image_url": f"/api/pair-captures/{pair_id}/seed"}

    def seed_image(self, pair_id: str) -> Path:
        prepared = self._prepared(pair_id)
        path = Path(prepared["initial_frame_path"])
        if not path.is_file():
            raise PairingError("the prepared Isaac seed frame is no longer available")
        return path

    def finalize(self, pair_id: str, media: bytes, *, content_type: str) -> dict:
        if not media:
            raise PairingError("the Reactor recording is empty")
        if len(media) > 100 * 1024 * 1024:
            raise PairingError("the Reactor recording exceeds the 100 MiB limit")
        prepared = self._prepared(pair_id)
        isaac_payload = self.store.get_experiment(prepared["isaac_run_id"])
        if isaac_payload is None:
            raise PairingError("the source Isaac recording no longer exists")
        isaac_record = _record_from_store(isaac_payload)
        reactor_run_id = str(uuid4())
        run_directory = self.runs_root / "reactor-paired" / pair_id / reactor_run_id
        run_directory.mkdir(parents=True, exist_ok=False)
        extension = ".mp4" if "mp4" in content_type else ".webm"
        video_path = run_directory / f"reactor_recording{extension}"
        video_path.write_bytes(media)
        reactor_scenario = Scenario(
            environment="reactor/lingbot-world-2", task=isaac_record.scenario.task,
            seed=isaac_record.scenario.seed,
            parameters={"prompt": prepared["prompt"], "seed_image_path": prepared["initial_frame_path"]},
        )
        trajectory_path = run_directory / "trajectory.jsonl"
        trajectory_path.write_text(
            json.dumps({"record_type": "initial_observation", "observation": {"simulation_time": 0, "sensor_refs": [str(video_path)], "state": {"backend_kind": "generative_video", "pair_id": pair_id}}}) + "\n",
            encoding="utf-8",
        )
        evaluation = EvaluationResult(False, False, None, Severity.NONE, evidence_refs=(str(video_path),))
        (run_directory / "scenario.json").write_text(json.dumps(reactor_scenario.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_directory / "metadata.json").write_text(json.dumps({"run_id": reactor_run_id, "backend": "reactor/lingbot-world-2"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_directory / "result.json").write_text(json.dumps(evaluation.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_directory / "summary.json").write_text(json.dumps({"pair_id": pair_id, "prompt": prepared["prompt"], "prompt_response_id": prepared["prompt_response_id"], "initial_frame_path": prepared["initial_frame_path"], "objective": prepared["objective"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        reactor_record = ExperimentRecord(reactor_run_id, reactor_scenario, "reactor/lingbot-world-2", str(trajectory_path), evaluation, datetime.now(UTC).isoformat())
        specification = MatchedExperimentSpec(
            task=isaac_record.scenario.task, seed=isaac_record.scenario.seed,
            isaac_scenario=isaac_record.scenario, neural_scenario=reactor_scenario,
            action_alignment=ActionAlignment.SEMANTIC,
            alignment_note="Isaac robot motion and Reactor camera navigation are semantically aligned only.",
            pair_id=pair_id,
        )
        matched = MatchedExperiment(specification, isaac_record, reactor_record)
        comparison = PlanCComparator().compare(matched)
        comparison_path = PairedDatasetRecorder(self.runs_root / "paired").record(matched, comparison)
        self.store.upsert_experiment(reactor_record, run_directory=run_directory)
        self.store.register_artifact("experiment", reactor_run_id, "video", video_path, {"source": "browser_webrtc_recording"})
        self.store.register_artifact("experiment", reactor_run_id, "metadata", run_directory / "summary.json")
        self.store.upsert_pair({"matched_experiment": matched.to_dict(), "comparison": comparison.to_dict()}, comparison_path)
        self.store.register_artifact("pair", pair_id, "metadata", comparison_path)
        return {"pair_id": pair_id, "reactor_run_id": reactor_run_id, "comparison_status": comparison.status, "comparison_url": f"/pairs/{pair_id}"}

    def _prepared_path(self, pair_id: str) -> Path:
        return self.runs_root / "pending_pairs" / pair_id / "prepared.json"

    def _prepared(self, pair_id: str) -> dict:
        path = self._prepared_path(pair_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PairingError("paired capture was not prepared or has expired") from error


def _record_from_store(payload: dict) -> ExperimentRecord:
    evaluation = payload["evaluation"]
    trajectory_ref = payload.get("trajectory_path") or payload.get("trajectory_ref")
    if not trajectory_ref:
        raise PairingError("experiment record has no trajectory reference")
    return ExperimentRecord(
        payload["run_id"], Scenario.from_dict(payload["scenario"]), payload["backend"],
        trajectory_ref, EvaluationResult(
            evaluation["task_success"], evaluation["environmental_failure"], evaluation.get("failure_type"),
            Severity(evaluation["severity"]), terminal=evaluation.get("terminal", False),
            evidence_refs=tuple(evaluation.get("evidence_refs", ())),
        ), payload["created_at"],
    )
