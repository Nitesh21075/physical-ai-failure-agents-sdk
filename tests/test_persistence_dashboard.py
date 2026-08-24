import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from harness.dashboard.app import create_app
from harness.media import export_isaac_replay, normalize_reactor_media
from harness.pairing import PairedCaptureService
from harness.persistence import ExperimentStore, ReviewState, reindex_runs
from harness.research.world_prompt import WorldPromptResult
from harness.schemas import EvaluationResult, ExperimentRecord, Scenario, Severity


def _comparison_payload(root: Path) -> dict:
    isaac_dir = root / "isaac" / "run-isaac"
    reactor_dir = root / "reactor"
    isaac_dir.mkdir(parents=True)
    reactor_dir.mkdir()
    trajectory = isaac_dir / "trajectory.jsonl"
    trajectory.write_text(
        json.dumps({"record_type": "initial_observation", "observation": {"simulation_time": 0}})
        + "\n"
        + json.dumps({"record_type": "step", "action": {"name": "move"}, "result": {"simulation_time": 1, "done": True, "events": [{"event_type": "structural_collapse"}]}})
        + "\n",
        encoding="utf-8",
    )
    frame = reactor_dir / "frame.png"
    Image.new("RGB", (16, 12), "black").save(frame)
    scenario = {"scenario_id": "scenario-1", "task": "reach", "seed": 4, "parameters": {}, "hazards": {}}
    return {
        "matched_experiment": {
            "specification": {"pair_id": "pair-1", "task": "reach", "seed": 4, "action_alignment": "semantic", "alignment_note": "visual steering", "isaac_scenario": {**scenario, "environment": "isaac_sim"}, "neural_scenario": {**scenario, "environment": "reactor/lingbot-world-2"}},
            "isaac_record": {"run_id": "isaac-1", "backend": "isaac_sim", "scenario": {**scenario, "environment": "isaac_sim"}, "trajectory_ref": str(trajectory), "created_at": "2026-01-01T00:00:00+00:00", "evaluation": {"task_success": False, "environmental_failure": True, "failure_type": "structural_collapse", "severity": "high", "terminal": True}},
            "neural_record": {"run_id": "reactor-1", "backend": "reactor/lingbot-world-2", "scenario": {**scenario, "environment": "reactor/lingbot-world-2"}, "trajectory_ref": str(reactor_dir / "summary.json"), "created_at": "2026-01-01T00:00:00+00:00", "evaluation": {"task_success": False, "environmental_failure": False, "failure_type": None, "severity": "none", "terminal": False, "evidence_refs": [str(frame)]}},
        },
        "comparison": {"pair_id": "pair-1", "status": "inconclusive", "action_alignment": "semantic", "physics_environmental_failure": True, "physics_failure_type": "structural_collapse", "reason": "no decision", "evidence_refs": [], "visual_assessment": {"event_type": "structural_collapse", "observed": None, "confidence": 0.0, "assessor": "reviewer", "evidence_refs": []}},
    }


def test_store_persists_pair_review_and_dashboard_view(tmp_path: Path):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    payload = _comparison_payload(tmp_path)
    for record in (payload["matched_experiment"]["isaac_record"], payload["matched_experiment"]["neural_record"]):
        store.upsert_experiment(record, run_directory=Path(record["trajectory_ref"]).parent)
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(payload), encoding="utf-8")
    store.upsert_pair(payload, comparison)
    store.register_artifact("experiment", "reactor-1", "image", tmp_path / "reactor" / "frame.png")
    store.set_review_state("pair-1", ReviewState.INCONCLUSIVE)

    client = TestClient(create_app(store))
    assert client.get("/api/pairs").json()[0]["review_state"] == "inconclusive"
    detail = client.get("/api/pairs/pair-1").json()
    assert detail["authority"]["isaac"] == "PHYSICS-GROUNDED SIMULATION"
    assert detail["isaac_timeline"][-1]["kind"] == "termination"
    pair_page = client.get("/pairs/pair-1")
    assert pair_page.status_code == 200
    assert "Physics-grounded simulation — reference" in pair_page.text
    assert "Neural-world visual evidence — not physics ground truth" in pair_page.text
    assert 'id="pair-selector"' in pair_page.text
    assert 'src="/static/recording-selector.js"' in pair_page.text
    assert "<details>" in pair_page.text
    assert client.get("/api/experiments/isaac-1").json()["authority"] == "PHYSICS-GROUNDED SIMULATION"
    assert client.put("/api/pairs/pair-1/review", json={"review_state": "valid_discrepancy"}).status_code == 200


def test_live_reactor_tab_and_token_guard(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("REACTOR_API_KEY", raising=False)
    client = TestClient(create_app(ExperimentStore(tmp_path / "experiments.sqlite3")))

    page = client.get("/reactor")
    assert page.status_code == 200
    assert "Reactor control room" in page.text
    assert 'src="/static/reactor-live.js"' in page.text
    assert client.get("/api/reactor/live-config").json()["enabled"] is False
    assert client.post("/api/reactor/token").status_code == 503


def test_paired_capture_persists_browser_video_as_a_plan_c_pair(tmp_path: Path):
    runs = tmp_path / "runs"
    run_dir = runs / "isaac" / "run-1"
    camera = runs / "isaac" / "camera" / "capture" / "rgb_000000.png"
    camera.parent.mkdir(parents=True); run_dir.mkdir(parents=True)
    varied = np.zeros((64, 64, 4), dtype=np.uint8)
    varied[:, :, :3] = np.random.default_rng(7).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    varied[:, :, 3] = 255
    Image.fromarray(varied[:, :, :3], "RGB").save(camera)
    scenario = Scenario(
        environment="mine_v1", task="mine_roof_support_interaction", seed=7,
        parameters={"rover_linear_velocity_mps": 0.25, "control_steps": 180},
        hazards={"zone": "RoofSupportZone", "event": "structural_collapse"},
    )
    (run_dir / "trajectory.jsonl").write_text(
        json.dumps({"record_type": "initial_observation", "observation": {"sensor_refs": [str(camera)]}}) + "\n"
    )
    (run_dir / "reactor_seed.json").write_text(
        json.dumps(
            {
                "seed_image_path": "/workspace/project/runs/isaac/camera/capture/rgb_000000.png",
                "camera_role": "tracking",
            }
        ),
        encoding="utf-8",
    )
    record = ExperimentRecord(
        "isaac-run", scenario, "isaac_sim", str(run_dir / "trajectory.jsonl"),
        EvaluationResult(False, True, "structural_collapse", Severity.HIGH, terminal=True),
        "2026-01-01T00:00:00+00:00",
    )
    store = ExperimentStore(runs / "experiments.sqlite3")
    store.upsert_experiment(record, run_directory=run_dir)

    class FakePromptModel:
        def create_prompt(self, request):
            assert request.initial_frame_path.is_file()
            assert request.isaac_hazards["zone"] == "RoofSupportZone"
            return WorldPromptResult("A blue robot approaches an unstable support beam.", "resp_prompt")

    service = PairedCaptureService(store, runs, prompt_factory=lambda _model: FakePromptModel())
    prepared = service.prepare("isaac-run", objective="Compare collapse visuals", model="test-model")
    assert Path(prepared["initial_frame_path"]).is_file()
    client = TestClient(create_app(store))
    loaded = client.get(f"/api/pair-captures/{prepared['pair_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["prompt"] == prepared["prompt"]
    assert "initial_frame_path" not in loaded.json()
    assert client.get("/api/pair-captures/not-a-uuid").status_code == 404
    result = service.finalize(prepared["pair_id"], b"webm-bytes", content_type="video/webm")

    pair = store.get_pair(result["pair_id"])
    assert pair is not None
    assert pair["isaac_run_id"] == "isaac-run"
    reactor = store.get_experiment(result["reactor_run_id"])
    assert reactor is not None
    assert any(item["kind"] == "video" for item in store.artifacts_for("experiment", reactor["run_id"]))


def test_dashboard_exposes_agents_sdk_campaign_controls(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    client = TestClient(create_app(ExperimentStore(tmp_path / "experiments.sqlite3")))
    created = client.post(
        "/api/agent/campaigns",
        json={"objective": "Find the roof-support failure boundary", "experiment_budget": 2},
    )
    assert created.status_code == 201
    campaign_id = created.json()["campaign_id"]
    listed = client.get("/api/agent/campaigns").json()
    assert [row["campaign_id"] for row in listed] == [campaign_id]
    detail = client.get(f"/api/agent/campaigns/{campaign_id}").json()
    assert detail["latest_iteration"] is None
    assert detail["model_provider"] == "openai_agents_sdk"
    assert any(event["event_type"] == "campaign_created" for event in detail["events"])


def test_reindex_rebuilds_standard_and_paired_run_index(tmp_path: Path):
    runs = tmp_path / "runs"
    run_dir = runs / "isaac-batch" / "run-1"
    camera_dir = runs / "isaac-batch" / "camera" / "hash"
    run_dir.mkdir(parents=True); camera_dir.mkdir(parents=True)
    (run_dir / "scenario.json").write_text(json.dumps({"scenario_id": "s-1", "environment": "isaac_sim", "task": "reach", "seed": 3, "parameters": {}, "hazards": {}}))
    (run_dir / "metadata.json").write_text(json.dumps({"run_id": "run-1", "backend": "isaac_sim"}))
    (run_dir / "result.json").write_text(json.dumps({"task_success": False, "environmental_failure": True, "failure_type": "collapse", "severity": "high", "terminal": True}))
    raw = camera_dir / "rgb_000000.npy"; np.save(raw, np.zeros((12, 16, 4), dtype=np.uint8))
    (run_dir / "trajectory.jsonl").write_text(json.dumps({"record_type": "initial_observation", "observation": {"simulation_time": 0, "sensor_refs": [str(raw)]}}) + "\n")
    payload = _comparison_payload(tmp_path)
    pair_path = runs / "paired" / "pair-1" / "comparison.json"; pair_path.parent.mkdir(parents=True)
    pair_path.write_text(json.dumps(payload), encoding="utf-8")

    store = ExperimentStore(tmp_path / "index.sqlite3")
    counts = reindex_runs(store, runs)

    assert counts["pairs"] == 1
    artifacts = store.artifacts_for("experiment", "run-1")
    assert any(item["kind"] == "video" for item in artifacts)
    assert store.get_pair("pair-1") is not None


def test_media_export_and_reactor_manifest_preserve_authority_boundary(tmp_path: Path):
    run_dir = tmp_path / "runs" / "isaac" / "run-1"
    camera_dir = run_dir.parent / "camera" / "capture"
    run_dir.mkdir(parents=True)
    camera_dir.mkdir(parents=True)
    np.save(camera_dir / "invalid.npy", np.zeros((4,), dtype=np.uint8))
    np.save(camera_dir / "rgb.npy", np.zeros((12, 16, 4), dtype=np.uint8))
    (run_dir / "trajectory.jsonl").write_text(
        json.dumps({"record_type": "initial_observation", "observation": {"sensor_refs": [str(camera_dir / "invalid.npy"), str(camera_dir / "rgb.npy")]}}) + "\n"
    )
    replay = export_isaac_replay(run_dir)
    assert Path(replay["video_path"]).is_file()
    assert [Path(path).name for path in replay["preview_frame_paths"]] == ["frame_0000.png"]

    reactor_dir = tmp_path / "reactor"
    frames = reactor_dir / "frames"
    frames.mkdir(parents=True)
    frame = frames / "chunk-0001.png"
    Image.new("RGB", (16, 12), "black").save(frame)
    (reactor_dir / "summary.json").write_text(json.dumps({"chunks": [{"index": 1, "elapsed_ms": 42}]}))
    manifest = normalize_reactor_media(reactor_dir, [frame])
    assert Path(manifest["derived_video_path"]).is_file()
    assert manifest["media_sequence"] == [{"position": 0, "path": str(frame)}]
    assert manifest["chunk_metadata"] == [{"index": 1, "elapsed_ms": 42}]
    assert "synchronized" in manifest["timing_note"]
