from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from agents import SQLiteSession

from harness.agent_runtime.context import AgentRuntimeConfig, AgentRuntimeContext
from harness.agent_runtime.iro_service import IsaacIROToolService
from harness.agent_runtime.iro_spec import IROBuildStore, IROSceneSpec, compile_iro_config
from harness.agent_runtime.isaac_service import MineIsaacToolService
from harness.agent_runtime.researcher import create_researcher
from harness.agent_runtime.scenario_spec import CAMERA_PRESETS, ScenarioBuildStore, ScenarioSpec
from harness.agent_runtime.tools import validate_scenario
from harness.persistence.store import ExperimentStore
from harness.research.campaign import ResearchCampaignStore
from harness.structural_authoring import runtime_zone_for_scenario


def _context(tmp_path: Path, *, budget: int = 2) -> AgentRuntimeContext:
    root = tmp_path / "project"
    (root / "runs").mkdir(parents=True)
    database = root / "runs" / "experiments.sqlite3"
    campaigns = ResearchCampaignStore(database)
    campaign_id = campaigns.create_campaign("Find roof support boundary", experiment_budget=budget)
    return AgentRuntimeContext(
        campaign_id=campaign_id,
        config=AgentRuntimeConfig.for_project(root),
        experiment_store=ExperimentStore(database),
        campaign_store=campaigns,
        paired_capture=object(),
        mine_isaac_service=object(),
        iro_service=object(),
        model="test-model",
    )


@pytest.mark.parametrize("speed", [-1.0, 0.09, 0.81, 9.0])
def test_isaac_speed_rejected_before_launch(speed: float) -> None:
    with pytest.raises(ValueError, match="between 0.1 and 0.8"):
        MineIsaacToolService._validate(speed, 300, 1)


def test_one_isaac_experiment_per_turn_guard(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.claim_isaac_budget()
    with pytest.raises(RuntimeError, match="at most one"):
        context.claim_isaac_budget()


def test_runs_path_guard(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assert context.require_under_runs(context.config.runs_root / "safe.json").name == "safe.json"
    with pytest.raises(ValueError, match="underneath"):
        context.require_under_runs(tmp_path / "outside.json")


def test_complex_scenario_loop_has_bounded_recovery_turns(tmp_path: Path) -> None:
    assert _context(tmp_path).config.max_turns == 20


def test_agent_has_only_bounded_research_and_scenario_tools() -> None:
    names = [tool.name for tool in create_researcher("test-model").tools]
    assert names == [
        "get_campaign_state",
        "list_scenario_capabilities",
        "inspect_mine_world",
        "get_recent_experiments",
        "inspect_isaac_run",
        "inspect_iro_run",
        "validate_scenario",
        "build_scenario",
        "run_scenario",
        "validate_iro_scene",
        "build_iro_scene",
        "run_iro_scene",
        "inspect_simulation_checkpoint",
        "compare_isaac_runs",
        "prepare_reactor_comparison",
        "get_pair_status",
        "assess_and_compare_pair",
    ]
    assert not {"shell", "python", "docker", "filesystem"}.intersection(names)


def test_agent_tool_array_schemas_are_accepted_function_call_shapes() -> None:
    def arrays_without_items(value: object) -> list[dict]:
        if isinstance(value, dict):
            found = [value] if value.get("type") == "array" and "items" not in value else []
            return found + [
                item
                for child in value.values()
                for item in arrays_without_items(child)
            ]
        if isinstance(value, list):
            return [item for child in value for item in arrays_without_items(child)]
        return []

    for tool in create_researcher("test-model").tools:
        assert arrays_without_items(tool.params_json_schema) == [], tool.name


def _iro_scene() -> IROSceneSpec:
    return IROSceneSpec.model_validate(
        {
            "objects": [
                {
                    "label": "falling_blocks",
                    "shape": "cube",
                    "count": 3,
                    "physics": "rigidbody",
                    "size_cm": 25,
                    "position_min_xyz_cm": [-50, -50, 40],
                    "position_max_xyz_cm": [50, 50, 100],
                }
            ],
            "frame_count": 2,
            "image_width": 320,
            "image_height": 256,
        }
    )


def test_iro_spec_compiles_only_trusted_core_contract() -> None:
    spec = _iro_scene()
    config = compile_iro_config(spec)["isaacsim.replicator.object"]
    assert config["version"] == "0.9.12"
    assert config["falling_blocks"]["physics"] == "rigidbody"
    assert config["falling_blocks"]["count"] == 3
    assert config["output_switches"]["caption"] is False
    assert "usd_path" not in str(config)


def test_iro_spec_rejects_unsafe_ranges_and_workloads() -> None:
    payload = _iro_scene().normalized()
    payload["objects"][0]["position_max_xyz_cm"] = [500, 0, 100]
    with pytest.raises(ValueError, match="within the 500 cm room"):
        IROSceneSpec.model_validate(payload)
    payload = _iro_scene().normalized()
    payload["objects"] = [
        {**payload["objects"][0], "label": f"group_{index}", "count": 12} for index in range(3)
    ]
    with pytest.raises(ValueError, match="at most 32"):
        IROSceneSpec.model_validate(payload)


def test_iro_build_store_detects_config_tampering(tmp_path: Path) -> None:
    spec = _iro_scene()
    store = IROBuildStore(tmp_path / "runs")
    built = store.build(spec)
    loaded, config_path, payload = store.load(built["scene_id"])
    assert loaded == spec
    assert payload["scene_digest"] == spec.digest()
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace('"count": 3', '"count": 4'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config digest"):
        store.load(built["scene_id"])


def test_iro_command_uses_fixed_headless_experience_and_no_key(tmp_path: Path) -> None:
    root = tmp_path / "project"
    runs = root / "runs"
    runs.mkdir(parents=True)
    service = IsaacIROToolService(root, runs, ExperimentStore(runs / "experiments.sqlite3"))
    command = service._command("00000000-0000-0000-0000-000000000001")
    joined = " ".join(command)
    assert "isaacsim.exp.action_and_event_data_generation.base.kit" in joined
    assert "isaacsim.replicator.object.core" in joined
    assert "--no-window" in joined
    assert "--network none" in joined
    assert "API_KEY" not in joined
    assert "NVIDIA_API_KEY" not in joined


def test_isaac_world_routes_are_fixed_and_local() -> None:
    assert MineIsaacToolService.WORLD_ROUTES == {
        "mine_v1": ("assets/worlds/mine_v1/mine_world.usda", "roof_support"),
        "mine_v2_subt": (
            "assets/worlds/mine_v2_subt/mine_world.usda",
            "roof_support",
        ),
        "warehouse_danger_v1": (
            "assets/worlds/warehouse_danger_v1/warehouse_world.usda",
            "rack_collapse",
        ),
    }


def test_scenario_spec_rejects_incompatible_hazard_and_unsafe_placement() -> None:
    base = {
        "base_world": "mine_v1",
        "hazard": {"template": "roof_support"},
        "controller": {"linear_velocity_mps": 0.3, "control_steps": 180},
    }
    assert ScenarioSpec.model_validate(base).base_world == "mine_v1"
    with pytest.raises(ValueError, match="requires hazard template"):
        ScenarioSpec.model_validate({**base, "hazard": {"template": "rack_collapse"}})
    with pytest.raises(ValueError, match="outside the approved placement region"):
        ScenarioSpec.model_validate(
            {
                **base,
                "placed_assets": [
                    {
                        "asset_id": "rubble_block",
                        "position_xyz_m": [100.0, 0.0, 1.0],
                    }
                ],
            }
        )


def test_scenario_spec_accepts_only_bounded_registered_goal_pose() -> None:
    base = {
        "base_world": "mine_v1",
        "hazard": {"template": "roof_support"},
        "controller": {
            "controller_id": "goal_pose",
            "target_position_xy_m": [24.0, -0.9],
            "max_control_steps": 300,
            "stagnation_steps": 120,
        },
    }
    scenario = ScenarioSpec.model_validate(base)
    assert scenario.controller.controller_id == "goal_pose"
    with pytest.raises(ValueError, match="outside the approved XY region"):
        ScenarioSpec.model_validate(
            {
                **base,
                "controller": {
                    **base["controller"],
                    "target_position_xy_m": [100.0, -0.9],
                },
            }
        )
    with pytest.raises(ValueError, match="stagnation_steps"):
        ScenarioSpec.model_validate(
            {
                **base,
                "controller": {
                    **base["controller"],
                    "max_control_steps": 100,
                    "stagnation_steps": 100,
                },
            }
        )


def _authored_structure_scenario() -> dict:
    return {
        "base_world": "warehouse_danger_v1",
        "hazard": {"template": "rack_collapse"},
        "robot": {"start_offset_xyz_m": [0.0, 0.0, 0.0]},
        "controller": {
            "controller_id": "fixed_velocity",
            "linear_velocity_mps": 0.8,
            "control_steps": 450,
        },
        "authored_structure": {
            "assembly_id": "two_pole_platform",
            "description": "A loaded platform resting on one movable pole and one fixed pole.",
            "collapse_threshold_m": 0.5,
            "bodies": [
                {
                    "body_id": "movable_pole",
                    "semantic_role": "impact_support",
                    "position_xyz_m": [0.0, 0.0, 1.25],
                    "dimensions_xyz_m": [0.3, 0.4, 2.5],
                    "physics": "dynamic",
                    "mass_kg": 45.0,
                    "friction": 0.1,
                    "display_color_rgb": [0.95, 0.65, 0.05],
                },
                {
                    "body_id": "fixed_pole",
                    "semantic_role": "support",
                    "position_xyz_m": [1.4, 0.0, 1.25],
                    "dimensions_xyz_m": [0.3, 0.4, 2.5],
                    "physics": "static",
                },
                {
                    "body_id": "loaded_platform",
                    "semantic_role": "falling_body",
                    "position_xyz_m": [0.7, 0.0, 2.65],
                    "dimensions_xyz_m": [3.2, 1.2, 0.3],
                    "physics": "dynamic",
                    "mass_kg": 250.0,
                    "friction": 0.6,
                    "display_color_rgb": [0.25, 0.45, 0.85],
                },
            ],
        },
    }


def test_model_directed_structure_is_bounded_and_paths_are_compiler_owned() -> None:
    spec = ScenarioSpec.model_validate(_authored_structure_scenario())
    manifest_zone = {
        "root": "/World/FailureZones/RackCollapseZone",
        "witness_camera": {"prim": "/World/Sensors/RackCollapseCamera"},
    }
    runtime = runtime_zone_for_scenario(manifest_zone, spec.normalized())
    assert runtime["support"] == "/World/Experiment/GeneratedStructure/movable_pole"
    assert runtime["primary_falling_body"] == (
        "/World/Experiment/GeneratedStructure/loaded_platform"
    )
    assert runtime["disabled_source_hazard_root"] == manifest_zone["root"]
    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        ScenarioSpec.model_validate(
            {
                **_authored_structure_scenario(),
                "authored_structure": {
                    **_authored_structure_scenario()["authored_structure"],
                    "bodies": [
                        {
                            **_authored_structure_scenario()["authored_structure"]["bodies"][0],
                            "body_id": "../../unsafe",
                        },
                        *_authored_structure_scenario()["authored_structure"]["bodies"][1:],
                    ],
                },
            }
        )


def test_model_directed_structure_rejects_wrong_world_and_missing_roles() -> None:
    payload = _authored_structure_scenario()
    with pytest.raises(ValueError, match="requires warehouse_danger_v1"):
        ScenarioSpec.model_validate(
            {
                **payload,
                "base_world": "mine_v1",
                "hazard": {"template": "roof_support"},
            }
        )


def test_validate_scenario_tool_returns_actionable_geometry_errors() -> None:
    payload = _authored_structure_scenario()
    payload["authored_structure"]["bodies"][0]["position_xyz_m"][1] = 1.4
    result = asyncio.run(
        validate_scenario.on_invoke_tool(None, json.dumps({"scenario": payload}))
    )
    parsed = json.loads(result)
    assert parsed["status"] == "invalid"
    assert "0.6-3.0 m ahead" in parsed["validation_errors"][0]["message"]
    bodies = payload["authored_structure"]["bodies"]
    with pytest.raises(ValueError, match="exactly one falling_body"):
        ScenarioSpec.model_validate(
            {
                **payload,
                "authored_structure": {
                    **payload["authored_structure"],
                    "bodies": [
                        bodies[0],
                        bodies[1],
                        {**bodies[2], "semantic_role": "load"},
                    ],
                },
            }
        )
def test_physics_only_keeps_a_valid_dormant_camera_resolution() -> None:
    preset = CAMERA_PRESETS["physics_only"]
    assert preset == {"enabled": False, "height": 180, "width": 320}


def test_scenario_build_store_detects_tampering(tmp_path: Path) -> None:
    spec = ScenarioSpec.model_validate(
        {
            "base_world": "mine_v2_subt",
            "hazard": {"template": "roof_support", "support_mass_kg": 80.0},
            "controller": {
                "linear_velocity_mps": 0.25,
                "angular_velocity_radps": 0.1,
                "control_steps": 180,
            },
        }
    )
    store = ScenarioBuildStore(tmp_path / "runs")
    built = store.build(spec)
    loaded, path, payload = store.load(built["scenario_id"])
    assert loaded == spec
    assert payload["scenario_digest"] == spec.digest()
    content = path.read_text(encoding="utf-8").replace('"seed": 42', '"seed": 43')
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        store.load(built["scenario_id"])


def test_campaign_pair_linkage_is_persistent(tmp_path: Path) -> None:
    context = _context(tmp_path)
    iteration_id = context.campaign_store.begin_iteration(context.campaign_id, {})
    context.campaign_store.record_isaac_run(iteration_id, "isaac-run")
    context.campaign_store.record_plan_c_pair(context.campaign_id, "isaac-run", "pair-id")
    reopened = ResearchCampaignStore(context.config.database_path)
    latest = reopened.latest_iteration(context.campaign_id)
    assert latest["plan_c_pair_id"] == "pair-id"
    assert latest["state"] == "waiting_for_assessment"


def test_sqlite_agent_session_survives_reopen(tmp_path: Path) -> None:
    async def exercise() -> list[dict]:
        database = tmp_path / "agent_sessions.sqlite3"
        first = SQLiteSession("campaign-1", db_path=database)
        await first.add_items([{"role": "user", "content": "first hypothesis"}])
        first.close()
        reopened = SQLiteSession("campaign-1", db_path=database)
        items = await reopened.get_items()
        reopened.close()
        return items

    assert asyncio.run(exercise()) == [{"role": "user", "content": "first hypothesis"}]
