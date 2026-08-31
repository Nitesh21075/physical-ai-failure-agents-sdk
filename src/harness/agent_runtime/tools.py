"""Narrow model-visible tool surface; no shell or generic filesystem tools."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from agents import FunctionTool, function_tool
from agents.tool_context import ToolContext
from pydantic import ValidationError

from harness.agent_runtime.context import AgentRuntimeContext
from harness.agent_runtime.iro_spec import IROBuildStore, IROSceneSpec, iro_capability_catalog
from harness.agent_runtime.scenario_spec import (
    ScenarioBuildStore,
    ScenarioSpec,
    scenario_capability_catalog,
)
from harness.comparison.plan_c import (
    ActionAlignment,
    MatchedExperiment,
    MatchedExperimentSpec,
    PlanCComparator,
)
from harness.pairing import PairingError, _record_from_store
from harness.research.campaign import IterationState
from harness.research.visual_assessment import (
    OpenAIResponsesVisualAssessor,
    VisualComparisonRequest,
)
from harness.schemas import Scenario


def _compact_iteration(iteration: dict[str, Any] | None) -> dict[str, Any] | None:
    if not iteration:
        return None
    return {
        key: iteration.get(key)
        for key in (
            "iteration_id",
            "ordinal",
            "state",
            "isaac_run_id",
            "reactor_run_id",
            "plan_c_pair_id",
            "comparison_status",
            "human_review_state",
            "error",
            "updated_at",
        )
    }


@function_tool
def get_campaign_state(ctx: ToolContext[AgentRuntimeContext]) -> dict[str, Any]:
    """Read campaign objective, budgets, state, instructions, latest iteration, and recent events."""
    local = ctx.context
    campaign = local.campaign_store.get_campaign(local.campaign_id)
    if campaign is None:
        raise KeyError(f"unknown campaign: {local.campaign_id}")
    events = local.campaign_store.list_events(local.campaign_id)
    return {
        "campaign_id": local.campaign_id,
        "objective": campaign["objective"],
        "experiment_budget": campaign["experiment_budget"],
        "experiments_used": campaign["experiments_used"],
        "experiments_remaining": campaign["experiment_budget"] - campaign["experiments_used"],
        "status": campaign["status"],
        "pending_operator_instructions": local.campaign_store.pending_instructions(
            local.campaign_id
        ),
        "latest_iteration": _compact_iteration(
            local.campaign_store.latest_iteration(local.campaign_id)
        ),
        "recent_events": events[-local.config.max_recent_events :],
    }


@function_tool
def inspect_mine_world(ctx: ToolContext[AgentRuntimeContext]) -> dict[str, Any]:
    """Inspect the authoritative mine_v1 manifest and report only verified capabilities."""
    path = ctx.context.config.mine_manifest_path
    if not path.is_file():
        raise FileNotFoundError(f"mine manifest is unavailable: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    zones = manifest.get("failure_zones", {})
    legacy_zones = manifest.get("hazard_zones", {})
    return {
        "world_id": manifest.get("world_id"),
        "description": manifest.get("description"),
        "robot_type": "NVIDIA Nova Carter articulated wheeled rover; authored proxy fallback exists",
        "stable_prim_ids": manifest.get("interactive_prims", {}),
        "cameras": manifest.get("cameras", {}),
        "failure_zones": {
            name: {
                "root": zone.get("root"),
                "physics_status": zone.get("physics_status"),
                "causal_chain": zone.get("causal_chain") or zone.get("mechanism"),
            }
            for name, zone in {**legacy_zones, **zones}.items()
        },
        "supported_interaction_recipes": manifest.get("rover_interactions", {}),
        "executable_experiments": {
            "roof_support": {
                "status": "verified real Isaac Sim 6.0.1 Nova Carter wheel-actuated experiment",
                "scenario_tool": "validate_scenario -> build_scenario -> run_scenario",
                "tunable_parameters": [
                    "linear_velocity_mps",
                    "angular_velocity_radps",
                    "control_steps",
                    "robot start offset",
                    "support offset",
                    "support/falling-body mass",
                    "support friction",
                    "camera and lighting presets",
                    "approved prop placements",
                    "seed",
                ],
                "acceptance_note": (
                    "physics-only start/support offsets, masses, friction, steering, and one "
                    "static prop passed a live ScenarioSpec run; new RTX combinations and "
                    "dynamic props remain pending"
                ),
            }
        },
        "warnings": [
            "Only roof_support is an executable mine_v1 hazard template in ScenarioSpec v1.",
            "Rockfall and debris zones are authored but not executable agent tools.",
        ],
    }


@function_tool
def inspect_isaac_capabilities(ctx: ToolContext[AgentRuntimeContext]) -> dict[str, Any]:
    """Inspect the LLM's bounded Isaac Sim capabilities and executable world catalog.

    This reports what the agent can actually request through tools, not general
    capabilities available in the Isaac Sim GUI or arbitrary Python scripts.
    """
    root = ctx.context.config.project_root
    worlds: dict[str, Any] = {}
    for world_id, (stage_relative, default_zone) in sorted(
        ctx.context.mine_isaac_service.WORLD_ROUTES.items()
    ):
        manifest_path = root / Path(stage_relative).parent / "manifest.json"
        if not manifest_path.is_file():
            worlds[world_id] = {
                "status": "unavailable",
                "reason": f"missing manifest: {manifest_path}",
            }
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        zone = manifest.get("failure_zones", {}).get(default_zone, {})
        worlds[world_id] = {
            "status": "executable" if (root / stage_relative).is_file() else "unavailable",
            "description": manifest.get("description"),
            "stage": stage_relative,
            "failure_zone": default_zone,
            "physics_status": zone.get("physics_status"),
            "causal_chain": zone.get("causal_chain"),
            "native_isaac_usd": manifest.get("composition", {}).get(
                "environment_is_native_isaac_usd"
            ),
            "offline_ready": manifest.get("composition", {}).get("environment_collected_locally"),
            "packaging_note": manifest.get("composition", {}).get("local_packaging_status"),
            "presentation_status": manifest.get("composition", {}).get("presentation_status"),
        }
    return {
        "executor": "fixed Isaac Sim 6.0.1 Docker runner",
        "robot": "NVIDIA Nova Carter, wheel-velocity articulation control only",
        "agent_can": [
            "inspect declared worlds and prior indexed evidence",
            "select one of the explicitly allowlisted experiment tools",
            "choose bounded speed, duration, and seed",
            "receive measured body poses, collapse metrics, camera gates, and artifact paths",
            "prepare a Reactor comparison only from a quality-gated real Isaac frame",
        ],
        "agent_cannot": [
            "run arbitrary Python, shell, Docker, or Omniverse commands",
            "teleport the robot or directly toggle a collapse",
            "edit USD source stages",
            "invent frames, contacts, motion, or measurements",
            "execute more than one new Isaac run per ordinary research step",
        ],
        "worlds": worlds,
    }


@function_tool
def list_scenario_capabilities(ctx: ToolContext[AgentRuntimeContext]) -> dict[str, Any]:
    """List the exact typed Isaac scenario capabilities available to this agent.

    The catalog distinguishes executable adapters from unavailable controller
    families and reports all parameter bounds, approved assets, sensors, and
    run limits. It describes the harness, not every API Isaac Sim supports.
    """
    catalog = scenario_capability_catalog(ctx.context.config.project_root)
    catalog["synthetic_scene_generation"] = iro_capability_catalog()
    return catalog


@function_tool
def validate_iro_scene(
    ctx: ToolContext[AgentRuntimeContext], scene: IROSceneSpec
) -> dict[str, Any]:
    """Validate and normalize a bounded keyless IRO scene without launching Isaac.

    Unknown fields, arbitrary assets or paths, excessive object/frame counts,
    unsafe distributions, and unsupported outputs are rejected by the schema.
    """
    del ctx
    return {
        "status": "valid",
        "scene_digest": scene.digest(),
        "normalized_scene": scene.normalized(),
        "execution_cost": {"isaac_runs": 1, "repeat_count": 1},
        "evidence_scope": (
            "synthetic scene and annotation generation; not physical-failure evidence"
        ),
    }


@function_tool
def build_iro_scene(ctx: ToolContext[AgentRuntimeContext], scene: IROSceneSpec) -> dict[str, Any]:
    """Compile a bounded scene into trusted IRO YAML below runs/iro-builds.

    The compiler owns every YAML key and expression. The model supplies no
    paths, source text, Python, shell, Docker flags, or Omniverse commands.
    """
    return IROBuildStore(ctx.context.config.runs_root).build(scene)


@function_tool(timeout=1000.0, timeout_behavior="raise_exception")
async def run_iro_scene(ctx: ToolContext[AgentRuntimeContext], scene_id: str) -> dict[str, Any]:
    """Run one previously built scene through local Isaac IRO without an NVIDIA API key.

    This consumes the same one-Isaac-run budget as a physical experiment. Only
    a tamper-checked build UUID is accepted; this is synthetic-data evidence,
    not a robot experiment or physical-failure outcome.
    """
    local = ctx.context
    spec, _, payload = IROBuildStore(local.config.runs_root).load(scene_id)
    for record in local.experiment_store.list_experiments():
        if (
            record["backend"] == "isaac_iro"
            and record["scenario"].get("scenario_digest") == payload["scene_digest"]
        ):
            raise ValueError(
                "this exact IRO scene already has generated evidence; change the specification "
                "or explain why a repeat is scientifically required"
            )
    local.claim_isaac_budget()
    local.campaign_store.transition_campaign(local.campaign_id, "running")
    iteration_id = local.campaign_store.begin_iteration(
        local.campaign_id,
        {
            "runtime": "openai_agents_sdk",
            "backend": "isaac_iro",
            "scene_id": scene_id,
            "scene_digest": payload["scene_digest"],
            "scene_spec": spec.normalized(),
        },
    )
    local.campaign_store.transition_iteration(iteration_id, IterationState.RUNNING_ISAAC)
    try:
        result = await asyncio.to_thread(local.iro_service.run_built_scene, scene_id)
        local.campaign_store.record_isaac_run(iteration_id, result["run_id"])
        local.campaign_store.transition_iteration(iteration_id, IterationState.RECORDED)
        return result
    except Exception as error:
        local.campaign_store.transition_iteration(
            iteration_id, IterationState.FAILED, error=str(error)[:1000]
        )
        raise


@function_tool
def inspect_iro_run(ctx: ToolContext[AgentRuntimeContext], run_id: str) -> dict[str, Any]:
    """Inspect one indexed IRO run's generated artifacts and exact evidence scope."""
    try:
        run_id = str(UUID(run_id))
    except (TypeError, ValueError) as error:
        raise ValueError("run_id must be a UUID") from error
    record = ctx.context.experiment_store.get_experiment(run_id)
    if record is None or record["backend"] != "isaac_iro" or not record.get("run_directory"):
        raise KeyError(f"indexed IRO run is unavailable: {run_id}")
    run_directory = ctx.context.require_under_runs(record["run_directory"])
    summary_path = run_directory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"IRO summary is unavailable: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = ctx.context.experiment_store.artifacts_for("experiment", run_id)
    return {
        "run_id": run_id,
        "scene_id": summary["scene_id"],
        "scene_digest": summary["scene_digest"],
        "extension_version": summary["extension_version"],
        "api_key_used": summary["api_key_used"],
        "output_counts": summary["output_counts"],
        "image_refs": [item["path"] for item in artifacts if item["kind"] == "image"],
        "evidence_scope": summary["evidence_scope"],
    }


@function_tool
def _validate_scenario_schema(
    ctx: ToolContext[AgentRuntimeContext], scenario: ScenarioSpec
) -> dict[str, Any]:
    """Validate and normalize one bounded ScenarioSpec without launching Isaac.

    Pydantic rejects unknown fields, incompatible world/hazard combinations,
    out-of-bounds physics, poses, asset placements, and controllers before this
    function executes.
    """
    del ctx
    return {
        "status": "valid",
        "scenario_digest": scenario.digest(),
        "normalized_scenario": scenario.normalized(),
        "execution_cost": {"isaac_runs": 1, "repeat_count": 1},
    }


async def _validate_scenario_with_diagnostics(
    ctx: ToolContext[AgentRuntimeContext], arguments: str
) -> str:
    """Keep the strict schema while returning safe, actionable semantic errors."""
    del ctx
    try:
        payload = json.loads(arguments)
        if not isinstance(payload, dict) or not isinstance(payload.get("scenario"), dict):
            raise TypeError("arguments must contain one scenario object")
        scenario = ScenarioSpec.model_validate(payload["scenario"])
    except ValidationError as error:
        errors = [
            {
                "path": ".".join(str(part) for part in item["loc"]),
                "message": item["msg"],
            }
            for item in error.errors(include_url=False, include_input=False)
        ]
        return json.dumps({"status": "invalid", "validation_errors": errors})
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return json.dumps({"status": "invalid", "validation_errors": [{"message": str(error)}]})
    return json.dumps(
        {
            "status": "valid",
            "scenario_digest": scenario.digest(),
            "normalized_scenario": scenario.normalized(),
            "execution_cost": {"isaac_runs": 1, "repeat_count": 1},
        }
    )


validate_scenario = FunctionTool(
    name="validate_scenario",
    description=_validate_scenario_schema.description,
    params_json_schema=_validate_scenario_schema.params_json_schema,
    on_invoke_tool=_validate_scenario_with_diagnostics,
    strict_json_schema=True,
)


@function_tool
def build_scenario(ctx: ToolContext[AgentRuntimeContext], scenario: ScenarioSpec) -> dict[str, Any]:
    """Persist one validated, execution-ready ScenarioSpec under the runs directory.

    This performs no simulation and never edits a source USD. The Isaac worker
    later compiles it into a run-owned derived entry and session layer.
    """
    return ScenarioBuildStore(ctx.context.config.runs_root).build(scenario)


async def _run_built_scenario(
    ctx: ToolContext[AgentRuntimeContext], scenario_id: str
) -> dict[str, Any]:
    local = ctx.context
    spec, path, payload = ScenarioBuildStore(local.config.runs_root).load(scenario_id)
    for record in local.experiment_store.list_experiments():
        if (
            record["backend"] == "isaac_sim"
            and record["scenario"].get("scenario_digest") == payload["scenario_digest"]
        ):
            raise ValueError(
                "this exact validated scenario already has Isaac evidence; change the specification or explain why a repeat is scientifically required"
            )
    local.claim_isaac_budget()
    local.campaign_store.transition_campaign(local.campaign_id, "running")
    iteration_id = local.campaign_store.begin_iteration(
        local.campaign_id,
        {
            "runtime": "openai_agents_sdk",
            "scenario_id": scenario_id,
            "scenario_digest": payload["scenario_digest"],
            "scenario_spec": spec.normalized(),
        },
    )
    local.campaign_store.transition_iteration(iteration_id, IterationState.RUNNING_ISAAC)
    try:
        result = await asyncio.to_thread(
            local.mine_isaac_service.run_scenario,
            spec,
            compiled_spec_path=path,
        )
        local.campaign_store.record_isaac_run(iteration_id, result["run_id"])
        local.campaign_store.transition_iteration(iteration_id, IterationState.RECORDED)
        return result
    except Exception as error:
        local.campaign_store.transition_iteration(
            iteration_id, IterationState.FAILED, error=str(error)[:1000]
        )
        raise


@function_tool(timeout=950.0, timeout_behavior="raise_exception")
async def run_scenario(ctx: ToolContext[AgentRuntimeContext], scenario_id: str) -> dict[str, Any]:
    """Run one previously built ScenarioSpec in real Isaac Sim.

    The expensive call consumes the ordinary one-Isaac-run-per-research-step
    budget. Only the immutable built specification identified by scenario_id is
    accepted; no paths, Python, shell, Docker flags, or USD code are accepted.
    """
    return await _run_built_scenario(ctx, scenario_id)


@function_tool
def inspect_simulation_checkpoint(
    ctx: ToolContext[AgentRuntimeContext],
    run_id: str,
    checkpoint: Literal["initial", "final"],
) -> dict[str, Any]:
    """Inspect synchronized state and saved sensor references at an initial or final checkpoint."""
    try:
        run_id = str(UUID(run_id))
    except (TypeError, ValueError) as error:
        raise ValueError("run_id must be a UUID") from error
    record = ctx.context.experiment_store.get_experiment(run_id)
    if record is None or record["backend"] != "isaac_sim" or not record.get("run_directory"):
        raise KeyError(f"indexed Isaac run is unavailable: {run_id}")
    run_directory = ctx.context.require_under_runs(record["run_directory"])
    trajectory_path = run_directory / "trajectory.jsonl"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory is unavailable: {trajectory_path}")
    records = [
        json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    selected = (
        records[0]["observation"]
        if checkpoint == "initial"
        else records[-1]["result"]["observation"]
    )
    return {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "simulation_time": selected["simulation_time"],
        "state": selected["state"],
        "sensor_refs": selected.get("sensor_refs", []),
        "sensor_streams": selected.get("sensor_streams", {}),
        "events": [] if checkpoint == "initial" else records[-1]["result"].get("events", []),
    }


@function_tool
def compare_isaac_runs(ctx: ToolContext[AgentRuntimeContext], run_ids: list[str]) -> dict[str, Any]:
    """Compare measured outcomes from two to eight indexed Isaac runs without rerunning them."""
    if not 2 <= len(run_ids) <= 8:
        raise ValueError("run_ids must contain between 2 and 8 runs")
    comparisons = []
    for raw_run_id in run_ids:
        try:
            run_id = str(UUID(raw_run_id))
        except (TypeError, ValueError) as error:
            raise ValueError("every run_id must be a UUID") from error
        record = ctx.context.experiment_store.get_experiment(run_id)
        if record is None or record["backend"] != "isaac_sim":
            raise KeyError(f"indexed Isaac run is unavailable: {run_id}")
        evaluation = record.get("evaluation") or {}
        comparisons.append(
            {
                "run_id": run_id,
                "world_id": record["scenario"].get("environment"),
                "scenario_digest": record["scenario"].get("scenario_digest"),
                "parameters": record["scenario"].get("parameters", {}),
                "environmental_failure": evaluation.get("environmental_failure"),
                "failure_type": evaluation.get("failure_type"),
                "metrics": evaluation.get("metrics", {}),
            }
        )
    return {
        "comparison_scope": "Isaac/PhysX simulation evidence; not real-world truth",
        "runs": comparisons,
    }


@function_tool
def get_recent_experiments(
    ctx: ToolContext[AgentRuntimeContext], limit: int = 5, world_or_zone: str | None = None
) -> list[dict[str, Any]]:
    """Return compact recent scientific records, optionally filtered by world or zone text."""
    if isinstance(limit, bool) or not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    needle = world_or_zone.casefold() if world_or_zone else None
    summaries: list[dict[str, Any]] = []
    pairs = ctx.context.experiment_store.list_pairs()
    pairs_by_run = {
        run_id: pair for pair in pairs for run_id in (pair["isaac_run_id"], pair["reactor_run_id"])
    }
    for record in ctx.context.experiment_store.list_experiments():
        scenario = record["scenario"]
        searchable = json.dumps(scenario, sort_keys=True).casefold()
        if needle and needle not in searchable:
            continue
        pair = pairs_by_run.get(record["run_id"])
        evaluation = record.get("evaluation") or {}
        summaries.append(
            {
                "run_id": record["run_id"],
                "backend": record["backend"],
                "task": record["task"],
                "key_parameters": scenario.get("parameters", {}),
                "task_success": evaluation.get("task_success"),
                "environmental_failure": evaluation.get("environmental_failure"),
                "failure_type": evaluation.get("failure_type"),
                "pair_id": pair.get("pair_id") if pair else None,
                "comparison_status": pair.get("comparison_status") if pair else None,
                "human_review": pair.get("review_state") if pair else None,
            }
        )
        if len(summaries) >= limit:
            break
    return summaries


@function_tool
def inspect_isaac_run(ctx: ToolContext[AgentRuntimeContext], run_id: str) -> dict[str, Any]:
    """Inspect measured physics and visual-readiness evidence for one indexed Isaac UUID run.

    Paths come only from the authoritative experiment index and must resolve
    below the configured runs directory. This tool cannot inspect arbitrary
    files supplied by the model.
    """
    try:
        run_id = str(UUID(run_id))
    except (TypeError, ValueError) as error:
        raise ValueError("run_id must be a UUID") from error
    record = ctx.context.experiment_store.get_experiment(run_id)
    if record is None or record["backend"] != "isaac_sim":
        raise KeyError(f"indexed Isaac run is unavailable: {run_id}")
    if not record.get("run_directory"):
        raise FileNotFoundError("indexed Isaac run has no artifact directory")
    run_directory = ctx.context.require_under_runs(record["run_directory"])
    summary_path = run_directory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Isaac summary is unavailable: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "world_id": summary.get("world_id"),
        "failure_zone": summary.get("failure_zone"),
        "parameters": record["scenario"].get("parameters", {}),
        "environmental_failure": record["evaluation"].get("environmental_failure"),
        "failure_type": record["evaluation"].get("failure_type"),
        "physics_metrics": record["evaluation"].get("metrics", {}),
        "pre_actuation_stability_gate": summary.get("pre_actuation_stability_gate"),
        "visual_evidence_gate": summary.get("visual_evidence_gate"),
        "camera_frame_count_by_role": summary.get("camera_frame_count_by_role", {}),
        "rover_pose_before": summary.get("rover_pose_before"),
        "rover_pose_after": summary.get("rover_pose_after"),
        "source_stage_unchanged": summary.get("source_stage_unchanged"),
        "source_stage_sha256_before": summary.get("source_stage_sha256_before"),
        "source_stage_sha256_after": summary.get("source_stage_sha256_after"),
        "authored_scenario_overrides": summary.get("authored_scenario_overrides"),
        "reactor_seed_available": bool(summary.get("reactor_seed_manifest")),
    }


async def _run_bounded_isaac_world(
    ctx: ToolContext[AgentRuntimeContext],
    *,
    world_id: str,
    rover_linear_velocity_mps: float,
    control_steps: int,
    seed: int,
) -> dict[str, Any]:
    local = ctx.context
    local.mine_isaac_service._validate(rover_linear_velocity_mps, control_steps, seed)
    previous = local.experiment_store.list_experiments()
    signature = {
        "world_id": world_id,
        "rover_linear_velocity_mps": float(rover_linear_velocity_mps),
        "control_steps": control_steps,
        "seed": seed,
    }
    for record in previous:
        parameters = record["scenario"].get("parameters", {})
        existing = {
            **parameters,
            "world_id": record["scenario"].get("environment"),
            "seed": record["scenario"].get("seed"),
        }
        if record["backend"] == "isaac_sim" and all(
            existing.get(key) == value for key, value in signature.items()
        ):
            raise ValueError(
                "this exact parameter configuration already exists; provide a scientific reason and choose a changed configuration"
            )
    local.claim_isaac_budget()
    local.campaign_store.transition_campaign(local.campaign_id, "running")
    iteration_id = local.campaign_store.begin_iteration(
        local.campaign_id, {"runtime": "openai_agents_sdk", "requested_parameters": signature}
    )
    local.campaign_store.transition_iteration(iteration_id, IterationState.RUNNING_ISAAC)
    try:
        result = await asyncio.to_thread(
            local.mine_isaac_service.run,
            rover_linear_velocity_mps=float(rover_linear_velocity_mps),
            control_steps=control_steps,
            seed=seed,
            world_id=world_id,
        )
        local.campaign_store.record_isaac_run(iteration_id, result["run_id"])
        local.campaign_store.transition_iteration(iteration_id, IterationState.RECORDED)
        return result
    except Exception as error:
        local.campaign_store.transition_iteration(
            iteration_id, IterationState.FAILED, error=str(error)[:1000]
        )
        raise


@function_tool
async def run_mine_roof_support_experiment(
    ctx: ToolContext[AgentRuntimeContext],
    rover_linear_velocity_mps: float,
    control_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run one real Isaac Sim 6.0.1 Nova Carter roof-support experiment.

    The rover moves only through wheel velocity targets. Valid speed is 0.1
    to 0.8 m/s and control_steps is 60 to 900. The call is expensive and uses
    the ordinary one-Isaac-run-per-research-step budget.
    """
    return await _run_bounded_isaac_world(
        ctx,
        world_id="mine_v1",
        rover_linear_velocity_mps=rover_linear_velocity_mps,
        control_steps=control_steps,
        seed=seed,
    )


@function_tool
async def run_warehouse_rack_collapse_experiment(
    ctx: ToolContext[AgentRuntimeContext],
    rover_linear_velocity_mps: float,
    control_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run one real native-USD warehouse rack-collapse experiment in Isaac Sim 6.0.1.

    Nova Carter starts at the declared danger-cell approach pose and moves only
    through wheel velocity targets toward the marginal rack support. Valid
    speed is 0.1 to 0.8 m/s and control_steps is 60 to 900. PhysX determines
    whether the loaded beam falls; the tool does not toggle or animate failure.
    """
    return await _run_bounded_isaac_world(
        ctx,
        world_id="warehouse_danger_v1",
        rover_linear_velocity_mps=rover_linear_velocity_mps,
        control_steps=control_steps,
        seed=seed,
    )


def _validated_pair_id(pair_id: str) -> str:
    try:
        return str(UUID(pair_id))
    except (TypeError, ValueError) as error:
        raise ValueError("pair_id must be a UUID") from error


@function_tool
async def prepare_reactor_comparison(
    ctx: ToolContext[AgentRuntimeContext],
    isaac_run_id: str,
    research_objective: str,
) -> dict[str, Any]:
    """Prepare a real browser Reactor comparison from an indexed Isaac run and stop for capture.

    This uses the real initial Isaac camera frame, creates a bounded world-model prompt,
    and returns a pending pair. It never invents or automatically supplies Reactor evidence.
    """
    if not research_objective.strip():
        raise ValueError("research_objective must not be empty")
    prepared = await asyncio.to_thread(
        ctx.context.paired_capture.prepare,
        isaac_run_id,
        objective=research_objective.strip(),
        model=ctx.context.model,
    )
    ctx.context.campaign_store.record_plan_c_pair(
        ctx.context.campaign_id, isaac_run_id, prepared["pair_id"]
    )
    seed_path = ctx.context.paired_capture.seed_image(prepared["pair_id"])
    return {
        "pair_id": prepared["pair_id"],
        "isaac_run_id": isaac_run_id,
        "seed_image_available": seed_path.is_file() and seed_path.stat().st_size > 0,
        "seed_image_url": prepared["seed_image_url"],
        "prompt": prepared["prompt"],
        "status": "WAITING_FOR_REACTOR_CAPTURE",
        "gui_route": f"/reactor?pair_id={prepared['pair_id']}",
    }


@function_tool
def get_pair_status(ctx: ToolContext[AgentRuntimeContext], pair_id: str) -> dict[str, Any]:
    """Check whether a prepared pair has real Reactor media, comparison, assessment, or review."""
    pair_id = _validated_pair_id(pair_id)
    pair = ctx.context.experiment_store.get_pair(pair_id)
    pending_path = ctx.context.config.runs_root / "pending_pairs" / pair_id / "prepared.json"
    if pair is None:
        if not pending_path.is_file():
            raise PairingError("paired capture was not prepared or has expired")
        prepared = json.loads(pending_path.read_text(encoding="utf-8"))
        return {
            "pair_id": pair_id,
            "status": "pending",
            "isaac_run_id": prepared["isaac_run_id"],
            "reactor_recording_saved": False,
            "comparison_exists": False,
            "visual_assessment_exists": False,
            "human_review": None,
            "gui_route": f"/reactor?pair_id={pair_id}",
        }
    reactor = ctx.context.experiment_store.get_experiment(pair["reactor_run_id"])
    media = (
        ctx.context.experiment_store.artifacts_for("experiment", pair["reactor_run_id"])
        if reactor
        else []
    )
    real_media = [
        item
        for item in media
        if item["kind"] == "video"
        and Path(item["path"]).is_file()
        and Path(item["path"]).stat().st_size > 0
    ]
    return {
        "pair_id": pair_id,
        "status": "recorded" if real_media else "needs_human_review",
        "isaac_run_id": pair["isaac_run_id"],
        "reactor_run_id": pair["reactor_run_id"],
        "reactor_recording_saved": bool(real_media),
        "comparison_exists": Path(pair["comparison_path"]).is_file(),
        "comparison_status": pair["comparison_status"],
        "visual_assessment_exists": pair["visual_event_type"] is not None,
        "human_review": pair["review_state"],
        "comparison_route": f"/pairs/{pair_id}",
    }


def _sample_video_frames(video_path: Path, output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    pattern = output_directory / "frame_%02d.png"
    completed = subprocess.run(
        [
            _ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "fps=1",
            "-frames:v",
            "4",
            str(pattern),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    frames = tuple(sorted(output_directory.glob("frame_*.png")))
    if completed.returncode != 0 or not frames:
        raise RuntimeError(f"could not decode Reactor recording: {completed.stderr[-1000:]}")
    return frames


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError("video assessment requires ffmpeg or imageio-ffmpeg") from error


def _matched_from_pair_payload(payload: dict[str, Any]) -> MatchedExperiment:
    matched = payload["matched_experiment"]
    spec = matched["specification"]
    return MatchedExperiment(
        MatchedExperimentSpec(
            task=spec["task"],
            seed=spec["seed"],
            isaac_scenario=Scenario.from_dict(spec["isaac_scenario"]),
            neural_scenario=Scenario.from_dict(spec["neural_scenario"]),
            action_alignment=ActionAlignment(spec["action_alignment"]),
            alignment_note=spec["alignment_note"],
            shared_action_sequence_ref=spec.get("shared_action_sequence_ref"),
            pair_id=spec["pair_id"],
        ),
        _record_from_store(matched["isaac_record"]),
        _record_from_store(matched["neural_record"]),
    )


@function_tool
async def assess_and_compare_pair(
    ctx: ToolContext[AgentRuntimeContext], pair_id: str
) -> dict[str, Any]:
    """Assess actual saved Isaac/Reactor frames for structural_collapse and persist Plan C.

    Isaac physical evaluation remains unchanged. If real media cannot be decoded or the
    visual assessor cannot decide, return needs_human_review instead of guessing.
    """
    pair_id = _validated_pair_id(pair_id)
    pair = ctx.context.experiment_store.get_pair(pair_id)
    if pair is None:
        raise PairingError("the Reactor recording has not been saved; complete /reactor first")
    comparison_path = ctx.context.require_under_runs(pair["comparison_path"])
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    reactor_artifacts = ctx.context.experiment_store.artifacts_for(
        "experiment", pair["reactor_run_id"]
    )
    videos = [Path(item["path"]) for item in reactor_artifacts if item["kind"] == "video"]
    videos = [path for path in videos if path.is_file() and path.stat().st_size > 0]
    if not videos:
        return {
            "pair_id": pair_id,
            "status": "needs_human_review",
            "reason": "no decodable Reactor recording is registered",
        }
    isaac_artifacts = ctx.context.experiment_store.artifacts_for("experiment", pair["isaac_run_id"])
    isaac_frames = tuple(
        Path(item["path"])
        for item in isaac_artifacts
        if item["kind"] == "image"
        and Path(item["path"]).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        and Path(item["path"]).is_file()
    )
    if not isaac_frames:
        return {
            "pair_id": pair_id,
            "status": "needs_human_review",
            "reason": "no real Isaac camera frames are registered",
        }
    reactor = ctx.context.experiment_store.get_experiment(pair["reactor_run_id"])
    if reactor is None or not reactor["run_directory"]:
        return {
            "pair_id": pair_id,
            "status": "needs_human_review",
            "reason": "Reactor experiment record is incomplete",
        }
    output_directory = (
        ctx.context.require_under_runs(reactor["run_directory"]) / "media" / "assessment_frames"
    )
    try:
        reactor_frames = await asyncio.to_thread(_sample_video_frames, videos[0], output_directory)
        assessment = await asyncio.to_thread(
            OpenAIResponsesVisualAssessor(ctx.context.model).assess,
            VisualComparisonRequest(
                "structural_collapse",
                (isaac_frames[0], isaac_frames[-1]) if len(isaac_frames) > 1 else isaac_frames,
                (reactor_frames[0], reactor_frames[-1])
                if len(reactor_frames) > 1
                else reactor_frames,
            ),
        )
    except Exception as error:  # noqa: BLE001 - all assessor/provider failures require human review
        ctx.context.campaign_store.record_event(
            ctx.context.campaign_id,
            "visual_assessment_needs_human_review",
            {"pair_id": pair_id, "error": str(error)[:1000]},
        )
        return {"pair_id": pair_id, "status": "needs_human_review", "reason": str(error)[:500]}
    matched = _matched_from_pair_payload(payload)
    comparison = PlanCComparator().compare(matched, assessment.world_model_assessment)
    payload["comparison"] = comparison.to_dict()
    payload["visual_assessment_provenance"] = assessment.to_dict()
    comparison_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ctx.context.experiment_store.upsert_pair(payload, comparison_path)
    for frame in reactor_frames:
        ctx.context.experiment_store.register_artifact(
            "experiment",
            pair["reactor_run_id"],
            "image",
            frame,
            {"source": "ffmpeg_assessment_sample"},
        )
    ctx.context.campaign_store.record_pair_comparison(
        ctx.context.campaign_id, pair_id, pair["reactor_run_id"], comparison.status.value
    )
    observed = assessment.world_model_assessment.observed
    return {
        "pair_id": pair_id,
        "reactor_run_id": pair["reactor_run_id"],
        "reactor_event_observed": observed,
        "confidence": assessment.world_model_assessment.confidence,
        "evidence_frame_refs": list(assessment.world_model_assessment.evidence_refs),
        "assessor_provenance": assessment.world_model_assessment.assessor,
        "plan_c_status": comparison.status.value,
        "reason": comparison.reason,
        "needs_human_review": observed is None or comparison.status.value == "inconclusive",
    }


AGENT_TOOLS = [
    get_campaign_state,
    list_scenario_capabilities,
    inspect_mine_world,
    get_recent_experiments,
    inspect_isaac_run,
    inspect_iro_run,
    validate_scenario,
    build_scenario,
    run_scenario,
    validate_iro_scene,
    build_iro_scene,
    run_iro_scene,
    inspect_simulation_checkpoint,
    compare_isaac_runs,
    prepare_reactor_comparison,
    get_pair_status,
    assess_and_compare_pair,
]
