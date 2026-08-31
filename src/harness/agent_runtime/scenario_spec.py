"""Typed, bounded scenario language for model-authored Isaac experiments."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


Vector2 = Annotated[list[float], Field(min_length=2, max_length=2)]
Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Color3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class WorldId(StrEnum):
    MINE_V1 = "mine_v1"
    MINE_V2_SUBT = "mine_v2_subt"
    WAREHOUSE_DANGER_V1 = "warehouse_danger_v1"


class HazardTemplate(StrEnum):
    ROOF_SUPPORT = "roof_support"
    RACK_COLLAPSE = "rack_collapse"


class CameraPreset(StrEnum):
    PHYSICS_ONLY = "physics_only"
    RGB_SEMANTIC_LOW = "rgb_semantic_low"
    RGB_SEMANTIC_STANDARD = "rgb_semantic_standard"


class LightingPreset(StrEnum):
    LOW_LIGHT = "low_light"
    STANDARD = "standard"
    BRIGHT = "bright"


class AssetId(StrEnum):
    SAFETY_BARRIER_BOX = "safety_barrier_box"
    RUBBLE_BLOCK = "rubble_block"


class RobotSpec(StrictModel):
    asset_id: Literal["nova_carter"] = "nova_carter"
    start_pose_preset: Literal["manifest_default"] = "manifest_default"
    start_offset_xyz_m: Vector3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class FixedVelocityControllerSpec(StrictModel):
    controller_id: Literal["fixed_velocity"] = "fixed_velocity"
    linear_velocity_mps: float = Field(ge=0.1, le=0.8)
    angular_velocity_radps: float = Field(default=0.0, ge=-0.6, le=0.6)
    control_steps: int = Field(ge=60, le=900)


class GoalPoseControllerSpec(StrictModel):
    controller_id: Literal["goal_pose"] = "goal_pose"
    target_position_xy_m: Vector2
    target_heading_rad: float | None = Field(default=None, ge=-3.1416, le=3.1416)
    max_linear_velocity_mps: float = Field(default=0.3, ge=0.1, le=0.6)
    max_angular_velocity_radps: float = Field(default=0.6, ge=0.1, le=0.6)
    position_tolerance_m: float = Field(default=0.1, ge=0.05, le=0.3)
    heading_tolerance_rad: float = Field(default=0.15, ge=0.05, le=0.5)
    max_control_steps: int = Field(default=360, ge=60, le=900)
    stagnation_steps: int = Field(default=120, ge=30, le=240)

    @model_validator(mode="after")
    def validate_stagnation_window(self) -> GoalPoseControllerSpec:
        if self.stagnation_steps >= self.max_control_steps:
            raise ValueError("stagnation_steps must be less than max_control_steps")
        return self


ControllerSpec = Annotated[
    FixedVelocityControllerSpec | GoalPoseControllerSpec,
    Field(discriminator="controller_id"),
]


class HazardSpec(StrictModel):
    template: HazardTemplate
    support_offset_xyz_m: Vector3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    support_mass_kg: float | None = Field(default=None, ge=1.0, le=500.0)
    falling_body_mass_kg: float | None = Field(default=None, ge=1.0, le=2000.0)
    support_static_friction: float | None = Field(default=None, ge=0.05, le=1.5)
    support_dynamic_friction: float | None = Field(default=None, ge=0.05, le=1.5)

    @model_validator(mode="after")
    def validate_friction_pair(self) -> HazardSpec:
        if (
            self.support_static_friction is not None
            and self.support_dynamic_friction is not None
            and self.support_dynamic_friction > self.support_static_friction
        ):
            raise ValueError("support_dynamic_friction cannot exceed support_static_friction")
        return self


class EnvironmentSpec(StrictModel):
    lighting_preset: LightingPreset = LightingPreset.STANDARD


class SensorRigSpec(StrictModel):
    preset: CameraPreset = CameraPreset.RGB_SEMANTIC_LOW
    capture_every_steps: int = Field(default=15, ge=5, le=120)
    preferred_camera_role: Literal["manifest_default", "ego", "tracking", "witness"] = (
        "manifest_default"
    )


class ApprovedAssetPlacement(StrictModel):
    asset_id: AssetId
    position_xyz_m: Vector3
    scale_xyz_m: Vector3 = Field(default_factory=lambda: [0.5, 0.5, 0.5])
    dynamic: bool = False
    mass_kg: float = Field(default=20.0, ge=1.0, le=200.0)
    friction: float = Field(default=0.6, ge=0.05, le=1.5)


class StructuralBodySpec(StrictModel):
    """One compiler-owned cuboid in an experimental structural assembly."""

    body_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    semantic_role: Literal["impact_support", "support", "falling_body", "load", "obstacle"]
    position_xyz_m: Vector3
    dimensions_xyz_m: Vector3
    rotation_rpy_deg: Vector3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    physics: Literal["static", "dynamic"]
    mass_kg: float = Field(default=50.0, ge=1.0, le=2000.0)
    friction: float = Field(default=0.5, ge=0.05, le=1.5)
    display_color_rgb: Color3 = Field(default_factory=lambda: [0.7, 0.7, 0.7])

    @model_validator(mode="after")
    def validate_body_geometry(self) -> StructuralBodySpec:
        if any(value < 0.1 or value > 5.0 for value in self.dimensions_xyz_m):
            raise ValueError("dimensions_xyz_m values must be within [0.1, 5.0]")
        if any(abs(value) > 45.0 for value in self.rotation_rpy_deg):
            raise ValueError("rotation_rpy_deg values must be within [-45, 45]")
        if any(value < 0.0 or value > 1.0 for value in self.display_color_rgb):
            raise ValueError("display_color_rgb values must be within [0, 1]")
        if self.semantic_role in {"impact_support", "falling_body"} and self.physics != "dynamic":
            raise ValueError(f"{self.semantic_role} must use dynamic physics")
        return self


class StructuralAssemblySpec(StrictModel):
    """Bounded model-directed structure compiled into a run-owned USD layer."""

    assembly_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    description: str = Field(min_length=10, max_length=500)
    bodies: list[StructuralBodySpec] = Field(min_length=3, max_length=10)
    collapse_threshold_m: float = Field(default=0.5, ge=0.2, le=3.0)

    @model_validator(mode="after")
    def validate_roles_and_ids(self) -> StructuralAssemblySpec:
        ids = [body.body_id for body in self.bodies]
        if len(ids) != len(set(ids)):
            raise ValueError("structural body_id values must be unique")
        roles = [body.semantic_role for body in self.bodies]
        for required in ("impact_support", "falling_body"):
            if roles.count(required) != 1:
                raise ValueError(f"structural assembly requires exactly one {required}")
        if not any(role == "support" for role in roles):
            raise ValueError("structural assembly requires at least one secondary support")
        return self


class ScenarioSpec(StrictModel):
    """One executable scenario from the allowlisted Isaac scenario language."""

    schema_version: Literal["1.0"] = "1.0"
    backend: Literal["isaac_sim"] = "isaac_sim"
    base_world: WorldId
    hazard: HazardSpec
    robot: RobotSpec = Field(default_factory=RobotSpec)
    controller: ControllerSpec
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    sensors: SensorRigSpec = Field(default_factory=SensorRigSpec)
    placed_assets: list[ApprovedAssetPlacement] = Field(default_factory=list, max_length=4)
    authored_structure: StructuralAssemblySpec | None = None
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    repeat_count: Literal[1] = 1

    @model_validator(mode="before")
    @classmethod
    def preserve_v1_fixed_velocity_specs(cls, value: object) -> object:
        if isinstance(value, dict) and isinstance(value.get("controller"), dict):
            controller = value["controller"]
            if "controller_id" not in controller:
                value = {**value, "controller": {"controller_id": "fixed_velocity", **controller}}
        return value

    @model_validator(mode="after")
    def validate_compatibility_and_regions(self) -> ScenarioSpec:
        expected_hazard = {
            WorldId.MINE_V1: HazardTemplate.ROOF_SUPPORT,
            WorldId.MINE_V2_SUBT: HazardTemplate.ROOF_SUPPORT,
            WorldId.WAREHOUSE_DANGER_V1: HazardTemplate.RACK_COLLAPSE,
        }[self.base_world]
        if self.hazard.template != expected_hazard:
            raise ValueError(
                f"{self.base_world.value} requires hazard template {expected_hazard.value}"
            )
        for name, offset, bound in (
            ("robot.start_offset_xyz_m", self.robot.start_offset_xyz_m, (1.0, 1.0, 0.25)),
            ("hazard.support_offset_xyz_m", self.hazard.support_offset_xyz_m, (0.5, 0.5, 0.25)),
        ):
            if any(abs(value) > maximum for value, maximum in zip(offset, bound, strict=True)):
                raise ValueError(f"{name} exceeds its approved XYZ offset bounds {bound}")
        regions = {
            WorldId.MINE_V1: ((20.0, 28.0), (-5.0, 5.0), (0.1, 4.0)),
            WorldId.MINE_V2_SUBT: ((20.0, 28.0), (-5.0, 5.0), (0.1, 4.0)),
            WorldId.WAREHOUSE_DANGER_V1: ((-5.0, 5.0), (-6.0, 6.0), (0.1, 5.0)),
        }[self.base_world]
        if isinstance(self.controller, GoalPoseControllerSpec):
            x_region, y_region, _ = regions
            target_x, target_y = self.controller.target_position_xy_m
            if not x_region[0] <= target_x <= x_region[1] or not y_region[0] <= target_y <= y_region[1]:
                raise ValueError(
                    "controller.target_position_xy_m is outside the approved XY region "
                    f"{(x_region, y_region)}"
                )
        for index, placement in enumerate(self.placed_assets):
            if any(value <= 0.0 or value > 2.0 for value in placement.scale_xyz_m):
                raise ValueError(f"placed_assets[{index}].scale_xyz_m must be within (0, 2]")
            if any(
                not lower <= value <= upper
                for value, (lower, upper) in zip(placement.position_xyz_m, regions, strict=True)
            ):
                raise ValueError(
                    f"placed_assets[{index}] is outside the approved placement region {regions}"
                )
        if self.authored_structure is not None:
            if self.base_world != WorldId.WAREHOUSE_DANGER_V1:
                raise ValueError("authored_structure is experimental and requires warehouse_danger_v1")
            if self.placed_assets:
                raise ValueError("placed_assets cannot be combined with authored_structure")
            for index, body in enumerate(self.authored_structure.bodies):
                if any(
                    not lower <= value <= upper
                    for value, (lower, upper) in zip(body.position_xyz_m, regions, strict=True)
                ):
                    raise ValueError(
                        f"authored_structure.bodies[{index}] is outside the approved region {regions}"
                    )
            impact = next(
                body
                for body in self.authored_structure.bodies
                if body.semantic_role == "impact_support"
            )
            falling = next(
                body
                for body in self.authored_structure.bodies
                if body.semantic_role == "falling_body"
            )
            supports = [
                body
                for body in self.authored_structure.bodies
                if body.semantic_role in {"impact_support", "support"}
            ]
            rover_start_x = self.robot.start_offset_xyz_m[0]
            rover_start_y = -1.65 + self.robot.start_offset_xyz_m[1]
            if abs(impact.position_xyz_m[0] - rover_start_x) > 0.6:
                raise ValueError("impact_support must be aligned within 0.6 m of rover start X")
            if not 0.6 <= impact.position_xyz_m[1] - rover_start_y <= 3.0:
                raise ValueError("impact_support must be 0.6-3.0 m ahead of the rover in +Y")
            falling_bottom = falling.position_xyz_m[2] - falling.dimensions_xyz_m[2] / 2.0
            contacting_supports = 0
            for support in supports:
                support_top = support.position_xyz_m[2] + support.dimensions_xyz_m[2] / 2.0
                overlaps_xy = all(
                    abs(support.position_xyz_m[axis] - falling.position_xyz_m[axis])
                    <= (support.dimensions_xyz_m[axis] + falling.dimensions_xyz_m[axis]) / 2.0
                    for axis in (0, 1)
                )
                if overlaps_xy and abs(support_top - falling_bottom) <= 0.15:
                    contacting_supports += 1
            if contacting_supports < 2:
                raise ValueError(
                    "falling_body must initially contact impact_support and at least one support"
                )
        return self

    def normalized(self) -> dict:
        return self.model_dump(mode="json")

    def digest(self) -> str:
        encoded = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


CAMERA_PRESETS = {
    CameraPreset.PHYSICS_ONLY: {"enabled": False, "height": 180, "width": 320},
    CameraPreset.RGB_SEMANTIC_LOW: {"enabled": True, "height": 180, "width": 320},
    CameraPreset.RGB_SEMANTIC_STANDARD: {
        "enabled": True,
        "height": 360,
        "width": 640,
    },
}


WORLD_ROUTES: dict[WorldId, tuple[str, HazardTemplate]] = {
    WorldId.MINE_V1: ("assets/worlds/mine_v1/mine_world.usda", HazardTemplate.ROOF_SUPPORT),
    WorldId.MINE_V2_SUBT: (
        "assets/worlds/mine_v2_subt/mine_world.usda",
        HazardTemplate.ROOF_SUPPORT,
    ),
    WorldId.WAREHOUSE_DANGER_V1: (
        "assets/worlds/warehouse_danger_v1/warehouse_world.usda",
        HazardTemplate.RACK_COLLAPSE,
    ),
}


def scenario_capability_catalog(project_root: Path) -> dict:
    worlds = {}
    for world_id, (stage_relative, hazard) in WORLD_ROUTES.items():
        manifest_path = project_root / Path(stage_relative).parent / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        )
        worlds[world_id.value] = {
            "status": "executable" if (project_root / stage_relative).is_file() else "unavailable",
            "description": manifest.get("description"),
            "hazard_templates": [hazard.value],
            "approved_placement_region_xyz_m": {
                "mine_v1": [[20.0, 28.0], [-5.0, 5.0], [0.1, 4.0]],
                "mine_v2_subt": [[20.0, 28.0], [-5.0, 5.0], [0.1, 4.0]],
                "warehouse_danger_v1": [[-5.0, 5.0], [-6.0, 6.0], [0.1, 5.0]],
            }[world_id.value],
            "physics_status": manifest.get("failure_zones", {})
            .get(hazard.value, {})
            .get("physics_status"),
            "model_directed_structural_authoring": (
                {
                    "status": "experimental",
                    "scope": "3-10 compiler-owned cuboids in a run-owned session layer",
                    "body_physics": ["static", "dynamic"],
                    "semantic_roles": [
                        "impact_support",
                        "support",
                        "falling_body",
                        "load",
                        "obstacle",
                    ],
                    "body_fields": {
                        "required": [
                            "body_id",
                            "semantic_role",
                            "position_xyz_m",
                            "dimensions_xyz_m",
                            "rotation_rpy_deg",
                            "physics",
                            "mass_kg",
                            "friction",
                            "display_color_rgb",
                        ],
                        "body_id_pattern": "lowercase letters, digits, and underscores",
                        "dimensions_xyz_m": [0.1, 5.0],
                        "rotation_rpy_deg": [-45.0, 45.0],
                        "mass_kg": [1.0, 2000.0],
                        "friction": [0.05, 1.5],
                        "display_color_rgb": [0.0, 1.0],
                    },
                    "geometry_contract": [
                        "impact_support is dynamic and aligned within 0.6 m of rover start X",
                        "impact_support is 0.6-3.0 m ahead of the rover in +Y",
                        "falling_body is dynamic and initially contacts impact_support",
                        "falling_body initially contacts at least one secondary support",
                    ],
                    "legacy_hazard_fields": (
                        "required by ScenarioSpec v1 strict JSON; ignored for authored_structure"
                    ),
                }
                if world_id == WorldId.WAREHOUSE_DANGER_V1
                else {"status": "unavailable"}
            ),
        }
    return {
        "scenario_schema_version": "1.0",
        "backends": {"isaac_sim": {"status": "executable", "version": "6.0.1"}},
        "worlds": worlds,
        "robots": {
            "nova_carter": {
                "status": "executable",
                "start_pose_presets": ["manifest_default"],
                "bounded_start_offset_xyz_m": [1.0, 1.0, 0.25],
            }
        },
        "controllers": {
            "fixed_velocity": {
                "status": "executable",
                "linear_velocity_mps": [0.1, 0.8],
                "angular_velocity_radps": [-0.6, 0.6],
                "control_steps": [60, 900],
            },
            "goal_pose": {
                "status": "executable",
                "adapter_version": "1.0",
                "target_position": "bounded to the selected world's approved XY region",
                "target_heading_rad": [-3.1416, 3.1416],
                "max_linear_velocity_mps": [0.1, 0.6],
                "max_angular_velocity_radps": [0.1, 0.6],
                "position_tolerance_m": [0.05, 0.3],
                "heading_tolerance_rad": [0.05, 0.5],
                "max_control_steps": [60, 900],
                "stagnation_steps": [30, 240],
                "termination_reasons": [
                    "goal_reached",
                    "stagnation_detected",
                    "max_control_steps_reached",
                ],
            },
            "navigation": {"status": "unavailable", "reason": "no navigation adapter registered"},
            "vla": {"status": "unavailable", "reason": "no VLA adapter registered"},
            "rl_policy": {"status": "unavailable", "reason": "no policy adapter registered"},
            "ros": {"status": "unavailable", "reason": "no ROS controller adapter registered"},
        },
        "sensors": {preset.value: config for preset, config in CAMERA_PRESETS.items()},
        "editable_physics": {
            "support_offset_xyz_m": [[-0.5, 0.5], [-0.5, 0.5], [-0.25, 0.25]],
            "support_mass_kg": [1.0, 500.0],
            "falling_body_mass_kg": [1.0, 2000.0],
            "support_friction": [0.05, 1.5],
        },
        "approved_assets": [asset.value for asset in AssetId],
        "lighting_presets": [preset.value for preset in LightingPreset],
        "observations": [
            "initial_state",
            "final_state",
            "RGB checkpoints",
            "semantic bounding boxes",
            "camera poses",
            "support and falling-body displacement",
            "collapse classification",
        ],
        "limits": {
            "repeat_count": 1,
            "max_placed_assets": 4,
            "ordinary_runs_per_research_step": 1,
            "source_worlds_are_immutable": True,
            "max_authored_structural_bodies": 10,
        },
    }


class ScenarioBuildStore:
    """Persists validated specifications below runs without compiling arbitrary code."""

    def __init__(self, runs_root: Path) -> None:
        self.root = runs_root / "scenario-builds"

    def build(self, spec: ScenarioSpec) -> dict:
        scenario_id = str(uuid4())
        directory = self.root / scenario_id
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "scenario_spec.json"
        payload = {
            "scenario_id": scenario_id,
            "scenario_digest": spec.digest(),
            "scenario_spec": spec.normalized(),
            "compile_status": "validated_execution_ready",
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**payload, "compiled_spec_path": str(path)}

    def load(self, scenario_id: str) -> tuple[ScenarioSpec, Path, dict]:
        try:
            scenario_id = str(UUID(scenario_id))
        except (TypeError, ValueError) as error:
            raise ValueError("scenario_id must be a UUID") from error
        path = (self.root / scenario_id / "scenario_spec.json").resolve()
        if self.root.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(f"built scenario is unavailable: {scenario_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = ScenarioSpec.model_validate(payload["scenario_spec"])
        if payload.get("scenario_digest") != spec.digest():
            raise ValueError("built scenario digest does not match its specification")
        return spec, path, payload
