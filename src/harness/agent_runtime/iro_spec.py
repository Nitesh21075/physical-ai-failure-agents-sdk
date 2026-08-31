"""Typed scene language and deterministic compiler for Isaac Replicator Object."""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

IRO_CONFIG_VERSION = "0.9.12"
IRO_EXTENSION_VERSION = "0.11.12"
IRO_OUTPUT_PLACEHOLDER = "__HARNESS_IRO_OUTPUT_PATH__"


class StrictIROModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class PrimitiveShape(StrEnum):
    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CONE = "cone"
    TORUS = "torus"


class PhysicsMode(StrEnum):
    COLLISION = "collision"
    RIGIDBODY = "rigidbody"


class IROPrimitiveGroup(StrictIROModel):
    """A bounded group of built-in IRO primitives, never an arbitrary asset."""

    label: str = Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_]*$")
    shape: PrimitiveShape
    count: int = Field(default=1, ge=1, le=12)
    physics: PhysicsMode = PhysicsMode.COLLISION
    size_cm: float = Field(default=30.0, ge=5.0, le=100.0)
    position_min_xyz_cm: Vector3
    position_max_xyz_cm: Vector3
    color_min_rgb: Vector3 = Field(default_factory=lambda: [0.2, 0.2, 0.2])
    color_max_rgb: Vector3 = Field(default_factory=lambda: [0.9, 0.9, 0.9])
    friction: float = Field(default=0.5, ge=0.05, le=1.5)
    initial_velocity_xyz_cmps: Vector3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    @model_validator(mode="after")
    def validate_ranges(self) -> IROPrimitiveGroup:
        for axis, (lower, upper) in enumerate(
            zip(self.position_min_xyz_cm, self.position_max_xyz_cm, strict=True)
        ):
            if lower > upper:
                raise ValueError(f"position minimum exceeds maximum on axis {axis}")
        for point_name, point in (
            ("position_min_xyz_cm", self.position_min_xyz_cm),
            ("position_max_xyz_cm", self.position_max_xyz_cm),
        ):
            if not (-200.0 <= point[0] <= 200.0 and -200.0 <= point[1] <= 200.0):
                raise ValueError(f"{point_name} XY values must be within the 500 cm room")
            if not self.size_cm / 2.0 <= point[2] <= 300.0:
                raise ValueError(f"{point_name} Z must keep the primitive above the floor")
        for name, color in (
            ("color_min_rgb", self.color_min_rgb),
            ("color_max_rgb", self.color_max_rgb),
        ):
            if any(not 0.0 <= channel <= 1.0 for channel in color):
                raise ValueError(f"{name} channels must be between 0 and 1")
        if any(
            lower > upper
            for lower, upper in zip(self.color_min_rgb, self.color_max_rgb, strict=True)
        ):
            raise ValueError("color_min_rgb cannot exceed color_max_rgb")
        if any(abs(value) > 1500.0 for value in self.initial_velocity_xyz_cmps):
            raise ValueError("initial velocity components must be within +/-1500 cm/s")
        if self.physics != PhysicsMode.RIGIDBODY and any(self.initial_velocity_xyz_cmps):
            raise ValueError("initial velocity requires rigidbody physics")
        return self


class IRORoomSpec(StrictIROModel):
    width_cm: float = Field(default=500.0, ge=300.0, le=800.0)
    depth_cm: float = Field(default=500.0, ge=300.0, le=800.0)
    has_walls: bool = True
    floor_color_rgb: Vector3 = Field(default_factory=lambda: [0.35, 0.38, 0.42])

    @model_validator(mode="after")
    def validate_color(self) -> IRORoomSpec:
        if any(not 0.0 <= channel <= 1.0 for channel in self.floor_color_rgb):
            raise ValueError("floor_color_rgb channels must be between 0 and 1")
        return self


class IROOutputSpec(StrictIROModel):
    rgb: Literal[True] = True
    descriptions: Literal[True] = True
    bounding_boxes_2d: bool = True
    bounding_boxes_3d: bool = False
    semantic_segmentation: bool = True
    instance_segmentation: bool = False
    depth: bool = False
    normals: bool = False
    usd: bool = False


class IROSceneSpec(StrictIROModel):
    """One safe, keyless IRO synthetic-scene request."""

    schema_version: Literal["1.0"] = "1.0"
    backend: Literal["isaac_iro"] = "isaac_iro"
    template: Literal["primitive_room"] = "primitive_room"
    room: IRORoomSpec = Field(default_factory=IRORoomSpec)
    objects: list[IROPrimitiveGroup] = Field(min_length=1, max_length=6)
    camera_preset: Literal["diagonal_overview"] = "diagonal_overview"
    light_intensity: float = Field(default=2000.0, ge=300.0, le=5000.0)
    gravity_cmps2: float = Field(default=981.0, ge=0.0, le=2000.0)
    inter_frame_time_seconds: float = Field(default=0.05, ge=0.0, le=3.0)
    frame_count: int = Field(default=1, ge=1, le=20)
    image_width: int = Field(default=640, ge=256, le=1280)
    image_height: int = Field(default=360, ge=256, le=720)
    outputs: IROOutputSpec = Field(default_factory=IROOutputSpec)
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    repeat_count: Literal[1] = 1

    @model_validator(mode="after")
    def validate_scene(self) -> IROSceneSpec:
        if sum(group.count for group in self.objects) > 32:
            raise ValueError("IRO scenes may contain at most 32 primitive instances")
        labels = [group.label for group in self.objects]
        if len(labels) != len(set(labels)):
            raise ValueError("IRO object-group labels must be unique")
        if self.image_width * self.image_height * self.frame_count > 12_000_000:
            raise ValueError("requested IRO image workload exceeds the bounded pixel budget")
        for group in self.objects:
            for point in (group.position_min_xyz_cm, group.position_max_xyz_cm):
                if abs(point[0]) > self.room.width_cm / 2.0 - 10.0:
                    raise ValueError(f"{group.label} position exceeds the room's X interior")
                if abs(point[1]) > self.room.depth_cm / 2.0 - 10.0:
                    raise ValueError(f"{group.label} position exceeds the room's Y interior")
        return self

    def normalized(self) -> dict:
        return self.model_dump(mode="json")

    def digest(self) -> str:
        payload = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def iro_capability_catalog() -> dict:
    return {
        "backend": "isaac_iro",
        "status": "executable",
        "isaac_sim_version": "6.0.1",
        "extension": "isaacsim.replicator.object.core",
        "extension_version": IRO_EXTENSION_VERSION,
        "requires_nvidia_api_key": False,
        "schema_version": "1.0",
        "templates": ["primitive_room"],
        "primitive_shapes": [shape.value for shape in PrimitiveShape],
        "physics_modes": [mode.value for mode in PhysicsMode],
        "editable": [
            "primitive groups and counts",
            "bounded position/color distributions",
            "primitive size, friction, and initial velocity",
            "collision or rigid-body behavior",
            "room dimensions, walls, floor color, gravity, and frame timing",
            "dome-light intensity",
            "RGB, boxes, segmentation, depth, normals, descriptions, and USD outputs",
        ],
        "limits": {
            "object_groups": [1, 6],
            "primitive_instances_total": [1, 32],
            "frames": [1, 20],
            "resolution": {"width": [256, 1280], "height": [256, 720]},
            "repeat_count": 1,
            "ordinary_isaac_runs_per_research_step": 1,
        },
        "not_exposed": [
            "arbitrary YAML or expressions",
            "arbitrary USD or filesystem paths",
            "Python, shell, Docker arguments, or Omniverse commands",
            "Chat IRO's NVIDIA-hosted language model",
            "robot controllers or general structural-fracture mechanics",
        ],
        "evidence_scope": (
            "synthetic scene and annotation generation; not a physical-failure outcome or "
            "real-world ground truth"
        ),
    }


def compile_iro_config(spec: IROSceneSpec, output_path: str = IRO_OUTPUT_PLACEHOLDER) -> dict:
    """Compile trusted values into IRO's installed 0.9.12 config contract."""
    spec = IROSceneSpec.model_validate(spec)
    config: dict = {
        "version": IRO_CONFIG_VERSION,
        "parent_config": "global",
        "num_frames": spec.frame_count,
        "seed": spec.seed,
        "output_path": output_path,
        "screen_width": spec.image_width,
        "screen_height": spec.image_height,
        "gravity": spec.gravity_cmps2,
        "friction": 0.5,
        "linear_damping": 0.1,
        "angular_damping": 0.1,
        "inter_frame_time": spec.inter_frame_time_seconds,
        "output_switches": {
            "images": True,
            "labels": spec.outputs.bounding_boxes_2d,
            "descriptions": True,
            "3d_labels": spec.outputs.bounding_boxes_3d,
            "segmentation": spec.outputs.semantic_segmentation,
            "depth": spec.outputs.depth,
            "normal": spec.outputs.normals,
            "instance_id_segmentation": spec.outputs.instance_segmentation,
            "usd": spec.outputs.usd,
            "caption": False,
            "florance2": False,
            "object_caption": False,
            "caption_like": False,
        },
        "default_camera": {
            "type": "camera",
            "camera_parameters": "$[/camera_parameters]",
            "transform_operators": [
                {"translate": [553.5384050722222, 553.5384050722221, 553.5383919795806]},
                {"rotateXYZ": [54.73561, 0.0, 135.0]},
            ],
        },
        "dome_light": {
            "type": "light",
            "subtype": "dome",
            "intensity": spec.light_intensity,
            "transform_operators": [{"rotateX": 270.0}],
        },
        "generated_floor": {
            "type": "geometry",
            "subtype": "cube",
            "physics": "collision",
            "color": list(spec.room.floor_color_rgb),
            "transform_operators": [
                {"translate": [0.0, 0.0, -5.0]},
                {"scale": [spec.room.width_cm / 100.0, spec.room.depth_cm / 100.0, 0.1]},
            ],
        },
    }
    if spec.room.has_walls:
        wall_height = 200.0
        wall_thickness = 10.0
        walls = {
            "wall_x_negative": [
                -spec.room.width_cm / 2.0,
                0.0,
                wall_height / 2.0,
                wall_thickness,
                spec.room.depth_cm,
            ],
            "wall_x_positive": [
                spec.room.width_cm / 2.0,
                0.0,
                wall_height / 2.0,
                wall_thickness,
                spec.room.depth_cm,
            ],
            "wall_y_negative": [
                0.0,
                -spec.room.depth_cm / 2.0,
                wall_height / 2.0,
                spec.room.width_cm,
                wall_thickness,
            ],
            "wall_y_positive": [
                0.0,
                spec.room.depth_cm / 2.0,
                wall_height / 2.0,
                spec.room.width_cm,
                wall_thickness,
            ],
        }
        for name, (x, y, z, width, depth) in walls.items():
            config[name] = {
                "type": "geometry",
                "subtype": "cube",
                "physics": "collision",
                "color": list(spec.room.floor_color_rgb),
                "transform_operators": [
                    {"translate": [x, y, z]},
                    {"scale": [width / 100.0, depth / 100.0, wall_height / 100.0]},
                ],
            }
    for group in spec.objects:
        item = {
            "count": group.count,
            "type": "geometry",
            "subtype": group.shape.value,
            "physics": group.physics.value,
            "tracked": True,
            "friction": group.friction,
            "color": {
                "distribution_type": "range",
                "start": list(group.color_min_rgb),
                "end": list(group.color_max_rgb),
            },
            "transform_operators": [
                {
                    "translate": {
                        "distribution_type": "range",
                        "start": list(group.position_min_xyz_cm),
                        "end": list(group.position_max_xyz_cm),
                    }
                },
                {"scale": [group.size_cm / 100.0] * 3},
            ],
        }
        if any(group.initial_velocity_xyz_cmps):
            item["initial_velocity"] = list(group.initial_velocity_xyz_cmps)
        config[group.label] = item
    return {"isaacsim.replicator.object": config}


class IROBuildStore:
    """Persist validated IRO specs and compiler output with tamper detection."""

    def __init__(self, runs_root: Path) -> None:
        self.root = runs_root / "iro-builds"

    def build(self, spec: IROSceneSpec) -> dict:
        spec = IROSceneSpec.model_validate(spec)
        scene_id = str(uuid4())
        directory = self.root / scene_id
        directory.mkdir(parents=True, exist_ok=False)
        config = compile_iro_config(spec)
        config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
        config_digest = hashlib.sha256(config_text.encode()).hexdigest()
        config_path = directory / "iro_config.yaml"
        manifest_path = directory / "build.json"
        config_path.write_text(config_text, encoding="utf-8")
        payload = {
            "scene_id": scene_id,
            "scene_digest": spec.digest(),
            "config_digest": config_digest,
            "scene_spec": spec.normalized(),
            "compiler": {
                "extension": "isaacsim.replicator.object.core",
                "extension_version": IRO_EXTENSION_VERSION,
                "config_version": IRO_CONFIG_VERSION,
            },
            "compile_status": "validated_execution_ready",
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            **payload,
            "compiled_config_path": str(config_path),
            "build_manifest_path": str(manifest_path),
        }

    def load(self, scene_id: str) -> tuple[IROSceneSpec, Path, dict]:
        try:
            scene_id = str(UUID(scene_id))
        except (TypeError, ValueError) as error:
            raise ValueError("scene_id must be a UUID") from error
        directory = (self.root / scene_id).resolve()
        if self.root.resolve() not in directory.parents:
            raise ValueError("scene_id resolves outside the IRO build store")
        manifest_path = directory / "build.json"
        config_path = directory / "iro_config.yaml"
        if not manifest_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"built IRO scene is unavailable: {scene_id}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        spec = IROSceneSpec.model_validate(payload["scene_spec"])
        config_text = config_path.read_text(encoding="utf-8")
        if payload.get("scene_digest") != spec.digest():
            raise ValueError("built IRO scene digest does not match its specification")
        if payload.get("config_digest") != hashlib.sha256(config_text.encode()).hexdigest():
            raise ValueError("compiled IRO config digest does not match its build manifest")
        expected = json.dumps(compile_iro_config(spec), indent=2, sort_keys=True) + "\n"
        if config_text != expected:
            raise ValueError("compiled IRO config does not match the trusted compiler output")
        if not re.fullmatch(r"[0-9a-f]{64}", payload["config_digest"]):
            raise ValueError("invalid compiled IRO config digest")
        return spec, config_path, payload
