"""Typed, backend-independent contract for one mine-rover visual experiment.

The original composed USD stage is always an input. Simulator code writes any
sensor schema edits or experiment metadata to a per-run session layer instead.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MINE_WORLD_ID = "mine_v1"
ROVER_PRIM_PATH = "/World/Robot"
ROVER_CAMERA_PATH = "/World/Robot/VLACamera"
EGO_CAMERA_PATH = "/World/Sensors/RoverEgoCamera"
TRACKING_CAMERA_PATH = "/World/Sensors/RoverTrackingCamera"
WITNESS_CAMERA_PATH = "/World/Sensors/RoofSupportCamera"
REACTOR_SEED_CAMERA_ROLE = "tracking"


@dataclass(frozen=True, slots=True)
class RoverDriveCommand:
    """A bounded physical command, not a pose teleport or scripted collapse."""

    linear_velocity_mps: float = 0.25
    angular_velocity_radps: float = 0.0
    control_steps: int = 90

    def __post_init__(self) -> None:
        _bounded_number(self.linear_velocity_mps, "linear_velocity_mps", -0.8, 0.8)
        _bounded_number(self.angular_velocity_radps, "angular_velocity_radps", -1.5, 1.5)
        if isinstance(self.control_steps, bool) or not isinstance(self.control_steps, int) or not 1 <= self.control_steps <= 1800:
            raise ValueError("control_steps must be an integer between 1 and 1800")


@dataclass(frozen=True, slots=True)
class MineRoverExperiment:
    """Portable intent for the container-side mine runner."""

    run_id: str
    seed: int
    drive: RoverDriveCommand
    world_id: str = MINE_WORLD_ID
    camera_resolution: tuple[int, int] = (180, 320)  # height, width
    camera_tick_rate_hz: float = 10.0

    def __post_init__(self) -> None:
        if not self.run_id.strip() or any(part in self.run_id for part in ("/", "\\", "..")):
            raise ValueError("run_id must be a simple non-empty identifier")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")  # noqa: TRY004 - one public validation error type
        if not self.world_id.strip() or any(part in self.world_id for part in ("/", "\\", "..")):
            raise ValueError("world_id must be a simple non-empty identifier")
        height, width = self.camera_resolution
        if not all(isinstance(value, int) and 32 <= value <= 1920 for value in (height, width)):
            raise ValueError("camera_resolution must be height/width integers between 32 and 1920")
        _bounded_number(self.camera_tick_rate_hz, "camera_tick_rate_hz", 1.0, 60.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_wheel_dofs(dof_names: list[str]) -> tuple[list[str], list[str]]:
    """Find left/right wheel DOFs without guessing fixed Nova Carter names."""
    left = sorted(name for name in dof_names if "wheel" in name.casefold() and "left" in name.casefold())
    right = sorted(name for name in dof_names if "wheel" in name.casefold() and "right" in name.casefold())
    if not left or not right:
        raise ValueError("could not identify both left and right wheel DOFs from the loaded rover articulation")
    return left, right


def write_reactor_seed_manifest(
    run_directory: str | Path,
    experiment: MineRoverExperiment,
    *,
    seed_image: str | Path,
    source_stage: str | Path,
    session_layer: str | Path,
    rover_pose_before: list[float],
    rover_pose_after: list[float],
    camera_prim: str = TRACKING_CAMERA_PATH,
    camera_role: str = REACTOR_SEED_CAMERA_ROLE,
    visual_quality: dict[str, Any] | None = None,
    reactor_prompt: str | None = None,
) -> Path:
    """Record a real Isaac camera image as a bounded Reactor conditioning input.

    This deliberately creates no Reactor result. A Reactor transport must later
    upload the declared image and persist its own generated visual evidence.
    """
    run_directory = Path(run_directory)
    seed_image = Path(seed_image)
    if not seed_image.is_file():
        raise FileNotFoundError(f"Isaac seed image does not exist: {seed_image}")
    manifest = {
        "schema_version": "v1",
        "world_id": experiment.world_id,
        "source_authority": "physics_grounded_isaac_rgb",
        "usage": "Seed one bounded Reactor visual-world episode; it is not physical ground truth.",
        "reactor_model": "reactor/lingbot-world-2",
        "experiment": experiment.to_dict(),
        "camera_prim": camera_prim,
        "camera_role": camera_role,
        "visual_quality": visual_quality or assess_visual_frame(seed_image),
        "seed_image_path": str(seed_image),
        "source_stage_path": str(source_stage),
        "derived_session_layer_path": str(session_layer),
        "rover_pose_before": rover_pose_before,
        "rover_pose_after": rover_pose_after,
        "prompt": reactor_prompt
        or (
            "Tracking-camera RGB view of a Nova Carter inspection rover approaching a roof "
            "support in a dim underground mine drift. Keep the rover, support, beam, loose "
            "rock, and debris spatially consistent while continuing the future scene."
        ),
    }
    output = run_directory / "reactor_seed.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def assess_visual_frame(image_path: str | Path) -> dict[str, Any]:
    """Measure whether a rendered frame has enough variation to be evidence.

    This is deliberately a conservative image-content gate, not a claim that
    the correct semantic objects are visible. Simulator-side semantic boxes
    provide that separate check.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = pixels.mean(axis=2)
    horizontal = np.abs(np.diff(luminance, axis=1))
    vertical = np.abs(np.diff(luminance, axis=0))
    channel_std = [float(value) for value in pixels.std(axis=(0, 1)).tolist()]
    mean_luminance = float(luminance.mean())
    dark_pixel_fraction = float((luminance < 10.0).mean())
    dynamic_range = float(np.percentile(luminance, 99) - np.percentile(luminance, 1))
    mean_gradient = float(
        (horizontal.mean() if horizontal.size else 0.0)
        + (vertical.mean() if vertical.size else 0.0)
    )
    passed = (
        max(channel_std) >= 8.0
        and mean_luminance >= 12.0
        and dark_pixel_fraction <= 0.65
        and dynamic_range >= 20.0
        and mean_gradient >= 0.5
    )
    return {
        "passed": passed,
        "channel_std": channel_std,
        "mean_luminance": mean_luminance,
        "dark_pixel_fraction_below_10": dark_pixel_fraction,
        "luminance_dynamic_range_p01_p99": dynamic_range,
        "mean_gradient": mean_gradient,
        "criterion": (
            "max channel std >= 8, mean luminance >= 12, dark pixel fraction below 10 <= 0.65, "
            "luminance p99-p01 >= 20, mean gradient >= 0.5"
        ),
    }


def _bounded_number(value: float, name: str, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
