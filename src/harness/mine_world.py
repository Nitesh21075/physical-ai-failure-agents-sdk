"""Typed, backend-independent contract for one mine-rover visual experiment.

The original composed USD stage is always an input. Simulator code writes any
sensor schema edits or experiment metadata to a per-run session layer instead.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MINE_WORLD_ID = "mine_v1"
ROVER_PRIM_PATH = "/World/Robot"
ROVER_CAMERA_PATH = "/World/Robot/VLACamera"


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
    camera_resolution: tuple[int, int] = (180, 320)  # height, width
    camera_tick_rate_hz: float = 10.0

    def __post_init__(self) -> None:
        if not self.run_id.strip() or any(part in self.run_id for part in ("/", "\\", "..")):
            raise ValueError("run_id must be a simple non-empty identifier")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")  # noqa: TRY004 - one public validation error type
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
        "world_id": MINE_WORLD_ID,
        "source_authority": "physics_grounded_isaac_rgb",
        "usage": "Seed one bounded Reactor visual-world episode; it is not physical ground truth.",
        "reactor_model": "reactor/lingbot-world-2",
        "experiment": experiment.to_dict(),
        "camera_prim": ROVER_CAMERA_PATH,
        "seed_image_path": str(seed_image),
        "source_stage_path": str(source_stage),
        "derived_session_layer_path": str(session_layer),
        "rover_pose_before": rover_pose_before,
        "rover_pose_after": rover_pose_after,
        "prompt": "First-person RGB view from an inspection rover navigating a dim underground mine drift with roof supports, loose rock, and debris. Continue the visual scene consistently from this image.",
    }
    output = run_directory / "reactor_seed.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _bounded_number(value: float, name: str, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
