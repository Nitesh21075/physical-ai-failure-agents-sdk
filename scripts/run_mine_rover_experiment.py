"""Run one non-destructive Nova Carter + RTX-camera mine experiment.

Run this only with Isaac Sim's ``python.sh`` in the 6.0.1 container. The input
mine USD is never saved. Sensor schema edits are exported to a per-run session
layer and all RGB/Reactor handoff artifacts live below ``runs/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harness.mine_world import (
    EGO_CAMERA_PATH,
    REACTOR_SEED_CAMERA_ROLE,
    ROVER_CAMERA_PATH,
    ROVER_PRIM_PATH,
    TRACKING_CAMERA_PATH,
    WITNESS_CAMERA_PATH,
    MineRoverExperiment,
    RoverDriveCommand,
    assess_visual_frame,
    select_wheel_dofs,
    write_reactor_seed_manifest,
)

DEFAULT_STAGE = PROJECT_ROOT / "assets" / "worlds" / "mine_v1" / "mine_world.usda"
NOVA_CARTER_ASSET = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-velocity-mps", type=float, default=0.25)
    parser.add_argument("--angular-velocity-radps", type=float, default=0.0)
    parser.add_argument("--control-steps", type=int, default=90)
    parser.add_argument("--wheel-radius-m", type=float, default=0.125)
    parser.add_argument("--wheel-base-m", type=float, default=0.55)
    parser.add_argument("--camera-height", type=int, default=180)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-tick-rate-hz", type=float, default=10.0)
    parser.add_argument("--capture-every", type=int, default=15)
    parser.add_argument("--failure-zone", choices=("roof_support",), default="roof_support")
    parser.add_argument(
        "--disable-camera",
        action="store_true",
        help="Physics/articulation diagnostic; no Reactor seed is produced.",
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_derived_stage(source_stage: Path, run_directory: Path) -> Path:
    """Create the run-owned entry layer without changing the authored world.

    Selecting Nova Carter's ``Config=Full_Merged`` before composition is
    important: choosing it after opening the authored scene makes Kit compose
    the costly base configuration first.  The stronger derived layer selects
    NVIDIA's documented physics-capable merged representation from the start.
    """
    source_asset = str(source_stage.resolve()).replace("@", r"\@")
    derived_stage = run_directory / "mine_rover_derived_entry.usda"
    derived_stage.write_text(
        "\n".join(
            (
                "#usda 1.0",
                "(",
                f"    subLayers = [ @{source_asset}@ ]",
                '    defaultPrim = "World"',
                ")",
                "",
                'over "World"',
                "{",
                '    over "Robot" (',
                f"        prepend references = @{NOVA_CARTER_ASSET}@",
                "        variants = {",
                '            string robot_model = "nvidia_nova_carter"',
                '            string Configuration = "Full_Merged"',
                '            string Physics = "physx"',
                "        }",
                "    )",
                "    {",
                "    }",
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return derived_stage


def _wait_for_stage(app: object, context: object) -> object:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        app.update()
        if not context.get_stage_loading_status()[2]:
            stage = context.get_stage()
            if stage is not None:
                return stage
    raise RuntimeError("mine stage did not finish loading within 300 seconds")


def _as_pose(rover: object) -> list[float]:
    positions, _ = rover.get_world_poses()
    return [float(value) for value in positions.numpy()[0].tolist()]


def _world_pose(prim: object) -> tuple[object, object]:
    positions, orientations = prim.get_world_poses()
    return positions.numpy()[0].copy(), orientations.numpy()[0].copy()


def _rotate_vector(quaternion_wxyz: object, vector_xyz: object) -> object:
    """Rotate one XYZ vector by a WXYZ quaternion without simulator helpers."""
    import numpy as np

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    vector = np.asarray(vector_xyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    scalar, axis = quaternion[0], quaternion[1:]
    return (
        2.0 * np.dot(axis, vector) * axis
        + (scalar * scalar - np.dot(axis, axis)) * vector
        + 2.0 * scalar * np.cross(axis, vector)
    )


def _matrix_to_quaternion_wxyz(rotation: object) -> object:
    import numpy as np

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        quaternion = np.array(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = (1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) ** 0.5 * 2.0
            quaternion = np.array(
                [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale]
            )
        elif index == 1:
            scale = (1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) ** 0.5 * 2.0
            quaternion = np.array(
                [(matrix[0, 2] - matrix[2, 0]) / scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale]
            )
        else:
            scale = (1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) ** 0.5 * 2.0
            quaternion = np.array(
                [(matrix[1, 0] - matrix[0, 1]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
            )
    return quaternion / np.linalg.norm(quaternion)


def _look_at_pose(eye: object, target: object, up_hint: object = (0.0, 0.0, 1.0)) -> tuple[object, object]:
    """Return a camera pose using Isaac's +X right, +Y up, -Z forward convention."""
    import numpy as np

    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up_hint, dtype=np.float64))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    orientation = _matrix_to_quaternion_wxyz(np.column_stack((right, up, -forward)))
    return eye.astype(np.float32), orientation.astype(np.float32)


def _semantic_box_summary(data: object, info: dict[str, object]) -> list[dict[str, object]]:
    """Convert an RTX bounding-box result into compact JSON-safe evidence."""
    if data is None:
        return []
    labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    boxes: list[dict[str, object]] = []
    for box in data:
        semantic_id = int(box["semanticId"])
        label = labels.get(str(semantic_id), labels.get(semantic_id, "unknown"))
        boxes.append(
            {
                "semantic_id": semantic_id,
                "label": label,
                "xyxy": [int(box[name]) for name in ("x_min", "y_min", "x_max", "y_max")],
            }
        )
    return boxes


def _body_state(body: object) -> dict[str, list[float]]:
    positions, orientations = body.get_world_poses()
    linear, angular = body.get_velocities()
    return {
        "position_xyz_m": [float(value) for value in positions.numpy()[0].tolist()],
        "orientation_quaternion_wxyz": [float(value) for value in orientations.numpy()[0].tolist()],
        "linear_velocity_mps": [float(value) for value in linear.numpy()[0].tolist()],
        "angular_velocity_radps": [float(value) for value in angular.numpy()[0].tolist()],
    }


def _displacement(before: dict[str, list[float]], after: dict[str, list[float]]) -> float:
    return (
        sum(
            (end - start) ** 2
            for start, end in zip(before["position_xyz_m"], after["position_xyz_m"], strict=True)
        )
        ** 0.5
    )


def _write_standard_artifacts(
    run_directory: Path,
    experiment: MineRoverExperiment,
    *,
    before: dict[str, object],
    after: dict[str, object],
    support_displacement_m: float,
    beam_displacement_m: float,
    structural_collapse: bool,
    camera_frames: dict[str, list[Path]],
) -> None:
    scenario = {
        "environment": "mine_v1",
        "scenario_id": experiment.run_id,
        "task": "mine_roof_support_contact",
        "seed": experiment.seed,
        "parameters": {
            "failure_zone": "roof_support",
            "rover_linear_velocity_mps": experiment.drive.linear_velocity_mps,
            "control_steps": experiment.drive.control_steps,
        },
        "hazards": {"structural_collapse": True},
    }
    preferred_frames = camera_frames.get(REACTOR_SEED_CAMERA_ROLE, [])
    evidence_refs = [str(path) for path in preferred_frames]
    sensor_streams = {
        role: [str(path) for path in paths] for role, paths in camera_frames.items()
    }
    result = {
        "task_success": True,
        "environmental_failure": structural_collapse,
        "failure_type": "structural_collapse" if structural_collapse else None,
        "severity": "high" if structural_collapse else "none",
        "terminal": structural_collapse,
        "evidence_refs": evidence_refs,
        "metrics": {
            "support_displacement_m": support_displacement_m,
            "beam_displacement_m": beam_displacement_m,
            "beam_vertical_drop_m": before["beam"]["position_xyz_m"][2]
            - after["beam"]["position_xyz_m"][2],
            "collapse_criterion": "beam vertical drop > 0.8 m",
            "contact_evidence": "support displacement after wheel-actuated rover approach",
        },
    }
    metadata = {
        "run_id": experiment.run_id,
        "backend": "isaac_sim",
        "world_id": "mine_v1",
        "robot": "NVIDIA Nova Carter",
        "actuation": "wheel velocity targets",
    }
    trajectory = [
        {
            "record_type": "initial_observation",
            "observation": {
                "simulation_time": 0.0,
                "state": before,
                "sensor_refs": evidence_refs[:1],
                "sensor_streams": {role: refs[:1] for role, refs in sensor_streams.items()},
            },
        },
        {
            "record_type": "step",
            "action": {
                "name": "set_wheel_velocity_targets",
                "parameters": scenario["parameters"],
            },
            "result": {
                "simulation_time": experiment.drive.control_steps / 60.0,
                "observation": {
                    "simulation_time": experiment.drive.control_steps / 60.0,
                    "state": after,
                    "sensor_refs": evidence_refs[-1:],
                    "sensor_streams": {role: refs[-1:] for role, refs in sensor_streams.items()},
                },
                "events": (
                    [{"event_type": "structural_collapse", "severity": "high"}]
                    if structural_collapse
                    else []
                ),
                "done": True,
            },
        },
    ]
    (run_directory / "scenario.json").write_text(
        json.dumps(scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_directory / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_directory / "trajectory.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in trajectory), encoding="utf-8"
    )


def _rover_variant_snapshot(robot_prim: object) -> dict[str, dict[str, dict[str, object]]]:
    """Inspect only the shallow rover hierarchy; never traverse the full mine."""
    snapshot: dict[str, dict[str, dict[str, object]]] = {}
    prims = [robot_prim, *robot_prim.GetChildren()]
    for child in robot_prim.GetChildren():
        prims.extend(child.GetChildren())
    for prim in prims:
        variants = prim.GetVariantSets()
        names = variants.GetNames()
        if names:
            snapshot[str(prim.GetPath())] = {
                name: {
                    "selected": variants.GetVariantSet(name).GetVariantSelection(),
                    "available": list(variants.GetVariantSet(name).GetVariantNames()),
                }
                for name in names
            }
    return snapshot


def _ensure_articulation_root(robot_prim: object) -> str:
    """Find the Nova PhysX articulation and add the USD root API in-session.

    The NVIDIA asset exposes its native PhysX articulation but does not author
    ``UsdPhysics.ArticulationRootAPI``. Isaac Sim 6's experimental
    ``Articulation`` wrapper requires that USD API, so the derived run session
    supplies it without modifying the source asset.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    usd_roots = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(robot_prim)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(usd_roots) == 1:
        return usd_roots[0]
    if len(usd_roots) > 1:
        raise RuntimeError(
            f"expected one Nova Carter USD articulation root below {ROVER_PRIM_PATH}, found: {usd_roots}"
        )
    physx_roots = [
        prim for prim in Usd.PrimRange(robot_prim) if prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
    ]
    if len(physx_roots) != 1:
        paths = [str(prim.GetPath()) for prim in physx_roots]
        raise RuntimeError(
            f"expected one Nova Carter PhysX articulation below {ROVER_PRIM_PATH}, found: {paths}"
        )
    root = UsdPhysics.ArticulationRootAPI.Apply(physx_roots[0])
    _require(
        root, f"failed to apply session-only ArticulationRootAPI to {physx_roots[0].GetPath()}"
    )
    return str(physx_roots[0].GetPath())


def _save_rgb(data: object, output: Path) -> None:
    from PIL import Image

    pixels = data.numpy()
    _require(pixels.ndim == 3 and pixels.shape[2] >= 3, "RTX camera did not return an RGB image")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels[:, :, :3].astype("uint8"), mode="RGB").save(output)


def main() -> int:
    args = _arguments()
    _require(args.stage.is_file(), f"mine stage does not exist: {args.stage}")
    _require(args.wheel_radius_m > 0 and args.wheel_base_m > 0, "wheel dimensions must be positive")
    _require(args.capture_every > 0, "capture-every must be positive")
    run_id = args.run_id or str(uuid4())
    experiment = MineRoverExperiment(
        run_id=run_id,
        seed=args.seed,
        drive=RoverDriveCommand(
            args.linear_velocity_mps, args.angular_velocity_radps, args.control_steps
        ),
        camera_resolution=(args.camera_height, args.camera_width),
        camera_tick_rate_hz=args.camera_tick_rate_hz,
    )
    run_directory = args.runs_dir / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    derived_stage = _write_derived_stage(args.stage, run_directory)
    print(f"mine rover: prepared derived run directory {run_directory}", flush=True)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "None" if args.disable_camera else "RayTracedLighting",
            "/rtx/materialDb/syncLoads": False,
            "/rtx/hydra/materialSyncLoads": False,
            "/omni/kit/plugin/syncUsdLoads": False,
        }
    )
    try:
        import carb
        import numpy as np
        import omni.usd
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        # SimulationApp's launch configuration intentionally accepts a small
        # whitelist.  Set these loader knobs explicitly as well, before USD
        # opens the remote Nova Carter reference.  This lets Kit make progress
        # asynchronously instead of blocking its first physics update on every
        # material and plugin load.
        settings = carb.settings.get_settings()
        settings.set_bool("/rtx/materialDb/syncLoads", False)
        settings.set_bool("/rtx/hydra/materialSyncLoads", False)
        settings.set_bool("/omni/kit/plugin/syncUsdLoads", False)
        context = omni.usd.get_context()
        print(
            f"mine rover: opening derived entry layer {derived_stage} over immutable source {args.stage.resolve()}",
            flush=True,
        )
        _require(
            context.open_stage(str(derived_stage)), f"failed to open derived stage: {derived_stage}"
        )
        stage = _wait_for_stage(app, context)
        print("mine rover: stage loaded", flush=True)
        robot_prim = stage.GetPrimAtPath(ROVER_PRIM_PATH)
        _require(robot_prim.IsValid(), f"missing rover prim: {ROVER_PRIM_PATH}")
        authored_rover_camera = stage.GetPrimAtPath(ROVER_CAMERA_PATH)

        # The original stage is input-only. The RTX sensor schema is authored
        # into the anonymous session layer and exported separately per run.
        session_layer = stage.GetSessionLayer()
        stage.SetEditTarget(session_layer)
        selected_variants = [
            f"{ROVER_PRIM_PATH}:Configuration=Full_Merged",
            f"{ROVER_PRIM_PATH}:Physics=physx",
        ]
        resolved_variants = _rover_variant_snapshot(robot_prim)
        print(
            "mine rover: derived entry requested physics-capable rover variants "
            + ", ".join(selected_variants)
            + f"; resolved selections={resolved_variants}",
            flush=True,
        )
        articulation_root = _ensure_articulation_root(robot_prim)
        print(f"mine rover: session-only articulation root={articulation_root}", flush=True)
        if not args.disable_camera:
            import isaacsim.core.experimental.utils.semantics as semantics_utils
            from pxr import Gf, Usd, UsdGeom, UsdLux

            for prim in Usd.PrimRange(robot_prim):
                if prim.IsA(UsdGeom.Gprim):
                    semantics_utils.add_labels(prim, labels="nova_carter")
            semantics_utils.add_labels(articulation_root, labels="nova_carter")
            for prim_path, label in (
                ("/World/FailureZones/RoofSupportZone/SupportPrimary/CollisionAndVisual", "roof_support"),
                ("/World/FailureZones/RoofSupportZone/BeamPrimary/CollisionAndVisual", "roof_beam"),
            ):
                semantics_utils.add_labels(prim_path, labels=label)
            fill_light = UsdLux.SphereLight.Define(stage, "/World/Sensors/RoverEvidenceFill")
            fill_light.CreateIntensityAttr(4500.0)
            fill_light.CreateRadiusAttr(0.18)
            fill_light.CreateColorAttr(Gf.Vec3f(0.62, 0.72, 1.0))
            UsdGeom.Xformable(fill_light).AddTranslateOp().Set(Gf.Vec3d(24.0, -2.5, 2.5))
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(10):
            app.update()
        print("mine rover: existing PhysX timeline initialized", flush=True)
        rover = Articulation(articulation_root)
        left_dofs, right_dofs = select_wheel_dofs(rover.dof_names)
        print(
            f"mine rover: discovered articulation={articulation_root} "
            f"left={left_dofs} right={right_dofs}",
            flush=True,
        )
        wheel_indices = rover.get_dof_indices(left_dofs + right_dofs)
        left_velocity = (
            experiment.drive.linear_velocity_mps
            - experiment.drive.angular_velocity_radps * args.wheel_base_m / 2
        ) / args.wheel_radius_m
        right_velocity = (
            experiment.drive.linear_velocity_mps
            + experiment.drive.angular_velocity_radps * args.wheel_base_m / 2
        ) / args.wheel_radius_m
        wheel_targets = np.array(
            [[left_velocity] * len(left_dofs) + [right_velocity] * len(right_dofs)],
            dtype=np.float32,
        )
        manifest = json.loads((args.stage.parent / "manifest.json").read_text(encoding="utf-8"))
        start_pose = manifest["failure_zones"][args.failure_zone]["experiment_start_pose"]
        rover.set_world_poses(
            positions=np.array([start_pose["position_xyz_m"]], dtype=np.float32),
            orientations=np.array([start_pose["orientation_quaternion_wxyz"]], dtype=np.float32),
        )
        zero_targets = np.zeros_like(wheel_targets)
        for _ in range(30):
            rover.set_dof_velocity_targets(zero_targets, dof_indices=wheel_indices)
            app.update()
        beam = RigidPrim("/World/FailureZones/RoofSupportZone/BeamPrimary")
        support = RigidPrim("/World/FailureZones/RoofSupportZone/SupportPrimary")
        camera_sensors: dict[str, object] = {}
        camera_prims: dict[str, object] = {}
        camera_pose_timeline: list[dict[str, object]] = []
        semantic_visibility: dict[str, list[dict[str, object]]] = {
            role: [] for role in ("ego", "tracking", "witness")
        }
        tracking_eye = None
        tracking_target = None
        if not args.disable_camera:
            rover_position, rover_orientation = _world_pose(rover)
            ego_eye = rover_position + _rotate_vector(rover_orientation, (0.0, 0.0, 1.65))
            ego_target = ego_eye + _rotate_vector(rover_orientation, (5.0, 0.0, -0.25))
            tracking_eye = rover_position + _rotate_vector(rover_orientation, (-2.1, 0.0, 3.5))
            tracking_target = rover_position.copy()
            tracking_target[2] = 1.15
            initial_poses = {
                "ego": (EGO_CAMERA_PATH, *_look_at_pose(ego_eye, ego_target)),
                "tracking": (
                    TRACKING_CAMERA_PATH,
                    *_look_at_pose(tracking_eye, tracking_target),
                ),
                "witness": (
                    WITNESS_CAMERA_PATH,
                    *_look_at_pose((27.0, -3.3, 3.0), (24.0, -0.3, 1.4)),
                ),
            }
            for role, (path, position, orientation) in initial_poses.items():
                existing_camera = stage.GetPrimAtPath(path)
                if existing_camera.IsValid() and "OmniSensorAPI" not in existing_camera.GetAppliedSchemas():
                    existing_camera.ApplyAPI("OmniSensorAPI")
                camera_prim = RtxCamera(
                    path,
                    tick_rate=experiment.camera_tick_rate_hz,
                    positions=np.array([position], dtype=np.float32),
                    orientations=np.array([orientation], dtype=np.float32),
                    reset_xform_op_properties=True,
                )
                focal_length = {"ego": 18.0, "tracking": 12.0, "witness": 24.0}[role]
                UsdGeom.Camera(stage.GetPrimAtPath(path)).GetFocalLengthAttr().Set(focal_length)
                camera_prims[role] = camera_prim
                camera_sensors[role] = CameraSensor(
                    camera_prim,
                    resolution=experiment.camera_resolution,
                    annotators=["rgb", "bounding_box_2d_tight"],
                )
            print(
                "mine rover: configured ego, smoothed tracking, and fixed witness RTX cameras; "
                f"authored rover camera valid={authored_rover_camera.IsValid()}",
                flush=True,
            )
            for _ in range(10):
                app.update()
        rover_pose_before = _as_pose(rover)
        before = {
            "rover": {"position_xyz_m": rover_pose_before},
            "support": _body_state(support),
            "beam": _body_state(beam),
        }
        camera_frames: dict[str, list[Path]] = {
            role: [] for role in ("ego", "tracking", "witness")
        }
        for step in range(experiment.drive.control_steps):
            rover.set_dof_velocity_targets(wheel_targets, dof_indices=wheel_indices)
            app.update()
            if camera_sensors:
                rover_position, rover_orientation = _world_pose(rover)
                ego_eye = rover_position + _rotate_vector(
                    rover_orientation, (0.0, 0.0, 1.65)
                )
                ego_target = ego_eye + _rotate_vector(
                    rover_orientation, (5.0, 0.0, -0.25)
                )
                desired_eye = rover_position + _rotate_vector(
                    rover_orientation, (-2.1, 0.0, 3.5)
                )
                desired_target = rover_position.copy()
                desired_target[2] = 1.15
                tracking_eye = 0.88 * tracking_eye + 0.12 * desired_eye
                tracking_target = 0.88 * tracking_target + 0.12 * desired_target
                moving_poses = {
                    "ego": _look_at_pose(ego_eye, ego_target),
                    "tracking": _look_at_pose(tracking_eye, tracking_target),
                }
                for role, (position, orientation) in moving_poses.items():
                    camera_prims[role].set_world_poses(
                        positions=np.array([position], dtype=np.float32),
                        orientations=np.array([orientation], dtype=np.float32),
                    )
            if camera_sensors and (
                step % args.capture_every == 0 or step == experiment.drive.control_steps - 1
            ):
                pose_record: dict[str, object] = {
                    "control_step": step,
                    "rover_position_xyz_m": [
                        float(value) for value in rover_position.tolist()
                    ],
                    "cameras": {},
                }
                for role, sensor in camera_sensors.items():
                    frame, _ = sensor.get_data("rgb")
                    if frame is None:
                        continue
                    frame_path = (
                        run_directory / "camera" / role /
                        f"rgb_{len(camera_frames[role]):06d}.png"
                    )
                    _save_rgb(frame, frame_path)
                    camera_frames[role].append(frame_path)
                    boxes, box_info = sensor.get_data("bounding_box_2d_tight")
                    semantic_visibility[role].append(
                        {
                            "control_step": step,
                            "boxes": _semantic_box_summary(boxes, box_info),
                        }
                    )
                    position, orientation = _world_pose(camera_prims[role])
                    pose_record["cameras"][role] = {
                        "prim": camera_prims[role].paths[0],
                        "position_xyz_m": [float(value) for value in position.tolist()],
                        "orientation_quaternion_wxyz": [
                            float(value) for value in orientation.tolist()
                        ],
                    }
                camera_pose_timeline.append(pose_record)
        rover_pose_after = _as_pose(rover)
        after = {
            "rover": {"position_xyz_m": rover_pose_after},
            "support": _body_state(support),
            "beam": _body_state(beam),
        }
        support_displacement_m = _displacement(before["support"], after["support"])
        beam_displacement_m = _displacement(before["beam"], after["beam"])
        beam_vertical_drop_m = (
            before["beam"]["position_xyz_m"][2] - after["beam"]["position_xyz_m"][2]
        )
        structural_collapse = beam_vertical_drop_m > 0.8
        timeline.stop()
        session_path = run_directory / "mine_rover_session.usda"
        session_layer.Export(str(session_path))
        seed_manifest = None
        visual_quality: dict[str, list[dict[str, object]]] = {
            role: [assess_visual_frame(path) for path in paths]
            for role, paths in camera_frames.items()
        }
        semantic_labels_by_role = {
            role: sorted(
                {
                    str(box["label"])
                    for sample in samples
                    for box in sample["boxes"]
                }
            )
            for role, samples in semantic_visibility.items()
        }
        tracking_content_passed = bool(camera_frames["tracking"]) and all(
            item["passed"] for item in visual_quality["tracking"]
        )
        tracking_label_text = " ".join(semantic_labels_by_role["tracking"]).casefold()
        witness_label_text = " ".join(semantic_labels_by_role["witness"]).casefold()
        tracking_pose_samples = [
            sample["cameras"]["tracking"]
            for sample in camera_pose_timeline
            if "tracking" in sample["cameras"]
        ]
        rover_geometric_visibility = bool(tracking_pose_samples) and all(
            1.0
            <= sum(
                (camera_position - rover_position) ** 2
                for camera_position, rover_position in zip(
                    sample["position_xyz_m"],
                    camera_pose_timeline[index]["rover_position_xyz_m"],
                    strict=True,
                )
            )
            ** 0.5
            <= 6.0
            for index, sample in enumerate(tracking_pose_samples)
        )
        tracking_semantics_passed = (
            "roof_support" in tracking_label_text and "roof_beam" in witness_label_text
        )
        visual_evidence_gate = {
            "passed": (
                tracking_content_passed
                and tracking_semantics_passed
                and rover_geometric_visibility
            ),
            "preferred_role": REACTOR_SEED_CAMERA_ROLE,
            "content_passed": tracking_content_passed,
            "semantic_visibility_passed": tracking_semantics_passed,
            "required_semantic_labels": ["roof_support", "roof_beam"],
            "observed_labels_by_role": semantic_labels_by_role,
            "rover_geometric_visibility_passed": rover_geometric_visibility,
            "rover_visibility_method": (
                "tracking camera is deterministically aimed at the measured articulation root; "
                "camera-to-rover distance must remain 1-6 m. NVIDIA's instanced Nova visuals do "
                "not emit the authored semantic label in Isaac Sim 6.0.1 bounding-box output."
            ),
        }
        if camera_sensors:
            _require(camera_frames["tracking"], "tracking camera did not produce an RGB frame")
            seed_manifest = write_reactor_seed_manifest(
                run_directory,
                experiment,
                seed_image=camera_frames["tracking"][0],
                source_stage=args.stage.resolve(),
                session_layer=session_path,
                rover_pose_before=rover_pose_before,
                rover_pose_after=rover_pose_after,
                camera_prim=TRACKING_CAMERA_PATH,
                camera_role=REACTOR_SEED_CAMERA_ROLE,
                visual_quality={
                    **visual_quality["tracking"][0],
                    "semantic_visibility": semantic_visibility["tracking"][0],
                    "run_visual_evidence_gate": visual_evidence_gate,
                },
            )
        _write_standard_artifacts(
            run_directory,
            experiment,
            before=before,
            after=after,
            support_displacement_m=support_displacement_m,
            beam_displacement_m=beam_displacement_m,
            structural_collapse=structural_collapse,
            camera_frames=camera_frames,
        )
        camera_streams = {
            role: {
                "camera_prim": {
                    "ego": EGO_CAMERA_PATH,
                    "tracking": TRACKING_CAMERA_PATH,
                    "witness": WITNESS_CAMERA_PATH,
                }[role],
                "camera_role": role,
                "resolution_height_width": list(experiment.camera_resolution),
                "frames": [str(path) for path in paths],
                "visual_quality": visual_quality[role],
                "semantic_visibility": semantic_visibility[role],
            }
            for role, paths in camera_frames.items()
        }
        camera_pose_path = run_directory / "camera" / "camera_poses.jsonl"
        camera_pose_path.parent.mkdir(parents=True, exist_ok=True)
        camera_pose_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in camera_pose_timeline),
            encoding="utf-8",
        )
        summary = {
            "run_id": run_id,
            "backend": "isaac_sim",
            "world_id": "mine_v1",
            "source_stage": str(args.stage.resolve()),
            "derived_entry_stage": str(derived_stage),
            "session_layer": str(session_path),
            "camera_prim": TRACKING_CAMERA_PATH,
            "camera_status": "disabled for physics diagnostic"
            if not camera_sensors
            else "captured synchronized ego, tracking, and witness RTX RGB streams",
            "camera_streams": camera_streams,
            "camera_pose_timeline": str(camera_pose_path),
            "reactor_seed_camera_role": REACTOR_SEED_CAMERA_ROLE,
            "visual_evidence_gate": visual_evidence_gate,
            "articulation_root": articulation_root,
            "session_only_robot_variants": selected_variants,
            "resolved_rover_variants": resolved_variants,
            "camera_frames": [str(path) for path in camera_frames[REACTOR_SEED_CAMERA_ROLE]],
            "camera_frame_count_by_role": {
                role: len(paths) for role, paths in camera_frames.items()
            },
            "left_wheel_dofs": left_dofs,
            "right_wheel_dofs": right_dofs,
            "wheel_velocity_targets_radps": {"left": left_velocity, "right": right_velocity},
            "rover_pose_before": rover_pose_before,
            "rover_pose_after": rover_pose_after,
            "physical_state_before": before,
            "physical_state_after": after,
            "support_displacement_m": support_displacement_m,
            "beam_displacement_m": beam_displacement_m,
            "beam_vertical_drop_m": beam_vertical_drop_m,
            "structural_collapse": structural_collapse,
            "collapse_criterion": "beam vertical drop > 0.8 m",
            "reactor_seed_manifest": str(seed_manifest),
            "source_stage_unchanged": True,
        }
        (run_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        # A disposable process avoids the known 6.0.1 teardown instability
        # while preserving all synchronously-written per-run artifacts.
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:  # noqa: BLE001 - report native simulator failures at the boundary
        print(f"mine rover experiment failed: {error}", file=sys.stderr)
        exit_code = 1
    os._exit(exit_code)
