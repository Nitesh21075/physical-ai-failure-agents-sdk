#!/usr/bin/env python3
"""Validate the persistent mine_v1 USD asset inside Isaac Sim 6.0.1.

This is intentionally a validator and renderer, not a scene generator. Run it
with /isaac-sim/python.sh in the AWS container.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLD_DIRECTORY = PROJECT_ROOT / "assets" / "worlds" / "mine_v1"
ENTRY_STAGE = WORLD_DIRECTORY / "mine_world.usda"

REQUIRED_PRIMS = (
    "/World", "/World/Environment", "/World/Environment/MineShell",
    "/World/Environment/Floor", "/World/Environment/Walls",
    "/World/Environment/Ceiling", "/World/Environment/StaticProps",
    "/World/FailureZones", "/World/FailureZones/Zone_RoofSupport",
    "/World/FailureZones/Zone_RoofSupport/BeamPrimary",
    "/World/FailureZones/Zone_RoofSupport/SupportPrimary",
    "/World/FailureZones/Zone_RoofSupport/InteractionTarget",
    "/World/FailureZones/Zone_Rockfall/Rock01",
    "/World/FailureZones/Zone_Rockfall/Rock02",
    "/World/FailureZones/Zone_Rockfall/RetainingObject",
    "/World/FailureZones/Zone_Debris/Debris01",
    "/World/FailureZones/Zone_Debris/Debris02",
    "/World/FailureZones/Zone_Debris/InteractionTarget", "/World/Robot",
    "/World/Sensors/ResearchCamera", "/World/Sensors/OverviewCamera", "/World/Lighting",
    "/World/Environment/MineExtension",
    "/World/FailureZones/RoofSupportZone",
    "/World/FailureZones/RoofSupportZone/BeamPrimary",
    "/World/FailureZones/RoofSupportZone/SupportPrimary",
    "/World/FailureZones/RoofSupportZone/SupportSecondary",
    "/World/FailureZones/RoofSupportZone/InteractionTarget",
    "/World/FailureZones/RoofSupportZone/InspectionMarker",
    "/World/FailureZones/RockfallZone",
    "/World/FailureZones/RockfallZone/Rock01",
    "/World/FailureZones/RockfallZone/Rock02",
    "/World/FailureZones/RockfallZone/RetainingBlock",
    "/World/FailureZones/RockfallZone/InteractionTarget",
    "/World/FailureZones/RockfallZone/InspectionMarker",
    "/World/FailureZones/DebrisCascadeZone/Debris01",
    "/World/FailureZones/DebrisCascadeZone/Debris02",
    "/World/FailureZones/DebrisCascadeZone/Debris03",
    "/World/FailureZones/DebrisCascadeZone/InteractionTarget",
    "/World/Sensors/RoofSupportCamera", "/World/Sensors/RockfallCamera",
    "/World/Sensors/DebrisCamera",
)
RIGID_BODIES = (
    "/World/FailureZones/Zone_RoofSupport/BeamPrimary",
    "/World/FailureZones/Zone_RoofSupport/SupportPrimary",
    "/World/FailureZones/Zone_Rockfall/Rock01",
    "/World/FailureZones/Zone_Rockfall/Rock02",
    "/World/FailureZones/Zone_Debris/Debris01",
    "/World/FailureZones/Zone_Debris/Debris02",
    "/World/FailureZones/RoofSupportZone/BeamPrimary",
    "/World/FailureZones/RoofSupportZone/SupportPrimary",
    "/World/FailureZones/RockfallZone/Rock01",
    "/World/FailureZones/RockfallZone/Rock02",
    "/World/FailureZones/RockfallZone/RetainingBlock",
    "/World/FailureZones/DebrisCascadeZone/Debris01",
    "/World/FailureZones/DebrisCascadeZone/Debris02",
    "/World/FailureZones/DebrisCascadeZone/Debris03",
)
COLLIDERS = (
    "/World/Environment/Floor",
    "/World/FailureZones/Zone_RoofSupport/BeamPrimary/CollisionAndVisual",
    "/World/FailureZones/Zone_RoofSupport/SupportPrimary/CollisionAndVisual",
    "/World/FailureZones/Zone_Rockfall/Rock01/CollisionAndVisual",
    "/World/FailureZones/Zone_Rockfall/Rock02/CollisionAndVisual",
    "/World/FailureZones/Zone_Debris/Debris01/CollisionAndVisual",
    "/World/FailureZones/Zone_Debris/Debris02/CollisionAndVisual",
    "/World/FailureZones/RoofSupportZone/BeamPrimary/CollisionAndVisual",
    "/World/FailureZones/RoofSupportZone/SupportPrimary/CollisionAndVisual",
    "/World/FailureZones/RockfallZone/Rock01/CollisionAndVisual",
    "/World/FailureZones/RockfallZone/Rock02/CollisionAndVisual",
    "/World/FailureZones/RockfallZone/RetainingBlock/CollisionAndVisual",
    "/World/FailureZones/DebrisCascadeZone/Debris01/CollisionAndVisual",
    "/World/FailureZones/DebrisCascadeZone/Debris02/CollisionAndVisual",
    "/World/FailureZones/DebrisCascadeZone/Debris03/CollisionAndVisual",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=ENTRY_STAGE)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "mine-world-validation",
        help="generated validation images; never modifies the persistent USD world",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="physics-only diagnostic for a host whose RTX shader pipeline is unavailable",
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_stage(stage: object) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    for path in REQUIRED_PRIMS:
        _require(stage.GetPrimAtPath(path).IsValid(), f"missing required prim: {path}")
    for path in (
        "/World/Sensors/ResearchCamera", "/World/Sensors/OverviewCamera",
        "/World/Sensors/RoofSupportCamera", "/World/Sensors/RockfallCamera",
        "/World/Sensors/DebrisCamera",
    ):
        _require(stage.GetPrimAtPath(path).IsA(UsdGeom.Camera), f"not a camera: {path}")
    for path in RIGID_BODIES:
        _require(UsdPhysics.RigidBodyAPI.Get(stage, path), f"rigid-body API missing: {path}")
    for path in COLLIDERS:
        _require(UsdPhysics.CollisionAPI.Get(stage, path), f"collision API missing: {path}")
    robot = stage.GetPrimAtPath("/World/Robot")
    robot_prims = list(Usd.PrimRange(robot, Usd.TraverseInstanceProxies()))
    robot_has_physics = any(
        prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        for prim in robot_prims
    )
    if not robot_has_physics:
        snapshot = [(str(prim.GetPath()), prim.GetTypeName(), list(prim.GetAppliedSchemas())) for prim in robot_prims]
        raise RuntimeError(f"rover reference did not resolve to collision/articulation prims: {snapshot}")
    _require(stage.GetPrimAtPath("/World/PhysicsScene").IsValid(), "missing PhysicsScene")


def _select_physics_rover(stage: object) -> None:
    """Select Nova Carter's physics-capable variants in the anonymous session.

    The persistent world deliberately leaves NVIDIA asset choices at their
    upstream defaults. Validation needs the same explicit, non-persistent
    variant selection as an experiment run before checking rigid bodies.
    """
    robot = stage.GetPrimAtPath("/World/Robot")
    variants = robot.GetVariantSets()
    # The authored proxy is the validator's deterministic local fallback.  The
    # dedicated experiment runner separately proves the external Nova Carter
    # articulation path; a network or upstream asset change must not prevent
    # validation of the mine's own PhysX support/beam causal chain.
    requested = {"robot_model": "proxy_fallback"}
    for name, selection in requested.items():
        variant = variants.GetVariantSet(name)
        available = list(variant.GetVariantNames())
        if available:
            _require(selection in available, f"Nova Carter variant {name}={selection} is unavailable: {available}")
            _require(variant.SetVariantSelection(selection), f"could not select {name}={selection}")
    print("validator rover: authored proxy_fallback (external Nova Carter is tested by the experiment runner)")


def _write_validation_entry(source_stage: Path, record_dir: Path) -> Path:
    """Select the authored fallback before composition in a disposable layer."""
    record_dir.mkdir(parents=True, exist_ok=True)
    source_asset = str(source_stage).replace("@", r"\@")
    output = record_dir / "mine_validation_entry.usda"
    output.write_text(
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
                '    over "Robot" (variants = { string robot_model = "proxy_fallback" }) {}',
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return output


def _settle(world: object, frames: int, *, render: bool = True) -> None:
    for _ in range(frames):
        world.step(render=render)


def _save_rgba(image: object, output: Path) -> None:
    import numpy as np
    from PIL import Image

    pixels = np.asarray(image)
    _require(pixels.ndim == 3 and pixels.shape[2] >= 3, "camera returned no RGB image")
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels * 255 if pixels.max() <= 1 else pixels, 0, 255).astype(np.uint8)
    Image.fromarray(pixels[:, :, :3], mode="RGB").save(output)


def _render_cameras(world: object, paths: Iterable[str], outputs: Iterable[Path]) -> None:
    from isaacsim.sensors.camera import Camera

    for path, output in zip(paths, outputs, strict=True):
        camera = Camera(prim_path=path, resolution=(640, 360), frequency=30)
        camera.initialize()
        _settle(world, 3, render=True)
        _save_rgba(camera.get_rgba(), output)
        camera.destroy()


def _beam_z(prim: object) -> float:
    positions, _ = prim.get_world_poses()
    return float(positions.numpy()[0][2])


def _find_rover_chassis_path(stage: object) -> str:
    """Find the actual rigid chassis below the referenced NVIDIA asset.

    This avoids hard-coding a child path from a particular Nova Carter release.
    """
    from pxr import Usd, UsdPhysics

    robot = stage.GetPrimAtPath("/World/Robot")
    rigid_bodies = [
        prim for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    candidates = [prim for prim in rigid_bodies if "chassis" in prim.GetName().lower()]
    if candidates:
        return str(candidates[0].GetPath())
    if rigid_bodies:
        return str(rigid_bodies[0].GetPath())
    if robot.HasAPI(UsdPhysics.RigidBodyAPI):
        return str(robot.GetPath())
    _require(False, "could not find a rigid rover chassis below /World/Robot")
    raise AssertionError("unreachable")


def _roof_support_smoke_test(world: object, stage: object, *, render: bool, record_dir: Path) -> None:
    """Run a bounded rover-contact test without writing the base stage.

    The chassis is positioned only as initial experiment state, then given a
    forward velocity.  It physically contacts the dynamic support; no beam or
    support pose is edited after stepping starts.
    """
    import numpy as np
    from isaacsim.core.experimental.prims import RigidPrim, XformPrim

    beam = XformPrim("/World/FailureZones/RoofSupportZone/BeamPrimary")
    support = XformPrim("/World/FailureZones/RoofSupportZone/SupportPrimary")
    chassis = RigidPrim(_find_rover_chassis_path(stage))
    initial_beam_z = _beam_z(beam)
    initial_support_poses, _ = support.get_world_poses()
    initial_support_pose = initial_support_poses.numpy()[0]
    _settle(world, 30, render=render)
    stable_beam_z = _beam_z(beam)
    _require(
        abs(stable_beam_z - initial_beam_z) < 0.25,
        f"roof support was not stable before rover contact: z {initial_beam_z:.3f} -> {stable_beam_z:.3f}",
    )

    # The explicit start pose is an ephemeral experiment setup.  The actual
    # disturbance is chassis velocity/contact, not an authored collapse event.
    chassis.set_world_poses(positions=np.array([[24.0, -1.35, 0.35]], dtype=np.float32))
    chassis_pose_before, _ = chassis.get_world_poses()
    commanded_velocity = np.array([[0.0, 2.4, 0.0]], dtype=np.float32)
    for _ in range(150):
        chassis.set_velocities(linear_velocities=commanded_velocity)
        world.step(render=render)
    chassis_pose_after, _ = chassis.get_world_poses()
    final_beam_z = _beam_z(beam)
    final_support_poses, _ = support.get_world_poses()
    final_support_pose = final_support_poses.numpy()[0]
    support_displacement = float(final_support_pose[1] - initial_support_pose[1])
    _require(
        support_displacement > 0.45,
        f"rover did not displace SupportPrimary enough: delta-y {support_displacement:.3f}",
    )
    _require(
        final_beam_z < stable_beam_z - 0.8,
        f"beam did not fall after support contact: z {stable_beam_z:.3f} -> {final_beam_z:.3f}",
    )
    record_dir.mkdir(parents=True, exist_ok=True)
    if render:
        _render_cameras(
            world,
            ("/World/Sensors/RoofSupportCamera",),
            (record_dir / "roof_support_after_contact.png",),
        )
    print(
        "roof-support rover-contact smoke test: "
        f"chassis {chassis_pose_before.numpy()[0].tolist()} -> {chassis_pose_after.numpy()[0].tolist()}; "
        f"support delta-y {support_displacement:.3f}; beam z {stable_beam_z:.3f} -> {final_beam_z:.3f}"
    )


def main() -> int:
    args = _args()
    source_stage_path = args.stage.resolve()
    _require(source_stage_path.is_file(), f"entry stage not found: {source_stage_path}")
    stage_path = _write_validation_entry(source_stage_path, args.record_dir.resolve())

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "None" if args.skip_render else "RayTracedLighting",
            # Keep the disposable validator from waiting for all RTX material
            # PSOs before the first sensor frame on cloud workstations.
            "/rtx/materialDb/syncLoads": False,
            "/rtx/hydra/materialSyncLoads": False,
            "/omni/kit/plugin/syncUsdLoads": False,
        }
    )
    import omni.usd
    from isaacsim.core.api import World

    try:
        context = omni.usd.get_context()
        _require(context.open_stage(str(stage_path)), f"failed to open stage: {stage_path}")
        for _ in range(300):
            app.update()
            if not context.get_stage_loading_status()[2]:
                break
        stage = context.get_stage()
        _require(stage is not None, "Isaac did not provide an opened stage")
        _validate_stage(stage)
        print(f"opened and validated: {stage_path}")
        world = World(stage_units_in_meters=1.0)
        world.reset()
        _settle(world, args.frames, render=not args.skip_render)
        if not args.skip_render:
            _render_cameras(
                world,
                ("/World/Sensors/ResearchCamera", "/World/Sensors/OverviewCamera"),
                (WORLD_DIRECTORY / "previews" / "research_camera.png", WORLD_DIRECTORY / "previews" / "overview_camera.png"),
            )
            print("rendered: previews/research_camera.png, previews/overview_camera.png")
        if args.smoke_test:
            _roof_support_smoke_test(world, stage, render=not args.skip_render, record_dir=args.record_dir.resolve())
        return 0
    finally:
        # Isaac Sim 6.0.1 can abort when tearing down active camera task groups.
        # This disposable container validator exits after flushing its files,
        # matching the workstation handoff's verified shutdown strategy.
        pass


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        print(f"mine-world validation failed: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        # The 6.0.1 image can abort in Kit camera task-group teardown. This is
        # a disposable validation container, so mirror the verified Phase 2
        # runner: flush artifacts and let process exit release Kit resources.
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(exit_code)
