import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from harness.mine_world import (
    TRACKING_CAMERA_PATH,
    MineRoverExperiment,
    RoverDriveCommand,
    assess_visual_frame,
    select_wheel_dofs,
    write_reactor_seed_manifest,
)


def test_select_wheel_dofs_uses_loaded_names_not_a_hardcoded_robot_layout():
    left, right = select_wheel_dofs(
        ["caster_joint", "rear_right_wheel_joint", "front_left_wheel_joint", "front_right_wheel_joint", "rear_left_wheel_joint"]
    )
    assert left == ["front_left_wheel_joint", "rear_left_wheel_joint"]
    assert right == ["front_right_wheel_joint", "rear_right_wheel_joint"]
    with pytest.raises(ValueError, match="left and right"):
        select_wheel_dofs(["front_wheel_joint"])


def test_rover_experiment_has_hard_bounds():
    assert MineRoverExperiment("run-1", 7, RoverDriveCommand()).camera_resolution == (180, 320)
    with pytest.raises(ValueError, match="linear_velocity"):
        RoverDriveCommand(linear_velocity_mps=1.0)
    with pytest.raises(ValueError, match="run_id"):
        MineRoverExperiment("../unsafe", 7, RoverDriveCommand())
    with pytest.raises(ValueError, match="world_id"):
        MineRoverExperiment("run-1", 7, RoverDriveCommand(), world_id="../unsafe")


def test_reactor_seed_manifest_references_a_real_isaac_image_and_derived_layer(tmp_path):
    image = tmp_path / "frames" / "rgb_000001.png"
    image.parent.mkdir()
    Image.effect_noise((64, 64), 80).convert("RGB").save(image)
    layer = tmp_path / "session.usda"
    layer.write_text("#usda 1.0\n", encoding="utf-8")
    experiment = MineRoverExperiment("run-1", 7, RoverDriveCommand())

    output = write_reactor_seed_manifest(
        tmp_path, experiment, seed_image=image, source_stage="assets/worlds/mine_v1/mine_world.usda",
        session_layer=layer, rover_pose_before=[0, 0, 0], rover_pose_after=[1, 0, 0],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["camera_prim"] == TRACKING_CAMERA_PATH
    assert payload["camera_role"] == "tracking"
    assert payload["visual_quality"]["passed"] is True
    assert payload["seed_image_path"] == str(image)
    assert payload["source_authority"] == "physics_grounded_isaac_rgb"
    assert "not physical ground truth" in payload["usage"]


def test_mine_v2_subt_has_pinned_local_asset_and_keeps_mine_v1_separate():
    project_root = Path(__file__).resolve().parents[1]
    v1 = project_root / "assets" / "worlds" / "mine_v1"
    v2 = project_root / "assets" / "worlds" / "mine_v2_subt"
    manifest = json.loads((v2 / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (v2 / "vendor" / "ltu_darpa_subt" / "cave_world.usdc.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    converted = v2 / "vendor" / "ltu_darpa_subt" / "cave_world.usdc"

    assert manifest["world_id"] == "mine_v2_subt"
    assert manifest["entry_stage"] == "mine_world.usda"
    assert manifest["composition"]["gazebo_physics_imported"] is False
    assert manifest["composition"]["gazebo_lighting_imported"] is False
    assert provenance["source_commit"] == "a15a1107e638eae090335d2d6f36c623439aaa7f"
    assert hashlib.sha256(converted.read_bytes()).hexdigest() == provenance["converted_sha256"]
    for relative_path, expected_hash in provenance["texture_sha256"].items():
        texture = v2 / "vendor" / "ltu_darpa_subt" / relative_path
        assert hashlib.sha256(texture.read_bytes()).hexdigest() == expected_hash
    assert (v2 / "vendor" / "ltu_darpa_subt" / "LICENSE").is_file()
    assert (v1 / "manifest.json").is_file()


def test_warehouse_danger_is_native_usd_and_declares_bounded_rack_failure():
    project_root = Path(__file__).resolve().parents[1]
    world = project_root / "assets" / "worlds" / "warehouse_danger_v1"
    manifest = json.loads((world / "manifest.json").read_text(encoding="utf-8"))
    zone = manifest["failure_zones"]["rack_collapse"]

    assert manifest["world_id"] == "warehouse_danger_v1"
    assert manifest["composition"]["conversion_required"] is False
    assert manifest["composition"]["environment_is_native_isaac_usd"] is True
    assert manifest["composition"]["environment_collected_locally"] is False
    assert "art-direction acceptance is pending" in manifest["composition"][
        "presentation_status"
    ]
    assert zone["support"].startswith("/World/FailureZones/RackCollapseZone/")
    assert zone["primary_falling_body"].startswith(
        "/World/FailureZones/RackCollapseZone/"
    )
    assert zone["collapse_threshold_m"] > 0
    assert zone["reactor_seed_camera_role"] == "witness"
    assert zone["tracking_camera"]["eye_offset_robot_xyz_m"][0] <= -4.0
    assert (world / manifest["entry_stage"]).is_file()
    assert (world / "hazards" / "rack_collapse.usda").is_file()


def test_reactor_seed_uses_selected_world_id(tmp_path):
    image = tmp_path / "frame.png"
    Image.effect_noise((64, 64), 80).convert("RGB").save(image)
    experiment = MineRoverExperiment(
        "run-v2", 7, RoverDriveCommand(), world_id="mine_v2_subt"
    )

    output = write_reactor_seed_manifest(
        tmp_path,
        experiment,
        seed_image=image,
        source_stage="assets/worlds/mine_v2_subt/mine_world.usda",
        session_layer=tmp_path / "session.usda",
        rover_pose_before=[0, 0, 0],
        rover_pose_after=[0, 1, 0],
    )

    assert json.loads(output.read_text(encoding="utf-8"))["world_id"] == "mine_v2_subt"


def test_visual_frame_gate_rejects_blank_and_accepts_structured_content(tmp_path):
    blank = tmp_path / "blank.png"
    structured = tmp_path / "structured.png"
    Image.new("RGB", (64, 64), (44, 18, 2)).save(blank)
    Image.effect_noise((64, 64), 80).convert("RGB").save(structured)

    assert assess_visual_frame(blank)["passed"] is False
    assert assess_visual_frame(structured)["passed"] is True


def test_visual_frame_gate_rejects_structured_but_severely_underexposed_content(tmp_path):
    image = tmp_path / "underexposed.png"
    pixels = Image.effect_noise((64, 64), 80).convert("RGB").point(lambda value: value // 20)
    pixels.save(image)

    assessment = assess_visual_frame(image)

    assert assessment["passed"] is False
    assert assessment["mean_luminance"] < 12
    assert assessment["dark_pixel_fraction_below_10"] > 0.65
