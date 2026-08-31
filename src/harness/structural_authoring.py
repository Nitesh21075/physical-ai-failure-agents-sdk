"""Pure compiler helpers for bounded, model-directed structural assemblies."""

from __future__ import annotations

import json

from harness.mine_world import WITNESS_CAMERA_PATH

GENERATED_STRUCTURE_ROOT = "/World/Experiment/GeneratedStructure"


def runtime_zone_for_scenario(zone: dict, spec: dict[str, object] | None) -> dict:
    """Resolve compiler-owned prim paths and evidence settings without accepting paths."""
    runtime_zone = json.loads(json.dumps(zone))
    structure = spec.get("authored_structure") if spec is not None else None
    if not structure:
        return runtime_zone
    bodies = structure["bodies"]
    by_role = {body["semantic_role"]: body for body in bodies}
    impact = by_role["impact_support"]
    falling = by_role["falling_body"]
    impact_path = f"{GENERATED_STRUCTURE_ROOT}/{impact['body_id']}"
    falling_path = f"{GENERATED_STRUCTURE_ROOT}/{falling['body_id']}"
    target = [
        (float(impact["position_xyz_m"][axis]) + float(falling["position_xyz_m"][axis]))
        / 2.0
        for axis in range(3)
    ]
    runtime_zone.update(
        {
            "disabled_source_hazard_root": zone.get("root"),
            "support": impact_path,
            "primary_falling_body": falling_path,
            "expected_initial_support_position_xyz_m": impact["position_xyz_m"],
            "expected_initial_falling_body_position_xyz_m": falling["position_xyz_m"],
            "pre_actuation_position_tolerance_m": {
                "rover": 0.75,
                "support": 0.35,
                "primary_falling_body": 0.5,
            },
            "witness_camera": {
                "prim": zone.get("witness_camera", {}).get("prim", WITNESS_CAMERA_PATH),
                "eye_xyz_m": [target[0] + 4.2, target[1] - 5.8, target[2] + 2.6],
                "target_xyz_m": target,
                "focal_length_mm": 22.0,
            },
            "semantic_targets": [
                {"prim": impact_path, "label": "generated_impact_support"},
                {"prim": falling_path, "label": "generated_falling_body"},
            ],
            "required_semantic_labels": {
                "tracking": "generated_impact_support",
                "witness": "generated_falling_body",
            },
            "collapse_threshold_m": structure["collapse_threshold_m"],
            "task": "model_directed_structural_contact",
            "reactor_prompt": (
                "Isaac witness-camera view of a Nova Carter rover approaching a newly authored "
                f"warehouse structure. Preserve this structure and scene: {structure['description']}"
            ),
        }
    )
    return runtime_zone
