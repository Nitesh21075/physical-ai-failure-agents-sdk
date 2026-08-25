#!/usr/bin/env python3
"""Convert the pinned LTU DARPA SubT cave mesh into a project-owned USD asset.

Run this script with Isaac Sim 6.0.1's ``python.sh``.  It converts one audited
OBJ input; it does not parse a Gazebo world, download models, or copy Gazebo
physics and lighting into the USD stage.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

EXPECTED_SOURCE_COMMIT = "a15a1107e638eae090335d2d6f36c623439aaa7f"
EXPECTED_MODEL_PATH = Path("worlds/models/cave_world/meshes/cave_world.obj")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repository",
        type=Path,
        required=True,
        help="checkout of github.com/LTU-RAI/darpa_subt_worlds at the pinned commit",
    )
    parser.add_argument("--output", type=Path, required=True, help="output .usd or .usdc file")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkout_commit(repository: Path) -> str:
    head = (repository / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = repository / ".git" / head.removeprefix("ref: ")
    if not ref.is_file():
        raise ValueError(f"cannot resolve checkout HEAD ref: {head}")
    return ref.read_text(encoding="utf-8").strip()


async def _convert(source: Path, output: Path) -> None:
    import omni.kit.asset_converter

    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animation = True
    context.ignore_cameras = True
    context.single_mesh = False
    context.smooth_normals = False
    context.preview_surface = True
    context.use_meter_as_world_unit = True

    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(source), str(output), lambda _progress, _total: None, context
    )
    if not await task.wait_until_finished():
        raise RuntimeError(
            f"asset conversion failed with status {task.get_status()}: {task.get_error_message()}"
        )


def main() -> int:
    args = _arguments()
    source_repository = args.source_repository.resolve()
    source = source_repository / EXPECTED_MODEL_PATH
    license_path = source_repository / "LICENSE"
    if not source.is_file() or not license_path.is_file():
        raise FileNotFoundError(
            f"expected LTU model and license below source repository: {source_repository}"
        )
    actual_commit = _checkout_commit(source_repository)
    if actual_commit != EXPECTED_SOURCE_COMMIT:
        raise ValueError(
            f"LTU checkout must be at {EXPECTED_SOURCE_COMMIT}, found {actual_commit}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    SimulationApp({"headless": True})
    from isaacsim.core.experimental.utils.app import enable_extension

    enable_extension("omni.kit.asset_converter")
    asyncio.get_event_loop().run_until_complete(_convert(source, args.output.resolve()))
    if not args.output.is_file():
        raise RuntimeError(f"asset converter did not create output: {args.output}")
    provenance = {
        "source_repository": "https://github.com/LTU-RAI/darpa_subt_worlds",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_model": str(EXPECTED_MODEL_PATH),
        "source_sha256": _sha256(source),
        "converted_asset": args.output.name,
        "converted_sha256": _sha256(args.output),
        "converter": "Isaac Sim 6.0.1 omni.kit.asset_converter",
        "materials_loaded": True,
        "gazebo_physics_or_lighting_imported": False,
        "texture_sha256": {
            str(path.relative_to(args.output.parent)): _sha256(path)
            for path in sorted((args.output.parent / "textures").glob("*"))
            if path.is_file()
        },
    }
    args.output.with_suffix(args.output.suffix + ".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:  # noqa: BLE001 - report native converter failures at the boundary
        print(f"LTU SubT cave import failed: {error}", file=sys.stderr)
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac Sim 6.0.1 can abort while tearing down a completed converter task.
    # This is a disposable process and all outputs above are synchronously written.
    os._exit(exit_code)
