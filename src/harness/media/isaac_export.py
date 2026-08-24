"""Export Isaac camera arrays or image frames into derived previews without touching raw frames."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _frame_paths(run_directory: Path, project_root: Path) -> list[Path]:
    trajectory = run_directory / "trajectory.jsonl"
    refs: list[str] = []
    if trajectory.exists():
        for line in trajectory.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            observation = item.get("observation", {})
            result = item.get("result", {}).get("observation", {})
            refs.extend(observation.get("sensor_refs", []))
            refs.extend(result.get("sensor_refs", []))
    frames: list[Path] = []
    for ref in refs:
        path = Path(ref)
        if not path.exists() and ref.startswith("/workspace/project/"):
            path = project_root / ref.removeprefix("/workspace/project/")
        if path.suffix.lower() in {".npy", ".png", ".jpg", ".jpeg", ".webp"} and path.exists() and path not in frames:
            frames.append(path)
    discovered = sorted(
        path
        for path in (run_directory / "camera").glob("**/*")
        if path.suffix.lower() in {".npy", ".png", ".jpg", ".jpeg", ".webp"}
    )
    if discovered:
        frames = discovered
    return frames


def _load_camera_array(array_path: Path) -> np.ndarray | None:
    try:
        array = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        return None
    return array


def _write_png(array_path: Path, target: Path) -> bool:
    if array_path.suffix.lower() != ".npy":
        try:
            with Image.open(array_path) as image:
                image.convert("RGB").save(target)
            return True
        except (OSError, ValueError):
            return False
    array = _load_camera_array(array_path)
    if array is None:
        return False
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array, "RGBA" if array.shape[-1] == 4 else "RGB").save(target)
    return True


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("MP4 export requires ffmpeg or imageio-ffmpeg") from exc


def export_isaac_replay(run_directory: str | Path, fps: int = 5) -> dict[str, Any]:
    """Create PNG previews, thumbnail, MP4, and a manifest under ``media/``.

    The source ``.npy`` arrays are only read.  The returned paths are suitable
    for registration in :class:`harness.persistence.ExperimentStore`.
    """
    if fps < 1:
        raise ValueError("fps must be at least one")
    run_directory = Path(run_directory).resolve()
    runs_root = next((parent for parent in run_directory.parents if parent.name == "runs"), None)
    project_root = runs_root.parent if runs_root is not None else run_directory.parent
    source_frames = _frame_paths(run_directory, project_root)
    if not source_frames:
        raise FileNotFoundError(f"no Isaac camera frames found for {run_directory}")
    output = run_directory / "media" / "isaac_replay"
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # These are derived files.  Remove only our own previous previews so a
    # re-export cannot accidentally retain a frame from an earlier run.
    for stale_preview in frames_dir.glob("frame_*.png"):
        stale_preview.unlink()
    png_frames: list[Path] = []
    for source in source_frames:
        target = frames_dir / f"frame_{len(png_frames):04d}.png"
        if _write_png(source, target):
            png_frames.append(target)
    if not png_frames:
        raise ValueError(f"no valid RGBA/RGB camera arrays found for {run_directory}")
    thumbnail = output / "thumbnail.png"
    with Image.open(png_frames[0]) as image:
        image.thumbnail((480, 270))
        image.save(thumbnail)
    video = output / "replay.mp4"
    subprocess.run(
        [
            _ffmpeg_executable(), "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    manifest = {
        "kind": "isaac_camera_replay", "fps": fps,
        "raw_frame_paths": [str(path) for path in source_frames],
        "preview_frame_paths": [str(path) for path in png_frames],
        "thumbnail_path": str(thumbnail), "video_path": str(video),
    }
    manifest_path = output / "media.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
