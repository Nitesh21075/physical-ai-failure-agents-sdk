"""Fixed-command host boundary for the real Isaac Sim mine runner."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from harness.media.isaac_export import export_isaac_replay
from harness.persistence.store import ExperimentStore


class MineIsaacToolService:
    IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.1"
    MIN_SPEED_MPS = 0.1
    MAX_SPEED_MPS = 0.8
    MIN_CONTROL_STEPS = 60
    MAX_CONTROL_STEPS = 900

    def __init__(
        self, project_root: str | Path, runs_root: str | Path, store: ExperimentStore
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.runs_root = Path(runs_root).resolve()
        self.store = store
        if self.runs_root != self.project_root / "runs":
            raise ValueError("Isaac runs_root must be the cloned project's runs directory")

    def run(self, *, rover_linear_velocity_mps: float, control_steps: int, seed: int) -> dict:
        self._validate(rover_linear_velocity_mps, control_steps, seed)
        run_id = str(uuid4())
        container_runs = "/workspace/project/runs/isaac-mine"
        command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--network",
            "host",
            "--user",
            "0:0",
            "--entrypoint",
            "bash",
            "-e",
            "ACCEPT_EULA=Y",
            "-e",
            "PRIVACY_CONSENT=Y",
            "-e",
            "HOME=/tmp",
            "-e",
            "OMNI_KIT_ALLOW_ROOT=1",
            "-v",
            "mine-rover-isaac-cache:/tmp/.cache",
            "-v",
            "mine-rover-omniverse-data:/tmp/.nvidia-omniverse",
            "-v",
            f"{self.project_root}:/workspace/project",
            self.IMAGE,
            "-lc",
            (
                f"trap 'chown -R {os.getuid()}:{os.getgid()} {container_runs}/{run_id}' EXIT; "
                "cd /workspace/project && /isaac-sim/python.sh "
                "scripts/run_mine_rover_experiment.py "
                f"--runs-dir {container_runs} --run-id {run_id} "
                f"--failure-zone roof_support --linear-velocity-mps {rover_linear_velocity_mps} "
                f"--control-steps {control_steps} --seed {seed}"
            ),
        ]
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=False,
        )
        run_directory = self.runs_root / "isaac-mine" / run_id
        log_directory = self.runs_root / "agent-isaac-logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        log_path = log_directory / f"{run_id}.log"
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            tail = completed.stdout[-4000:]
            raise RuntimeError(f"Isaac mine runner failed with exit {completed.returncode}: {tail}")
        summary = self._read_json(run_directory / "summary.json")
        scenario = self._read_json(run_directory / "scenario.json")
        evaluation = self._read_json(run_directory / "result.json")
        record = {
            "run_id": run_id,
            "backend": "isaac_sim",
            "scenario": scenario,
            "trajectory_ref": str(run_directory / "trajectory.jsonl"),
            "evaluation": evaluation,
            "created_at": datetime.now(UTC).isoformat(),
        }
        replay_available = False
        try:
            export_isaac_replay(run_directory)
            replay_available = True
        except (FileNotFoundError, IndexError, OSError, ValueError):
            # Physics evidence remains valid if rendering/export is unavailable.
            replay_available = False
        self.store.upsert_experiment(record, run_directory=run_directory)
        self.store.register_artifact("experiment", run_id, "log", log_path)
        for path in run_directory.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            kind = (
                "image"
                if suffix in {".png", ".jpg", ".jpeg", ".webp"}
                else "video"
                if suffix in {".mp4", ".webm"}
                else "trajectory"
                if suffix == ".jsonl"
                else "metadata"
            )
            self.store.register_artifact("experiment", run_id, kind, path)
        return {
            "run_id": run_id,
            "parameters": scenario["parameters"],
            "physical_outcome": (
                "structural_collapse" if summary["structural_collapse"] else "roof_support_stable"
            ),
            "environmental_failure": summary["structural_collapse"],
            "measured_support_displacement_m": summary["support_displacement_m"],
            "measured_beam_displacement_m": summary["beam_displacement_m"],
            "measured_beam_vertical_drop_m": summary["beam_vertical_drop_m"],
            "failure_type": "structural_collapse" if summary["structural_collapse"] else None,
            "collapse_criterion": summary["collapse_criterion"],
            "rover_pose_before": summary["rover_pose_before"],
            "rover_pose_after": summary["rover_pose_after"],
            "artifact_directory": str(run_directory),
            "camera_frame_count": len(summary["camera_frames"]),
            "replay_available": replay_available,
        }

    @classmethod
    def _validate(cls, speed: float, control_steps: int, seed: int) -> None:
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise TypeError("rover_linear_velocity_mps must be numeric")
        if not cls.MIN_SPEED_MPS <= float(speed) <= cls.MAX_SPEED_MPS:
            raise ValueError(
                f"rover_linear_velocity_mps must be between {cls.MIN_SPEED_MPS} and {cls.MAX_SPEED_MPS}"
            )
        if isinstance(control_steps, bool) or not isinstance(control_steps, int):
            raise TypeError("control_steps must be an integer")
        if not cls.MIN_CONTROL_STEPS <= control_steps <= cls.MAX_CONTROL_STEPS:
            raise ValueError(
                f"control_steps must be between {cls.MIN_CONTROL_STEPS} and {cls.MAX_CONTROL_STEPS}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
            raise ValueError("seed must be an integer between 0 and 2147483647")

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))
