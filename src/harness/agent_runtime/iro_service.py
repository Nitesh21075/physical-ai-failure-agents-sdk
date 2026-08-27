"""Fixed-command execution boundary for keyless Isaac Replicator Object scenes."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from harness.agent_runtime.iro_spec import IROBuildStore, compile_iro_config
from harness.persistence.store import ExperimentStore


class IsaacIROToolService:
    IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.1"
    EXTENSION = "isaacsim.replicator.object.core"
    EXPERIENCE = "/isaac-sim/apps/isaacsim.exp.action_and_event_data_generation.base.kit"

    def __init__(
        self, project_root: str | Path, runs_root: str | Path, store: ExperimentStore
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.runs_root = Path(runs_root).resolve()
        self.store = store
        self.builds = IROBuildStore(self.runs_root)
        if self.runs_root != self.project_root / "runs":
            raise ValueError("IRO runs_root must be the cloned project's runs directory")

    def run_built_scene(self, scene_id: str) -> dict:
        spec, _, build = self.builds.load(scene_id)
        run_id = str(uuid4())
        run_directory = self.runs_root / "isaac-iro" / run_id
        output_directory = run_directory / "output"
        run_directory.mkdir(parents=True, exist_ok=False)

        container_run_directory = f"/workspace/project/runs/isaac-iro/{run_id}"
        container_output = f"{container_run_directory}/output"
        run_config = compile_iro_config(spec, output_path=container_output)
        config_path = run_directory / "iro_config.yaml"
        config_path.write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        command = self._command(run_id)
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=950,
            check=False,
        )
        log_directory = self.runs_root / "agent-isaac-logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        log_path = log_directory / f"iro-{run_id}.log"
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Isaac IRO runner failed with exit {completed.returncode}: "
                f"{completed.stdout[-4000:]}"
            )

        images = sorted((output_directory / "images").glob("*"))
        images = [path for path in images if path.is_file() and path.stat().st_size > 0]
        descriptions = sorted((output_directory / "descriptions").glob("*.yaml"))
        if len(images) < spec.frame_count:
            raise RuntimeError(
                f"IRO completed without the requested RGB evidence: "
                f"expected {spec.frame_count}, found {len(images)}"
            )

        created_at = datetime.now(UTC).isoformat()
        scenario = {
            "scenario_id": scene_id,
            "environment": "iro_primitive_room",
            "task": "synthetic_scene_generation",
            "seed": spec.seed,
            "parameters": spec.normalized(),
            "hazards": {},
            "scenario_digest": build["scene_digest"],
        }
        output_counts = {
            name: len([path for path in directory.glob("*") if path.is_file()])
            for name in (
                "images",
                "labels",
                "3d_labels",
                "segmentation",
                "instance_id_segmentation",
                "depth",
                "normal",
                "descriptions",
                "usd",
            )
            if (directory := output_directory / name).is_dir()
        }
        evaluation = {
            "task_success": True,
            "environmental_failure": False,
            "failure_type": None,
            "severity": "none",
            "robot_safety_events": [],
            "terminal": True,
            "metrics": {
                "requested_frames": spec.frame_count,
                "rgb_frames": len(images),
                "description_files": len(descriptions),
                "output_counts": output_counts,
            },
            "evidence_refs": [str(path) for path in images],
        }
        summary = {
            "run_id": run_id,
            "backend": "isaac_iro",
            "scene_id": scene_id,
            "scene_digest": build["scene_digest"],
            "config_digest": build["config_digest"],
            "extension": self.EXTENSION,
            "extension_version": build["compiler"]["extension_version"],
            "isaac_sim_image": self.IMAGE,
            "api_key_used": False,
            "evidence_scope": (
                "synthetic scene and annotation generation; not a physical-failure outcome "
                "or real-world ground truth"
            ),
            "output_counts": output_counts,
            "created_at": created_at,
        }
        manifest_path = run_directory / "run_manifest.json"
        (run_directory / "scenario.json").write_text(
            json.dumps(scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_directory / "result.json").write_text(
            json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_path.write_text(
            json.dumps(
                {"scenario": scenario, "evaluation": evaluation, "summary": summary},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.store.upsert_experiment(
            {
                "run_id": run_id,
                "backend": "isaac_iro",
                "scenario": scenario,
                "trajectory_ref": str(manifest_path),
                "evaluation": evaluation,
                "created_at": created_at,
            },
            run_directory=run_directory,
        )
        self.store.register_artifact("experiment", run_id, "log", log_path)
        for path in run_directory.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            kind = "image" if suffix in {".jpg", ".jpeg", ".png"} else "metadata"
            self.store.register_artifact("experiment", run_id, kind, path)
        return {
            "run_id": run_id,
            "scene_id": scene_id,
            "scene_digest": build["scene_digest"],
            "status": "generated",
            "output_counts": output_counts,
            "rgb_frame_refs": [str(path) for path in images],
            "summary_ref": str(run_directory / "summary.json"),
            "evidence_scope": summary["evidence_scope"],
        }

    def _command(self, run_id: str) -> list[str]:
        container_run_directory = f"/workspace/project/runs/isaac-iro/{run_id}"
        return [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--network",
            "none",
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
                f"trap 'chown -R {os.getuid()}:{os.getgid()} {container_run_directory}' EXIT; "
                f"/isaac-sim/kit/kit {self.EXPERIENCE} --no-window "
                f"--enable {self.EXTENSION} --allow-root "
                f"--/log/file={container_run_directory}/isaac.log "
                "--/windowless=True "
                f"--/config/file={container_run_directory}/iro_config.yaml"
            ),
        ]
