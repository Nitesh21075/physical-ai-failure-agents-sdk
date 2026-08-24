"""Local dependencies and per-turn execution bounds for Agents SDK tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.pairing import PairedCaptureService
from harness.persistence import ExperimentStore
from harness.research import ResearchCampaignStore


@dataclass(frozen=True, slots=True)
class AgentRuntimeConfig:
    project_root: Path
    runs_root: Path
    database_path: Path
    session_database_path: Path
    mine_manifest_path: Path
    max_turns: int = 12
    max_recent_events: int = 12

    @classmethod
    def for_project(cls, project_root: str | Path) -> AgentRuntimeConfig:
        root = Path(project_root).resolve()
        runs = (root / "runs").resolve()
        return cls(
            project_root=root,
            runs_root=runs,
            database_path=runs / "experiments.sqlite3",
            session_database_path=runs / "agent_sessions.sqlite3",
            mine_manifest_path=root / "assets" / "worlds" / "mine_v1" / "manifest.json",
        )


@dataclass(slots=True)
class AgentRuntimeContext:
    campaign_id: str
    config: AgentRuntimeConfig
    experiment_store: ExperimentStore
    campaign_store: ResearchCampaignStore
    paired_capture: PairedCaptureService
    mine_isaac_service: Any
    model: str
    allow_multiple_isaac_experiments: bool = False
    isaac_experiments_this_turn: int = 0
    trace_id: str | None = None
    current_tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)

    def claim_isaac_budget(self) -> None:
        if self.isaac_experiments_this_turn >= 1 and not self.allow_multiple_isaac_experiments:
            raise RuntimeError("this research step authorizes at most one new Isaac experiment")
        campaign = self.campaign_store.get_campaign(self.campaign_id)
        if campaign is None:
            raise KeyError(f"unknown campaign: {self.campaign_id}")
        if campaign["experiments_used"] >= campaign["experiment_budget"]:
            raise RuntimeError("campaign experiment budget is exhausted")
        self.isaac_experiments_this_turn += 1

    def require_under_runs(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.config.runs_root)
        except ValueError as error:
            raise ValueError(
                "path must resolve underneath the configured runs directory"
            ) from error
        return resolved
