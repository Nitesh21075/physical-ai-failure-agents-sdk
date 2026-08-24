from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from agents import SQLiteSession

from harness.agent_runtime.context import AgentRuntimeConfig, AgentRuntimeContext
from harness.agent_runtime.isaac_service import MineIsaacToolService
from harness.agent_runtime.researcher import create_researcher
from harness.persistence import ExperimentStore
from harness.research import ResearchCampaignStore


def _context(tmp_path: Path, *, budget: int = 2) -> AgentRuntimeContext:
    root = tmp_path / "project"
    (root / "runs").mkdir(parents=True)
    database = root / "runs" / "experiments.sqlite3"
    campaigns = ResearchCampaignStore(database)
    campaign_id = campaigns.create_campaign("Find roof support boundary", experiment_budget=budget)
    return AgentRuntimeContext(
        campaign_id=campaign_id,
        config=AgentRuntimeConfig.for_project(root),
        experiment_store=ExperimentStore(database),
        campaign_store=campaigns,
        paired_capture=object(),
        mine_isaac_service=object(),
        model="test-model",
    )


@pytest.mark.parametrize("speed", [-1.0, 0.09, 0.81, 9.0])
def test_isaac_speed_rejected_before_launch(speed: float) -> None:
    with pytest.raises(ValueError, match="between 0.1 and 0.8"):
        MineIsaacToolService._validate(speed, 300, 1)


def test_one_isaac_experiment_per_turn_guard(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.claim_isaac_budget()
    with pytest.raises(RuntimeError, match="at most one"):
        context.claim_isaac_budget()


def test_runs_path_guard(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assert context.require_under_runs(context.config.runs_root / "safe.json").name == "safe.json"
    with pytest.raises(ValueError, match="underneath"):
        context.require_under_runs(tmp_path / "outside.json")


def test_agent_has_only_seven_narrow_tools() -> None:
    names = [tool.name for tool in create_researcher("test-model").tools]
    assert names == [
        "get_campaign_state",
        "inspect_mine_world",
        "get_recent_experiments",
        "run_mine_roof_support_experiment",
        "prepare_reactor_comparison",
        "get_pair_status",
        "assess_and_compare_pair",
    ]
    assert not {"shell", "python", "docker", "filesystem"}.intersection(names)


def test_campaign_pair_linkage_is_persistent(tmp_path: Path) -> None:
    context = _context(tmp_path)
    iteration_id = context.campaign_store.begin_iteration(context.campaign_id, {})
    context.campaign_store.record_isaac_run(iteration_id, "isaac-run")
    context.campaign_store.record_plan_c_pair(context.campaign_id, "isaac-run", "pair-id")
    reopened = ResearchCampaignStore(context.config.database_path)
    latest = reopened.latest_iteration(context.campaign_id)
    assert latest["plan_c_pair_id"] == "pair-id"
    assert latest["state"] == "waiting_for_assessment"


def test_sqlite_agent_session_survives_reopen(tmp_path: Path) -> None:
    async def exercise() -> list[dict]:
        database = tmp_path / "agent_sessions.sqlite3"
        first = SQLiteSession("campaign-1", db_path=database)
        await first.add_items([{"role": "user", "content": "first hypothesis"}])
        first.close()
        reopened = SQLiteSession("campaign-1", db_path=database)
        items = await reopened.get_items()
        reopened.close()
        return items

    assert asyncio.run(exercise()) == [{"role": "user", "content": "first hypothesis"}]
