"""Host entrypoint: context/session setup, then the SDK owns model-tool iteration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from agents import RunConfig, Runner, SQLiteSession
from agents.tracing import gen_trace_id

from harness.agent_runtime.context import AgentRuntimeConfig, AgentRuntimeContext
from harness.agent_runtime.hooks import CampaignRunHooks
from harness.agent_runtime.iro_service import IsaacIROToolService
from harness.agent_runtime.isaac_service import MineIsaacToolService
from harness.agent_runtime.researcher import create_researcher
from harness.agent_runtime.schemas import ResearchStepSummary
from harness.pairing import PairedCaptureService
from harness.persistence.store import ExperimentStore
from harness.research.campaign import ResearchCampaignStore


class MineFailureResearchService:
    def __init__(
        self,
        project_root: str | Path,
        *,
        model: str,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = AgentRuntimeConfig.for_project(project_root)
        self.config.runs_root.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.event_callback = event_callback
        self.experiment_store = ExperimentStore(self.config.database_path)
        self.campaign_store = ResearchCampaignStore(self.config.database_path)

    def create_campaign(self, objective: str, *, experiment_budget: int = 3) -> str:
        return self.campaign_store.create_campaign(
            objective,
            experiment_budget=experiment_budget,
            model_provider="openai_agents_sdk",
            model_name=self.model,
            capability_version="controller-registry-v1",
            simulator_metadata={"container_image": "nvcr.io/nvidia/isaac-sim:6.0.1"},
        )

    async def run_step(self, campaign_id: str, instruction: str) -> ResearchStepSummary:
        if self.campaign_store.get_campaign(campaign_id) is None:
            raise KeyError(f"unknown campaign: {campaign_id}")
        trace_id = gen_trace_id()
        context = AgentRuntimeContext(
            campaign_id=campaign_id,
            config=self.config,
            experiment_store=self.experiment_store,
            campaign_store=self.campaign_store,
            paired_capture=PairedCaptureService(self.experiment_store, self.config.runs_root),
            mine_isaac_service=MineIsaacToolService(
                self.config.project_root, self.config.runs_root, self.experiment_store
            ),
            iro_service=IsaacIROToolService(
                self.config.project_root, self.config.runs_root, self.experiment_store
            ),
            model=self.model,
            trace_id=trace_id,
        )
        session = SQLiteSession(campaign_id, db_path=self.config.session_database_path)
        agent = create_researcher(self.model)
        run_config = RunConfig(
            workflow_name="Physical AI Failure Research",
            trace_id=trace_id,
            group_id=campaign_id,
            trace_metadata={"campaign_id": campaign_id},
        )
        try:
            result = await Runner.run(
                agent,
                instruction,
                context=context,
                session=session,
                hooks=CampaignRunHooks(self.event_callback),
                max_turns=self.config.max_turns,
                run_config=run_config,
            )
            output = result.final_output
            if not isinstance(output, ResearchStepSummary):
                output = ResearchStepSummary.model_validate(output)
            usage = result.context_wrapper.usage
            self.campaign_store.record_event(
                campaign_id,
                "agent_run_usage",
                {
                    "trace_id": trace_id,
                    "requests": usage.requests,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                },
            )
            self.campaign_store.consume_instructions(campaign_id)
            return output
        except Exception as error:
            self.campaign_store.record_event(
                campaign_id,
                "agent_run_failed",
                {
                    "trace_id": trace_id,
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                },
            )
            raise
        finally:
            session.close()
