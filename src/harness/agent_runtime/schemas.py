"""Model-visible structured outputs for one bounded research step."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ResearchStepStatus(StrEnum):
    COMPLETED = "completed"
    WAITING_FOR_REACTOR = "waiting_for_reactor"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    BLOCKED = "blocked"
    FAILED = "failed"


class ResearchStepSummary(BaseModel):
    status: ResearchStepStatus
    hypothesis: str = Field(description="Concise testable hypothesis, without hidden reasoning.")
    evidence_summary: str = Field(description="Only evidence actually returned by tools.")
    isaac_run_id: str | None = None
    pair_id: str | None = None
    physics_outcome: str | None = None
    comparison_status: str | None = None
    next_recommendation: str
    user_action_required: str | None = None
