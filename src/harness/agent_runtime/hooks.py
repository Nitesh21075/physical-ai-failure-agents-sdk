"""Persist observable Agents SDK lifecycle events without model reasoning."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from agents import RunHooks

from harness.agent_runtime.context import AgentRuntimeContext


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _summary(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    try:
        encoded = json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        encoded = str(value)
    return encoded[:2000]


class CampaignRunHooks(RunHooks[AgentRuntimeContext]):
    def __init__(
        self, event_callback: Callable[[str, dict[str, Any]], None] | None = None
    ) -> None:
        self.event_callback = event_callback

    def _record(self, context: Any, event_type: str, payload: dict[str, Any]) -> None:
        local = context.context
        event_payload = {"timestamp": _now(), "trace_id": local.trace_id, **payload}
        local.campaign_store.record_event(
            local.campaign_id,
            event_type,
            event_payload,
        )
        if self.event_callback is not None:
            self.event_callback(event_type, event_payload)

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        self._record(context, "agent_run_started", {"agent": agent.name})

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: Any, input_items: Any
    ) -> None:
        self._record(
            context, "llm_started", {"agent": agent.name, "input_item_count": len(input_items)}
        )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self._record(
            context,
            "llm_finished",
            {"agent": agent.name, "output_item_count": len(response.output)},
        )

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        call_id = getattr(context, "tool_call_id", None)
        context.context.current_tool_calls[str(call_id or tool.name)] = {
            "started_at": _now(),
            "tool": tool.name,
        }
        self._record(
            context,
            "tool_started",
            {"agent": agent.name, "tool_name": tool.name, "tool_call_id": call_id},
        )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: object) -> None:
        call_id = getattr(context, "tool_call_id", None)
        self._record(
            context,
            "tool_finished",
            {
                "agent": agent.name,
                "tool_name": tool.name,
                "tool_call_id": call_id,
                "status": "completed",
                "compact_result": _summary(result),
            },
        )

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        self._record(
            context, "agent_run_finished", {"agent": agent.name, "output": _summary(output)}
        )
