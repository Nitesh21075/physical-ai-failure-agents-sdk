# Agents SDK Runtime Map

This document is authoritative for the additive `MineFailureResearcher`
runtime. Older research and primitive Isaac-worker modules remain in the
repository as a tested fallback, but they are not called by this runtime.

## What owns the loop

`MineFailureResearchService.run_step()` calls the official OpenAI Agents SDK:

```python
result = await Runner.run(
    agent,
    instruction,
    context=context,
    session=session,
    hooks=CampaignRunHooks(),
    max_turns=12,
    run_config=run_config,
)
```

The SDK sends the instruction, agent instructions, tool schemas, and session
history to the configured OpenAI model. The model selects zero or more tools.
The SDK dispatches those tool calls, appends results to the conversation, and
calls the model again. The host service does not select tool order.

## End-to-end call graph

```text
CLI or dashboard instruction
  [scripts/run_agents_researcher.py:_main]
  [src/harness/dashboard/app.py:run_agent_step]
       |
       v
campaign service and local dependencies
  [agent_runtime/service.py:MineFailureResearchService.run_step]
  [agent_runtime/context.py:AgentRuntimeContext]
       |
       +--> conversation memory
       |    [agents.SQLiteSession]
       |    [runs/agent_sessions.sqlite3]
       |
       +--> scientific memory
       |    [persistence/store.py:ExperimentStore]
       |    [research/campaign.py:ResearchCampaignStore]
       |    [runs/experiments.sqlite3 + runs/]
       |
       v
single researcher definition
  [agent_runtime/researcher.py:create_researcher]
  Agent(name="MineFailureResearcher", tools=AGENT_TOOLS,
        output_type=ResearchStepSummary)
       |
       v
SDK-owned model/tool loop
  [agents.Runner.run]
       |
       +--> LLM chooses get_campaign_state
       |    [agent_runtime/tools.py:get_campaign_state]
       |
       +--> LLM chooses inspect_mine_world
       |    [agent_runtime/tools.py:inspect_mine_world]
       |    -> [assets/worlds/mine_v1/manifest.json]
       |
       +--> LLM chooses get_recent_experiments
       |    [agent_runtime/tools.py:get_recent_experiments]
       |    -> [ExperimentStore]
       |
       +--> LLM may choose one real Isaac experiment
       |    [agent_runtime/tools.py:run_mine_roof_support_experiment]
       |    -> validate arguments and claim per-turn/campaign budget
       |       [AgentRuntimeContext.claim_isaac_budget]
       |    -> fixed host service
       |       [agent_runtime/isaac_service.py:MineIsaacToolService.run]
       |    -> fixed Docker/Isaac command
       |       [/isaac-sim/python.sh scripts/run_mine_rover_experiment.py]
       |    -> Nova Carter wheel targets, PhysX, camera, measurements
       |       [scripts/run_mine_rover_experiment.py]
       |    -> run files and scientific index
       |       [ExperimentStore.upsert_experiment/register_artifact]
       |
       +--> LLM may choose Reactor preparation
       |    [agent_runtime/tools.py:prepare_reactor_comparison]
       |    -> [pairing.py:PairedCaptureService.prepare]
       |    -> real Isaac seed frame + generated prompt + pending pair
       |    -> model returns waiting_for_reactor
       |
       +--> human completes browser capture
       |    [dashboard/static/reactor-live.js]
       |    -> [dashboard/app.py:finish_pair_capture]
       |    -> [pairing.py:PairedCaptureService.finalize]
       |    -> real WebRTC media + Reactor experiment + initial Plan C pair
       |
       +--> later Runner call with same campaign/session
            -> LLM chooses get_pair_status
               [agent_runtime/tools.py:get_pair_status]
            -> LLM may choose assess_and_compare_pair
               [agent_runtime/tools.py:assess_and_compare_pair]
            -> decode real frames
            -> [research/visual_assessment.py:OpenAIResponsesVisualAssessor]
            -> [comparison/plan_c.py:PlanCComparator.compare]
            -> persist comparison without replacing Isaac physics truth
            -> LLM proposes the next hypothesis/experiment
       |
       v
structured operator result
  [agent_runtime/schemas.py:ResearchStepSummary]
```

Every LLM/tool lifecycle event is observed by
`agent_runtime/hooks.py:CampaignRunHooks` and appended to campaign events. The
hooks persist tool names, call IDs, timestamps, statuses, trace IDs, and compact
results; they do not persist hidden reasoning.

## Model-visible tools

Only these seven functions are exposed:

1. `get_campaign_state`
2. `inspect_mine_world`
3. `get_recent_experiments`
4. `run_mine_roof_support_experiment`
5. `prepare_reactor_comparison`
6. `get_pair_status`
7. `assess_and_compare_pair`

There is no shell, arbitrary Python, unrestricted Docker, generic file writer,
Codex tool, or unrestricted Omniverse tool.

## Deterministic code versus agent decisions

The intelligent/model-selected decisions are which tool to call, its bounded
arguments, whether evidence is sufficient, the hypothesis, and the next-step
rationale. Deterministic code deliberately handles safety, Docker invocation,
physics measurement, media persistence, validation, and comparison rules.
That division is expected in a tool-using agent: intelligence chooses actions;
trusted code performs and records side effects.

## Legacy fallback boundary

The following path is older and not invoked by `MineFailureResearchService`:

```text
scripts/run_research_iteration.py
  -> research/ResearchAgent
  -> research/OpenAIResponsesResearchModel
  -> research/ScenarioCompiler
  -> research/CampaignExecutor
  -> primitive isaac_worker
```

Related generic environment adapters, mock agents, old proposal schemas, and
historical planning documents can be removed in a later cleanup branch after
their useful persistence, comparison, media, and dashboard dependencies are
classified. They are retained here so publication does not silently remove the
known fallback.
