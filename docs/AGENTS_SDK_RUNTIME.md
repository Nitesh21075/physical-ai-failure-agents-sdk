# MineFailureResearcher: authoritative runtime context

Last updated: 2026-08-24. This is the source of truth for the current codebase,
including the architecture explanation, operator flow, and acceptance status.
The former Responses-driven proposal runtime, generic environment abstraction,
primitive Isaac worker, mock agents, and historical planning documents were
removed after their retained dependencies were separated.

## Is this genuinely an intelligent agent?

Yes, within the precise meaning of a bounded tool-using agent. An OpenAI model
receives the research instruction, persistent session history, agent rules, and
seven tool schemas. The official Agents SDK `Runner` performs the model → tool
→ model loop. The host does **not** select a fixed tool sequence.

The model makes the open-ended decisions: what evidence to inspect, whether a
new experiment is warranted, which bounded parameters are informative, whether
to prepare or inspect a Reactor pair, and what hypothesis follows. Trusted
Python deliberately makes safety and evidence operations deterministic:
argument bounds, budget claims, the fixed Isaac container command, measurements,
media writes, indexing, and Plan C classification. This separation is what
makes the harness agentic without granting the model arbitrary machine access.

This is powered by the **OpenAI Agents SDK** (`openai-agents`), not a “Codex
SDK,” Codex CLI, LangGraph, CrewAI, or a hidden manual Responses loop.

## Exact agent definition and instructions

The agent is created by
`src/harness/agent_runtime/researcher.py:create_researcher` as:

```python
Agent(
    name="MineFailureResearcher",
    instructions=MINE_FAILURE_RESEARCHER_INSTRUCTIONS,
    model=model,
    model_settings=ModelSettings(tool_choice="required"),
    tools=AGENT_TOOLS,
    output_type=ResearchStepSummary,
)
```

Its exact instruction text is:

> You are MineFailureResearcher, one experiment-level researcher.
>
> Investigate environmental failure behavior in mine_v1 and identify interesting agreement or disagreement between Isaac/PhysX and Reactor.
>
> Rules:
> 1. Before claiming anything about the world, inspect the mine and/or existing experiment evidence with tools.
> 2. Never claim a physical outcome without an actual Isaac tool result.
> 3. Never claim a Reactor outcome without actual recorded Reactor evidence.
> 4. Isaac is a physics-grounded simulation reference, not real-world truth.
> 5. Reactor is neural-world visual evidence, not physical ground truth.
> 6. A disagreement is only a CANDIDATE DISCREPANCY.
> 7. Never fabricate experiments, files, frames, tool outputs, or measurements.
> 8. Prefer information-gaining experiments over arbitrary destruction and search near failure boundaries when possible.
> 9. In one top-level research step, execute at most one new Isaac experiment unless the user explicitly authorizes a separate multi-experiment mode.
> 10. Inspect previous experiments before selecting a new experiment. Do not repeat a parameter configuration without stating a scientific reason.
> 11. If Reactor evidence is needed but not captured, prepare the paired Reactor experiment and stop with status waiting_for_reactor.
> 12. When a paired recording is available on a later turn, inspect its status, assess and compare it with tools, then formulate the next hypothesis.
> 13. Do not reveal hidden chain-of-thought. Return only the structured concise hypothesis, evidence summary, and next-step rationale.
> 14. Treat tool errors and unavailable capabilities honestly. Never translate an authored or intended behavior into observed evidence.

## Exact end-to-end call graph

```text
Operator instruction (CLI or /agent)
  [scripts/run_agents_researcher.py:_main]
  [dashboard/app.py:run_agent_step]
       |
       v
Host constructs dependencies; it does not choose research actions
  [agent_runtime/service.py:MineFailureResearchService.run_step]
  [agent_runtime/context.py:AgentRuntimeContext]
       |
       +--> working conversation memory
       |    [agents.SQLiteSession]
       |    [runs/agent_sessions.sqlite3; session_id = campaign_id]
       |
       +--> authoritative scientific memory
       |    [persistence/store.py:ExperimentStore]
       |    [research/campaign.py:ResearchCampaignStore]
       |    [runs/experiments.sqlite3 + runs/ artifacts]
       |
       v
Single researcher
  [agent_runtime/researcher.py:create_researcher]
       |
       v
Official SDK-owned reasoning/tool loop
  [agents.Runner.run]
       |
       +--> model may inspect campaign
       |    [agent_runtime/tools.py:get_campaign_state]
       |
       +--> model may inspect verified world capabilities
       |    [agent_runtime/tools.py:inspect_mine_world]
       |    [assets/worlds/mine_v1/manifest.json]
       |
       +--> model may inspect compact prior evidence
       |    [agent_runtime/tools.py:get_recent_experiments]
       |    [persistence/store.py:ExperimentStore]
       |
       +--> model may request one real bounded Isaac run
       |    [agent_runtime/tools.py:run_mine_roof_support_experiment]
       |    -> bounds + per-turn budget [AgentRuntimeContext.claim_isaac_budget]
       |    -> fixed service [agent_runtime/isaac_service.py:MineIsaacToolService.run]
       |    -> fixed Docker command [/isaac-sim/python.sh]
       |    -> physical experiment [scripts/run_mine_rover_experiment.py]
       |    -> Nova Carter wheel targets + PhysX + measured poses + real frames
       |    -> artifact/index writes [ExperimentStore]
       |
       +--> model may prepare a human Reactor capture
       |    [agent_runtime/tools.py:prepare_reactor_comparison]
       |    [pairing.py:PairedCaptureService.prepare]
       |    -> actual Isaac seed + generated structured prompt + pending pair_id
       |    -> final status waiting_for_reactor
       |
       +--> operator records in browser
       |    [/reactor?pair_id=<pair_id>]
       |    [dashboard/static/reactor-live.js]
       |    [dashboard/app.py:finish_pair_capture]
       |    [pairing.py:PairedCaptureService.finalize]
       |    -> WebRTC video + Reactor record + initial paired record
       |
       +--> later process, same campaign_id/session database
            [MineFailureResearchService.run_step]
            [agents.SQLiteSession]
            -> model may call [get_pair_status]
            -> model may call [assess_and_compare_pair]
            -> real frame decode [tools.py:_sample_video_frames]
            -> visual assessment [research/visual_assessment.py]
            -> classification [comparison/plan_c.py:PlanCComparator.compare]
            -> persist comparison; Isaac evaluation remains unchanged
            -> model returns next hypothesis or selects a changed experiment
       |
       v
Typed operator result
  [agent_runtime/schemas.py:ResearchStepSummary]
```

The SDK call itself is:

```python
result = await Runner.run(
    agent,
    instruction,
    context=context,
    session=SQLiteSession(campaign_id, db_path=session_database_path),
    hooks=CampaignRunHooks(),
    max_turns=12,
    run_config=RunConfig(
        workflow_name="Physical AI Failure Research",
        trace_id=trace_id,
        group_id=campaign_id,
        trace_metadata={"campaign_id": campaign_id},
    ),
)
```

## Model-visible tools

Only these seven functions are exposed:

1. `get_campaign_state`
2. `inspect_mine_world`
3. `get_recent_experiments`
4. `run_mine_roof_support_experiment`
5. `prepare_reactor_comparison`
6. `get_pair_status`
7. `assess_and_compare_pair`

There is no shell, arbitrary Python, generic file writer, Codex tool,
unrestricted Docker, arbitrary path, or unrestricted Omniverse tool.

## Runtime modules retained after cleanup

- `agent_runtime/`: SDK agent, Runner service, tool context, hooks, typed output,
  and fixed Isaac subprocess service.
- `research/campaign.py`: scientific campaign/iteration/event metadata only.
- `research/world_prompt.py`: real OpenAI prompt generation for Reactor seeds.
- `research/visual_assessment.py`: real multimodal assessment of saved evidence.
- `persistence/`: experiment, pair, artifact, and review index.
- `pairing.py`: prepare/finalize the browser capture boundary.
- `comparison/plan_c.py`: data contracts and comparator; no deterministic
  coordinator remains.
- `media/`: real Isaac frame/video export and Reactor media normalization.
- `dashboard/`: existing evidence UI plus minimal agent controls.
- `mine_world.py`, `assets/worlds/mine_v1/`, and the two mine scripts: world
  validation and the physical experiment.

## Dashboard and data routes

- `/agent`: create/select a campaign, run/continue a step, view activity.
- `/reactor`: human browser/WebRTC capture.
- `/`: evidence library; `/experiments/<id>` and `/pairs/<id>` are detail views.
- `POST/GET /api/agent/campaigns`, `GET /api/agent/campaigns/<id>`,
  `POST .../step`, `POST .../continue`, and `GET .../events` expose the agent.
- `/api/pair-captures` prepares/finalizes capture; `/api/pairs` and
  `/api/experiments` expose indexed evidence.

`runs/` is intentionally untracked. The SQLite index stores metadata and paths;
large trajectories, frames, and videos stay as filesystem artifacts.

## Observability

`agent_runtime/hooks.py:CampaignRunHooks` records `agent_run_started`, model and
tool lifecycle events, completion/failure, call IDs, timestamps, trace ID, and
compact results in `campaign_events`. It never stores hidden reasoning. Agents
SDK tracing uses workflow `Physical AI Failure Research` and campaign grouping.

## Validation status

Post-cleanup evidence collected on 2026-08-24:

| Acceptance boundary | Result | Evidence |
|---|---|---|
| Lint | PASS | Ruff, all checks passed |
| Focused regression tests | PASS | 24 tests passed in 2.80 s |
| Real Agents SDK tool selection | PASS | Campaign `8909f753-fcd5-49d8-a48c-335fa58516c8`; model called `get_campaign_state`, `inspect_mine_world`, and `get_recent_experiments`; trace `trace_8964060f6f264b0e8c05594b3ce7c551` |
| Real Isaac mine physics smoke test | PASS | Isaac Sim 6.0.1; chassis `[24.0,-1.35,0.35]` → `[24.00495,3.48000,0.30000]`; support ΔY `2.384 m`; beam Z `3.240` → `0.400 m` |
| Agent-driven real Isaac experiment | PASS | Campaign `5f9198e2-9362-4ef1-869c-c38fb37305ba`; trace `trace_85ec803544bb4c0595ae4afeba8edd2e`; model selected 0.6 m/s, 300 steps; run `84f275f6-503c-4575-a88c-2d82844a41d7` |
| Scientific indexing/API | PASS | Run indexed in `ExperimentStore`; `/api/experiments/84f275f6-503c-4575-a88c-2d82844a41d7` returned 200 |
| Real Isaac recording | PASS | 21 real 320×180 PNG frames; first/last hashes differ; replay MP4 is 11,349 bytes and decodes successfully |
| Dashboard process/routes | PASS | Real Uvicorn process; `/agent`, agent campaign API, and `/reactor?pair_id=…` returned 200 |
| Ordinary process/session resume | PASS | A new CLI process reused campaign/session `8909f753-fcd5-49d8-a48c-335fa58516c8` and correctly recovered the earlier world conclusion and prior evidence |
| Real Reactor pair | NOT ACCEPTED | The permitted source `.env` has no `REACTOR_API_KEY`; no browser/WebRTC media was fabricated |
| Resume after a completed Reactor capture | NOT ACCEPTED | The loading/finalization/comparison code exists, but there is no completed real Reactor capture to resume from |
| Second evidence-driven research iteration | SKIPPED | It depends on the uncompleted Reactor comparison; no substitute run was claimed as this acceptance test |
| Negative tool boundaries | PASS | Direct validation rejects speeds outside 0.1–0.8 before Docker; a second budget claim is rejected; only seven tools exist; a real malicious-shell prompt returned `blocked` after campaign inspection and had no shell to call |

The browser handoff implementation was repaired during cleanup: an
agent-prepared pair now opens as `/reactor?pair_id=<id>`, the client fetches the
existing prepared record and seed, and finalizes that same pair. This is covered
by regression/API tests but remains live-unaccepted until a real Reactor key and
browser capture are available.

Repository provenance: the untouched fallback checkout remains at
`/home/ubuntu/physical-ai-failure-project`, SHA
`e4fcc213378e64ce93db3a245a7ed2a56a1081a6`. This cleaned checkout is
`/home/ubuntu/physical-ai-failure-agents-sdk` on `agents-sdk-harness`.
Installed live-test versions were `openai-agents 0.22.0`, `openai 3.3.1`, and
Isaac Sim container `nvcr.io/nvidia/isaac-sim:6.0.1`.

## Operator commands

Install and test:

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/pytest -q
```

Run the mine validator in the same container used by the tool:

```bash
docker run --rm --gpus all --network host \
  --user 0:0 --entrypoint bash \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e HOME=/tmp \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -v mine-rover-isaac-cache:/tmp/.cache \
  -v mine-rover-omniverse-data:/tmp/.nvidia-omniverse \
  -v "$PWD:/workspace/project" \
  nvcr.io/nvidia/isaac-sim:6.0.1 \
  -lc 'cd /workspace/project && /isaac-sim/python.sh \
    scripts/validate_mine_world.py --smoke-test --skip-render \
    --record-dir /workspace/project/runs/mine-world-validation'
```

The precise container flags used by the agent are assembled in
`agent_runtime/isaac_service.py`; inspect that module before changing the AWS
GPU/volume configuration.
