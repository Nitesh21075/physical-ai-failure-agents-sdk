# MineFailureResearcher: authoritative runtime context

Last updated: 2026-08-27. This is the source of truth for the current codebase,
including the architecture explanation, operator flow, and acceptance status.
The former Responses-driven proposal runtime, generic environment abstraction,
primitive Isaac worker, mock agents, and historical planning documents were
removed after their retained dependencies were separated.

## Is this genuinely an intelligent agent?

Yes, within the precise meaning of a bounded tool-using agent. An OpenAI model
receives the research instruction, persistent session history, agent rules, and
seventeen tool schemas. The official Agents SDK `Runner` performs the model → tool
→ model loop. The host does **not** select a fixed tool sequence.

The model makes the open-ended decisions: what evidence to inspect, which
typed scenario to validate and build, whether a new experiment is warranted,
which bounded parameters are informative, whether to prepare or inspect a
Reactor pair, and what hypothesis follows. Trusted Python deliberately owns
schema and compatibility validation, scenario digests, budget claims, the
fixed Isaac container command, run-owned USD/session changes, measurements,
media writes, indexing, and Plan C classification.

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
> Investigate environmental failure behavior in the allowlisted Isaac worlds and identify interesting agreement or disagreement between Isaac/PhysX and Reactor. The executable typed scenario catalog includes mine_v1, the bounded-route mine_v2_subt world, and warehouse_danger_v1.
>
> Rules:
> 1. Before claiming anything about Isaac capabilities or a world, call list_scenario_capabilities and inspect relevant existing experiment evidence. Treat unavailable catalog entries as unavailable even if Isaac supports them generally.
> 2. Never claim a physical outcome without an actual Isaac tool result.
> 3. Never claim a Reactor outcome without actual recorded Reactor evidence.
> 4. Isaac is a physics-grounded simulation reference, not real-world truth.
> 5. Reactor is neural-world visual evidence, not physical ground truth.
> 6. A disagreement is only a CANDIDATE DISCREPANCY.
> 7. Never fabricate experiments, files, frames, tool outputs, or measurements.
> 8. Prefer information-gaining experiments over arbitrary destruction and search near failure boundaries when possible. Use the validated ScenarioSpec workflow: validate_scenario, build_scenario, then run_scenario. Source worlds are immutable; all overrides and approved props belong to a run-owned session layer.
> 9. In one top-level research step, execute at most one new Isaac experiment unless the user explicitly authorizes a separate multi-experiment mode.
> 10. Inspect previous experiments before selecting a new experiment. Do not repeat a parameter configuration without stating a scientific reason.
> 11. If Reactor evidence is needed but not captured, prepare the paired Reactor experiment and stop with status waiting_for_reactor.
> 12. When a paired recording is available on a later turn, inspect its status, assess and compare it with tools, then formulate the next hypothesis.
> 13. Do not reveal hidden chain-of-thought. Return only the structured concise hypothesis, evidence summary, and next-step rationale.
> 14. Treat tool errors and unavailable capabilities honestly. Never translate an authored or intended behavior into observed evidence.
> 15. Treat offline asset readiness separately from physics readiness. A native remote USD that has not been collected locally is not offline-ready.
> 16. Treat an automated pixel/semantic gate separately from human presentation readiness. If the capability catalog says art-direction acceptance is pending, do not call that world recording-ready.
> 17. The fixed_velocity controller owns simulator-rate wheel commands. Do not claim navigation, VLA, RL, or ROS control unless the capability catalog reports a real registered adapter.
> 18. Use inspect_simulation_checkpoint and compare_isaac_runs for measured evidence. A built or authored scenario is not an observed physical outcome until run_scenario succeeds.
> 19. For bounded synthetic-data scene generation, use validate_iro_scene, build_iro_scene, then run_iro_scene. IRO scenes consume the same Isaac-run budget. Treat their images and annotations as synthetic scene evidence, never as a robot run, structural-failure result, or real-world ground truth.

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
       +--> model may inspect verified Isaac/world capabilities
       |    [agent_runtime/tools.py:list_scenario_capabilities]
       |    [agent_runtime/tools.py:inspect_mine_world]
       |    [assets/worlds/*/manifest.json]
       |
       +--> model may inspect compact prior evidence
       |    [agent_runtime/tools.py:get_recent_experiments]
       |    [agent_runtime/tools.py:inspect_isaac_run]
       |    [persistence/store.py:ExperimentStore]
       |
       +--> model may validate, build, then request one real bounded Isaac run
       |    [agent_runtime/tools.py:validate_scenario]
       |    [agent_runtime/tools.py:build_scenario]
       |    [runs/scenario-builds/<scenario_id>/scenario_spec.json]
       |    [agent_runtime/tools.py:run_scenario]
       |    -> bounds + per-turn budget [AgentRuntimeContext.claim_isaac_budget]
       |    -> fixed service [agent_runtime/isaac_service.py:MineIsaacToolService.run_scenario]
       |    -> fixed Docker command [/isaac-sim/python.sh]
       |    -> physical experiment [scripts/run_mine_rover_experiment.py]
       |    -> run-owned USD/session overrides + Nova wheel targets + PhysX
       |    -> artifact/index writes [ExperimentStore]
       |
       +--> model may inspect checkpoints or compare prior Isaac runs
       |    [agent_runtime/tools.py:inspect_simulation_checkpoint]
       |    [agent_runtime/tools.py:compare_isaac_runs]
       |
       +--> model may validate, build, and run one bounded synthetic IRO scene
       |    [agent_runtime/tools.py:validate_iro_scene]
       |    [agent_runtime/tools.py:build_iro_scene]
       |    [runs/iro-builds/<scene_id>/{build.json,iro_config.yaml}]
       |    [agent_runtime/tools.py:run_iro_scene]
       |    -> same per-turn budget [AgentRuntimeContext.claim_isaac_budget]
       |    -> fixed service [agent_runtime/iro_service.py:IsaacIROToolService]
       |    -> dedicated installed headless Kit experience
       |    -> keyless [isaacsim.replicator.object.core]
       |    -> RGB/annotation/USD outputs + artifact/index writes
       |    [agent_runtime/tools.py:inspect_iro_run]
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

Only these seventeen functions are exposed:

1. `get_campaign_state`
2. `list_scenario_capabilities`
3. `inspect_mine_world`
4. `get_recent_experiments`
5. `inspect_isaac_run`
6. `inspect_iro_run`
7. `validate_scenario`
8. `build_scenario`
9. `run_scenario`
10. `validate_iro_scene`
11. `build_iro_scene`
12. `run_iro_scene`
13. `inspect_simulation_checkpoint`
14. `compare_isaac_runs`
15. `prepare_reactor_comparison`
16. `get_pair_status`
17. `assess_and_compare_pair`

There is no shell, arbitrary Python, generic file writer, Codex tool,
unrestricted Docker, arbitrary path, or unrestricted Omniverse tool.

## Runtime modules retained after cleanup

- `agent_runtime/`: SDK agent, Runner service, tool context, hooks, typed output,
  typed ScenarioSpec/IROSceneSpec compiler stores, and fixed Isaac subprocess
  services.
- `research/campaign.py`: scientific campaign/iteration/event metadata only.
- `research/world_prompt.py`: real OpenAI prompt generation for Reactor seeds.
- `research/visual_assessment.py`: real multimodal assessment of saved evidence.
- `persistence/`: experiment, pair, artifact, and review index.
- `pairing.py`: prepare/finalize the browser capture boundary.
- `comparison/plan_c.py`: data contracts and comparator; no deterministic
  coordinator remains.
- `media/`: real Isaac frame/video export and Reactor media normalization.
- `dashboard/`: existing evidence UI plus minimal agent controls.
- `mine_world.py`, `assets/worlds/mine_v1/`,
  `assets/worlds/warehouse_danger_v1/`, `assets/worlds/mine_v2_subt/`, and the
  experiment scripts: deterministic regression world, native warehouse danger
  demo, separate SubT perception prototype, validation, and physical runs.

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
The CLI prints only lifecycle and tool names while a step is running, so a long
Isaac call is visible without exposing model reasoning or large tool payloads.
Completed runs also persist an `agent_run_usage` event with SDK request and
token counts.

## Validation status

Post-cleanup evidence collected from 2026-08-24 through 2026-08-27:

| Acceptance boundary | Result | Evidence |
|---|---|---|
| Lint | PASS | Ruff, all checks passed |
| Focused regression tests | PASS | 37 tests passed in 3.31 s on 2026-08-27. |
| Typed ScenarioSpec and Agents SDK schemas | PASS | All seventeen tools construct under `openai-agents 0.22.0`; unit tests cover world/hazard compatibility, placement regions, parameter bounds, build persistence, digest tamper detection, the one-run guard, and absence of shell/Python/filesystem tools. |
| ScenarioSpec live Isaac physics execution | PASS | Run `6436fab1-dc15-45cb-ba96-debab712f15c` loaded digest `25d897…6650`, authored session-only support/falling-body masses, support friction, support/start offsets, and one static collision prop, applied asymmetric wheel targets from a 0.1 rad/s steering command, passed the stability gate, preserved the source stage, and produced indexed measured state. The 60-step physics-only run remained stable. |
| ScenarioSpec RTX presets and dynamic approved props | PENDING | Historical default RTX routes remain accepted, but the new preferred-role/resolution/lighting combinations and dynamic-prop path have not yet received a dedicated live run. Do not generalize the physics-only acceptance to these combinations. |
| Typed IROSceneSpec and compiler | PASS | Unit coverage rejects arbitrary fields/paths, out-of-room distributions, excessive object workloads, and compiled-config tampering. The compiler emits only the installed IRO `0.9.12` contract and disables caption/hosted-model outputs. |
| Keyless IRO core live execution | PASS | Isaac Sim 6.0.1 run `d8c7717a-1ac0-48e1-95c9-bf3c4170257c`, scene `917920d9-d61b-4ce8-9d89-1177a0ffa186`, used installed `isaacsim.replicator.object.core 0.11.12` with Docker networking disabled and no API key. It produced one real 320×256 RGB image, one 2D-label file containing both rigid cubes, two semantic-segmentation artifacts, and one resolved description. Human review of the equivalent preceding run confirmed a rendered walled room and object. |
| Generic full-app IRO launch | FAIL, FIXED | Initial run `8dea5068-d2c9-4e29-b1be-a93ecdf18d99` combined the generic full GUI experience with `--no-window` and remained in `await_viewport`; it was stopped without evidence. The fixed service now launches the installed `isaacsim.exp.action_and_event_data_generation.base.kit` experience. |
| IRO convenience-room helper | FAIL, FIXED | Run `2df42f96-b4f4-41dc-a00d-00842793091d` reached IRO but failed because the headless base experience omits `omni.physxdemos`, which the `room` helper imports. The trusted compiler now creates floor/walls from ordinary collision cubes; the accepted live run verifies this route. |
| Real Agents SDK tool selection | PASS | Campaign `8909f753-fcd5-49d8-a48c-335fa58516c8`; model called `get_campaign_state`, `inspect_mine_world`, and `get_recent_experiments`; trace `trace_8964060f6f264b0e8c05594b3ce7c551` |
| Real Isaac mine physics smoke test | PASS | Isaac Sim 6.0.1; chassis `[24.0,-1.35,0.35]` → `[24.00495,3.48000,0.30000]`; support ΔY `2.384 m`; beam Z `3.240` → `0.400 m` |
| Agent-driven real Isaac experiment | PASS | Campaign `5f9198e2-9362-4ef1-869c-c38fb37305ba`; trace `trace_85ec803544bb4c0595ae4afeba8edd2e`; model selected 0.6 m/s, 300 steps; run `84f275f6-503c-4575-a88c-2d82844a41d7` |
| Scientific indexing/API | PASS | Run indexed in `ExperimentStore`; `/api/experiments/84f275f6-503c-4575-a88c-2d82844a41d7` returned 200 |
| Historical Isaac recording transport | PASS, SEMANTIC QUALITY FAIL | 21 real 320×180 PNG frames decoded and differed, but the later visual audit found only a near-uniform brown field with no identifiable mine or rover. This evidence must not be reused as a Reactor seed. |
| Dashboard process/routes | PASS | Real Uvicorn process; `/agent`, agent campaign API, and `/reactor?pair_id=…` returned 200 |
| Ordinary process/session resume | PASS | A new CLI process reused campaign/session `8909f753-fcd5-49d8-a48c-335fa58516c8` and correctly recovered the earlier world conclusion and prior evidence |
| Real Reactor pair | PASS | Campaign `26d90a6c-f2d3-41f6-9a45-73798fe38c4c`; pair `b67ac8dd-62b5-4ba5-a532-16c913bc0ca3`; real browser/WebRTC run `3066ccb8-7019-4b31-b505-c9dada522729`; 2,432,837-byte WebM decodes at 1664×960 |
| Resume after a completed Reactor capture | PASS | A new CLI process reused the same campaign/SQLite session; trace `trace_15363940233b4d57b0fb2b860e50e57d` shows `get_pair_status` then `assess_and_compare_pair`; result was honestly `inconclusive`/`needs_human_review` |
| Second evidence-driven research iteration | SKIPPED | The cost-bounded E2E campaign authorized exactly one experiment. The resumed agent proposed a different 600-step experiment based on the evidence, but the exhausted budget correctly prevented execution |
| Historical negative tool boundaries | PASS | The pre-warehouse seven-tool acceptance rejected invalid speed, a second budget claim, and a real malicious-shell prompt. The current seventeen-tool surface remains bounded and has unit coverage; the live malicious-prompt acceptance has not been repeated after adding ScenarioSpec/IROSceneSpec tools. |
| `mine_v2_subt` bounded physics | PASS | Real Isaac Sim 6.0.1 run `mine-v2-subt-physics-20260824`; Nova wheel targets moved Y −1.650 → −0.841 m at 0.25 m/s over 180 steps; support displacement 0.00875 m and effectively zero beam drop established a stable outcome |
| `mine_v2_subt` RTX/camera | PASS | Real run `mine-v2-subt-camera-lit-20260824`; synchronized ego/tracking/witness frames, support/beam semantic visibility, bounded tracking geometry, and stricter exposure/content gate all passed; tracking mean luminance 35.35/255 |
| Native warehouse danger stage and stability gate | PASS | Isaac Sim 6.0.1 composed NVIDIA's native warehouse USD with the local rack-collapse annex. Run `warehouse-danger-safe-spawn-physics-20260825` passed the stability gate and remained stable at 0.25 m/s for 180 steps under the earlier, wider support geometry. A stable-side point for the final geometry remains pending. |
| Warehouse rack-collapse physics | PASS | Run `warehouse-danger-collapse-physics-20260825` passed the pre-actuation gate; at 0.8 m/s for 450 steps the support moved 2.918 m and the loaded beam dropped 2.937 m, exceeding the 0.8 m collapse threshold. |
| Warehouse RTX/camera before final geometry revision | PASS | Run `warehouse-danger-final-20260825` produced 11 frames for each of ego/tracking/witness and passed exposure, content, semantics, and tracking-geometry gates. It remained stable; the preferred Reactor seed has since been changed to the clearer witness role and the support geometry narrowed. |
| Revised warehouse collapse plus cameras | TECHNICAL PASS, PRESENTATION FAIL | Combined run `warehouse-danger-collapse-camera-20260825` passed pre-actuation stability and the automated visual gate, captured 31 synchronized frames per role, and measured a 2.936 m beam drop. Human frame review found the witness view overlit and visually sparse with little of the native warehouse visible; it is not yet a worthwhile recording shot. |
| Live Agents SDK selection of generic ScenarioSpec/IROSceneSpec routes | PENDING | The seventeen-tool surface has unit coverage and both executors have direct live acceptance, but a real model-driven `validate` → `build` → `run` trace on either new route has not yet been accepted. |

`mine_v2_subt` is a distinct hybrid stage, not a replacement for `mine_v1`.
It vendors a hash-pinned MIT-licensed LTU cave conversion, reuses the existing
Nova Carter and roof-support USD layers, authors RTX lighting independently of
Gazebo, and provides a simplified collider only for the bounded experiment
route. The full cave mesh is visual-only, so wider traversal is not yet a
physics-backed capability. The typed ScenarioSpec now exposes only that accepted
bounded route; it does not claim full-cave collision or general navigation.

## Typed ScenarioSpec boundary

`agent_runtime/scenario_spec.py` is the model-facing experiment language. It
currently exposes three allowlisted worlds, their compatible roof-support or
rack-collapse template, Nova Carter's `fixed_velocity` differential-wheel
controller through a registered simulator-rate adapter, bounded linear/angular
commands, bounded robot/support offsets,
support and falling-body masses, support friction, three camera presets, three
lighting presets, and up to four allowlisted collision-enabled box props inside
world-specific placement regions. Unknown fields and incompatible combinations
are rejected. Repeat count is fixed to one.

`build_scenario` stores a normalized specification and SHA-256 digest below
`runs/scenario-builds/`. `run_scenario` accepts only its UUID, reloads and
revalidates the spec and digest, applies the ordinary campaign/per-turn budget,
and passes the fixed file into the Isaac worker. The worker checks the selected
world and hazard again, creates a run-owned derived stage and session layer, and
never saves the source world. Navigation, VLA, RL, ROS, weather, arbitrary USD
paths, arbitrary assets, arbitrary YAML, Python, shell, and Docker arguments are
not exposed. The capability catalog reports the controller adapters as
unavailable rather than suggesting that general Isaac support makes them usable.

## Keyless IRO boundary

Chat IRO and IRO core are different layers. NVIDIA's Chat IRO UI uses a hosted
language model and requires an NVIDIA API key. It is an Isaac Sim extension/UI,
not a documented remote scene-generation API for this harness. The installed
`isaacsim.replicator.object.core 0.11.12` extension needs no NVIDIA API key: it
executes a local scene-description YAML contract (`version: 0.9.12`) inside
Isaac Sim. The harness uses the latter directly and supplies its own Agents SDK
natural-language planning layer.

`agent_runtime/iro_spec.py` exposes a strict `IROSceneSpec` for one
`primitive_room` template: up to six groups and 32 total built-in cubes,
spheres, cylinders, cones, or tori; bounded position/color distributions;
collision or rigid-body physics; size, friction, initial velocity, gravity and
frame timing; bounded resolution/frame count; dome lighting; and selected RGB,
box, segmentation, depth, normal, description, or USD products. Unknown fields,
arbitrary YAML expressions, USD/filesystem paths, hosted captions, assets,
Python, shell, Docker arguments, and general Omniverse commands are rejected.

`build_iro_scene` stores the normalized spec, SHA-256 spec/config digests, build
manifest, and JSON-form YAML under `runs/iro-builds/`. Load verifies both
digests and recompiles the expected config byte-for-byte. `run_iro_scene`
accepts only the build UUID and uses the same one-Isaac-run-per-step/campaign
budget as physical experiments. `IsaacIROToolService` writes a run-owned config
with a fixed output path, launches the dedicated installed headless Action and
Event Data Generation Base Kit experience, requires the requested RGB evidence,
indexes artifacts, and labels the result `isaac_iro`. These outputs are useful
for perception variations and synthetic data. They are not evidence that a
robot acted or that a structural failure occurred.

This gives the agent much of Chat IRO's useful *bounded scene-authoring result*
without its NVIDIA-hosted language layer: the OpenAI agent converts the user's
intent into `IROSceneSpec`, trusted code validates and compiles it, and local
  IRO executes it with Docker networking disabled. It does not attempt feature parity with unrestricted Chat IRO;
new templates, approved assets, compound objects, or event/force primitives
must be deliberately added to the schema and receive live acceptance.

## Durable handoff and next work

This section must be updated with every material harness addition so a new
session can resume without reconstructing architecture decisions.

Current branch and runtime state on 2026-08-31:

- Checkout: `/home/ubuntu/physical-ai-failure-agents-sdk`; branch:
  `agents-sdk-harness`; push target: `origin/agents-sdk-harness`.
- Physical ScenarioSpec, keyless IROSceneSpec, seventeen Agents SDK tools, one
  shared Isaac budget, digest-checked build stores, and experiment/artifact
  indexing are implemented.
- Accepted physical spec run: `6436fab1-dc15-45cb-ba96-debab712f15c`.
- Accepted network-isolated keyless IRO run:
  `d8c7717a-1ac0-48e1-95c9-bf3c4170257c`.
- The first controller-registry cycle is accepted. The `fixed_velocity` adapter
  owns command generation and emits a per-step trace plus normalized termination
  evidence. Live run `eda39916-9959-40b3-98fd-24af2fa7a73d` executed 60 commands
  with the expected asymmetric 1.78/2.22 rad/s targets and measured rover motion.
- Generated evidence and SQLite state remain below ignored `runs/`; source,
  tests, and this handoff are committed. Never commit runs, media, caches, `.env`,
  or keys.
- Remaining acceptance gaps: a model-selected validate/build/run trace for the
  generic physical and IRO routes; physical RTX preset/dynamic-prop coverage;
  and the post-camera-fix Reactor comparison.

The next planned addition is a bounded `goal_pose` registry entry. Preserve the
separation between experiment-level agent decisions and simulator-rate control:

1. Extend the typed `ControllerSpec` to a discriminated union and keep registry
   capability entries explicit about accepted and unavailable adapters.
2. Keep `fixed_velocity` as the accepted baseline adapter. Add bounded goal pose
   next (target, timeout, tolerances, and stuck termination),
   then register VLA, RL-policy, and ROS adapters only when their packages,
   model/topic contracts, observation/action rates, and live acceptance exist.
3. The research agent selects controller ID, objective, checkpoints, and stop
   conditions. The adapter—not the LLM tool loop—owns high-rate wheel/joint/
   torque commands and safety interlocks.
4. Normalize all adapters into a common evidence contract: controller identity
   and digest, goals, observations, actions, contacts, safety events, termination
   reason, timing/rate statistics, and replay references.
5. Add low-frequency `observe_checkpoint` / bounded intervention only after the
   autonomous controller path is accepted. Never stream simulator-rate control
   through ordinary model tool calls.

Before further controller work, rerun `.venv/bin/ruff check .` and
`.venv/bin/pytest -q`, inspect this document plus `AGENTS.md`, and verify the
installed Isaac APIs for the exact 6.0.1 image rather than assuming a newer
Isaac Lab/ROS contract.

The completed live E2E exposed and fixed two browser timing defects: the page
now waits until the Reactor SDK reports transport status `ready` before upload,
then waits for `conditions_ready` before generation. It also exposed a stale
Plan C assumption that confused the world identifier (`mine_v1`) with the
executor backend (`isaac_sim`); those are now validated at their correct schema
boundaries. The accepted run and encountered failures are recorded in
[`runs/e2e-agents-sdk/20260824T090809Z/E2E_REPORT.md`](../runs/e2e-agents-sdk/20260824T090809Z/E2E_REPORT.md).

The historical pipeline completed, but its scientific comparison remains
inconclusive. A later pixel and human visual audit established that the original
single `/World/Robot/VLACamera` stream did not visibly contain the mine or rover;
therefore the historical recording transport passed while its semantic visual
acceptance failed. The current runner records synchronized ego, smoothed chase,
and fixed witness streams. It records camera poses and measured image-quality /
object-visibility gates, and `PairedCaptureService` refuses blank or rejected
seeds. See `docs/MINE_ROVER_EXPERIMENT.md` for the current camera contract.

Real camera diagnostics on 2026-08-24 established the cause and correction:
the authored fixed camera and the first chase offset were outside the opaque
mine wall, while the original ego camera was effectively against geometry. Run
`camera-diagnostic-20260824T143533Z` visibly shows Nova Carter in the chase view,
the support/beam bay in the witness view, and real mine lighting. Isaac 6.0.1
emits support/beam semantic boxes but not the authored label on NVIDIA's
instanced Nova visual; the runner records a bounded measured-articulation chase
check for that one vendor asset instead of fabricating a semantic result.

The post-fix real Agents SDK acceptance campaign is
`268d9f9b-2227-4ccc-bc8c-028b4fb4cc07`. Trace
`trace_c512d9901d1c40598143662277ad254a` shows the model selecting
`inspect_mine_world`, `get_recent_experiments`, `get_campaign_state`, and then
`run_mine_roof_support_experiment`; the host did not pre-call Isaac. The model
selected 0.8 m/s, 900 control steps, and seed 42. Real run
`47827760-a92b-4c7e-9689-82d4a0f6cafd` produced 61 frames per camera, three
decodable nonzero MP4s, distinct first/middle/last tracking frames, and a passing
visual evidence gate. The measured result was stable: rover translation about
0.993 m, support displacement 0.1915 m, beam displacement 0.1915 m, and beam
vertical drop 0.0000196 m (below the 0.8 m collapse criterion).

The first pair-preparation attempt exposed a container-to-host seed-path mapping
bug and was honestly returned as failed. After repairing that bounded path
translation, a new process reused the same campaign and SQLite SDK session.
Trace `trace_010c4014b76f4d5eb9eef8b71c53cbf2` shows the resumed model calling only
`prepare_reactor_comparison`; it did not launch another simulation. Pending pair
`ec4106a3-d5b9-4d64-937c-90a14d8439fc` now uses the real tracking PNG and is
`WAITING_FOR_REACTOR_CAPTURE`. A new Reactor result has not yet been recorded,
so no post-camera-fix Isaac/Reactor agreement claim exists yet.

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
