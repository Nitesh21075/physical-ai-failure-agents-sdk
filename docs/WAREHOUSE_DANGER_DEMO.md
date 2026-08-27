# Warehouse danger Agents SDK demo

## What the AI agent can do in Isaac Sim

The model does not receive Isaac Sim Python, a shell, Docker, or a generic USD
editor. The Agents SDK gives it seventeen typed tools. Nine are directly relevant
to Isaac scenario construction or evidence:

1. `list_scenario_capabilities` reports executable worlds, assets, controllers,
   sensors, parameter bounds, and explicit unavailable adapters.
2. `inspect_mine_world` retains the detailed `mine_v1` regression contract.
3. `get_recent_experiments` returns compact indexed scientific history.
4. `inspect_isaac_run` reopens measured physics, camera, stability, and visual
   readiness evidence for one indexed Isaac run.
5. `validate_scenario` rejects unknown or incompatible ScenarioSpec content.
6. `build_scenario` persists a normalized, digested execution request.
7. `run_scenario` executes one built request through the fixed Isaac boundary.
8. `inspect_simulation_checkpoint` reads initial or final synchronized state.
9. `compare_isaac_runs` compares measured evidence without rerunning Isaac.

Four additional tools validate, build, run, and inspect keyless IRO synthetic
scenes; they are not physical warehouse experiments. The remaining tools inspect
campaign state and manage the real Reactor pairing, status, visual assessment,
and comparison. One ordinary agent step may claim at most one new Isaac run. The
model cannot teleport Nova Carter or invoke a collapse; it selects a bounded
typed scenario and the fixed runner applies run-owned session overrides plus
real differential wheel-velocity targets.

## Scenario composition

`assets/worlds/warehouse_danger_v1/warehouse_world.usda` composes:

- NVIDIA's native Isaac Sim 6.0 `warehouse_with_forklifts.usd`;
- a local collision-backed danger cell and industrial backdrop;
- a low-friction movable rack support;
- a dynamic loaded crossbeam and attached cargo;
- warning geometry and neutral/warning lighting;
- Nova Carter, selected by the run-owned derived layer with physics enabled and
  unused vendor sensors disabled;
- ego, tracking, and fixed witness RTX evidence cameras.

The stock world is translated away from the bounded failure cell so its dense
collision layout cannot overlap the initial robot/hazard state. It remains the
native visual environment and background. The local cell provides the physics
surface used by the experiment.

## Evidence gates

An Isaac run is not considered usable merely because the process exits. The
runner enforces or records:

- pre-actuation stability: rover, support, and loaded beam must remain within
  manifest tolerances after zero-command settling;
- physical causality: Nova Carter moves through articulation wheel targets;
- failure measurement: support displacement and loaded-beam vertical drop;
- three synchronized RGB roles: ego, smoothed tracking, and fixed witness;
- exposure, contrast, dynamic-range, and gradient checks;
- required rack-support and loaded-beam semantic visibility;
- a bounded tracking-camera-to-rover geometry check;
- immutable source-stage and run-owned session-layer provenance.

`PairedCaptureService` refuses to prepare Reactor from a missing or rejected
Isaac seed.

## Current acceptance status

Live Isaac Sim 6.0.1 validation on 2026-08-25 established the collapse side and
the runner's stability/visual safeguards:

- `warehouse-danger-safe-spawn-physics-20260825` passed the pre-actuation
  stability gate at 0.25 m/s for 180 steps and did not collapse under the
  earlier, wider support geometry;
- `warehouse-danger-final-20260825` produced 11 frames from every camera and
  passed the automated exposure, content, semantic, and tracking-geometry
  gates, but remained stable;
- after narrowing the support interface,
  `warehouse-danger-collapse-physics-20260825` again passed stability, moved the
  support 2.918 m, and dropped the loaded beam 2.937 m at 0.8 m/s for 450 steps.
- the final combined run `warehouse-danger-collapse-camera-20260825` passed the
  stability and automated visual gates, captured 31 synchronized frames per
  role, moved the support 2.915 m, and dropped the loaded beam 2.936 m.

Human inspection of that combined run found the witness collapse legible but
not presentation-ready: the danger annex is overlit and visually sparse, and
the composed native warehouse is largely outside the shot. Art-direction
acceptance, a stable-side point using the same final geometry, and live model
selection of the new warehouse tool are therefore separate final rehearsal
boundaries. Do not describe the demo as fully accepted until they pass. The
environment is also not offline-ready until its NVIDIA references are collected
locally.

## Run through the agent

```bash
.venv/bin/python scripts/run_agents_researcher.py \
  'Create a warehouse danger demonstration. Inspect Isaac capabilities and prior evidence, run at most one information-gaining rack-collapse experiment, and prepare Reactor only if the Isaac visual gate passes.' \
  --experiment-budget 2
```

Keep the printed campaign ID. If the agent prepares a pair, start the dashboard,
open the returned `/reactor?pair_id=...` route, finish the real browser capture,
then resume the same campaign in a new process for assessment and comparison.

## Mandatory pre-recording sequence

1. Install the Isaac Sim 6.0 Local Assets Pack or collect the warehouse and Nova
   references locally, change them to relative references, and verify with
   networking disabled. The current stage is native USD but not yet offline.
2. Stop unrelated Isaac Sim processes. Rehearse on the same GPU/CPU allocation
   that will be used for recording.
3. Run one physics-only acceptance and require the pre-actuation stability gate
   to pass. The current accepted collapse-side point is 0.8 m/s for 450 steps.
   Re-measure a stable-side point after the final geometry changes; do not reuse
   the older 0.25 m/s result as though it came from the final geometry.
4. Run one camera acceptance and require all three camera streams plus the
   complete visual-evidence gate to pass. Inspect first/middle/final frames by
   eye; pixel gates are necessary but not sufficient. The present witness view
   fails this human presentation check, so revise lighting/framing and integrate
   more of the native warehouse background before recording.
5. Run the exact chosen speed/step configuration twice. Confirm the intended
   outcome is repeatable and select one stable and one collapse-side parameter
   point if the demo explains a failure boundary.
6. Export the tracking and witness replays and check framing, lighting, text/UI
   legibility, and frame cadence at presentation resolution.
7. Start a fresh campaign with enough experiment budget, verify the model name
   and API key, and use the final operator instruction verbatim in a rehearsal.
8. Prepare and complete one Reactor pair, resume the same campaign, and verify
   the comparison and human-review UI before the recording day.
9. Back up `runs/experiments.sqlite3`, `runs/agent_sessions.sqlite3`, and the
   accepted artifact directories. Do not rely on a single live take.

## Useful next capabilities after the first demo

The next high-value Isaac additions are a bounded camera-only preflight (no
experiment budget), allowlisted start-pose/camera presets, a second independent
warehouse hazard, near-boundary parameter sweeps under explicit multi-run
authorization, contact-force/time-series summaries, and an offline asset
dependency audit. Arbitrary Isaac Python should remain outside the model tool
surface.
