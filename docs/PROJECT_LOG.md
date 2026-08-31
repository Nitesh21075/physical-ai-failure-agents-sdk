# Project log

This is the append-only, cross-session chronology for material project cycles.
All times are UTC. `docs/AGENTS_SDK_RUNTIME.md` remains the curated source of
current truth; this log preserves what was tried and why it was kept, revised,
or removed.

## 2026-08-27 13:23:47 — Typed scenario and keyless IRO cycle

- Hypothesis: a strict ScenarioSpec and local IRO compiler can give the agent
  useful bounded Isaac authoring without arbitrary Python, USD paths, or an
  NVIDIA API key.
- Evidence: 37 unit tests and Ruff passed; physical run
  `6436fab1-dc15-45cb-ba96-debab712f15c` and network-isolated IRO run
  `d8c7717a-1ac0-48e1-95c9-bf3c4170257c` passed their stated live criteria.
- Decision: kept. Dedicated RTX/dynamic-prop and model-selected generic-route
  acceptance remained pending.
- Git: `23a1a3a` (`Add typed Isaac scenario and keyless IRO tools`), pushed to
  `origin/agents-sdk-harness`.

## 2026-08-31 09:09:04–09:19:31 — Cycle 1: fixed-velocity registry baseline

- Hypothesis: routing the accepted fixed command through a small controller
  registry will preserve Isaac behavior while creating a safe adapter boundary.
- Change: added a pure controller adapter contract, a fixed-velocity registry
  entry, differential-drive conversion, per-step trace, and normalized
  controller termination evidence. The Agents SDK tool/schema surface did not
  expand in this cycle.
- Tests: focused Ruff passed; 26 focused unit tests passed.
- Live evidence: Isaac run `eda39916-9959-40b3-98fd-24af2fa7a73d` passed its
  stability gate, executed exactly 60 registry commands, ended with
  `control_steps_completed`, applied 1.78/2.22 rad/s left/right targets, and
  moved the rover from `[24.0, -1.6500864, ~0]` to
  `[23.9883709, -1.3813437, ~0]`.
- Decision: keep. This establishes the registry baseline; it does not establish
  goal navigation or model selection.
- Git: `4c6c027` (`Add accepted fixed controller registry`), pushed to
  `origin/agents-sdk-harness`.

## 2026-08-31 09:20:00–09:52:13 — Cycle 2: bounded goal pose and model use

- Hypothesis: a closed-loop, registry-owned goal-pose adapter can reach a
  bounded target from measured Isaac pose, and the Agents SDK researcher can
  discover and use both registered adapters without simulator-rate LLM control.
- Change: added the discriminated `fixed_velocity | goal_pose` ScenarioSpec,
  bounded target/tolerances/speeds/timeouts, proportional differential-drive
  steering, stagnation detection, capability reporting, and common controller
  evidence. Fixed-length model inputs now use strict-length JSON arrays that
  remain valid OpenAI function schemas.
- Support checks: Ruff passed; 41 tests passed, including every model tool schema
  being free of array definitions without `items`.
- Direct real output: Isaac run `fc19f0c3-c18a-42d9-9e6a-68b9a28bb692`
  reached `[24.0, -0.9]` after 141 steps with measured error `0.0989733 m`,
  terminating `goal_reached` inside the `0.10 m` tolerance.
- Direct full-pose output: run `1a5e53f0-ac78-4bf2-a5af-5a2b77144a70`
  reached the same XY target plus heading `1.0 rad` after 183 steps, with
  `0.0942914 m` position error and `-0.147733 rad` heading error inside the
  requested `0.10 m`/`0.15 rad` tolerances.
- Failure and revision: the first real model campaign was rejected by OpenAI
  before tool execution because tuple-derived function schemas used
  `prefixItems` but omitted `items`. Local tests had passed and did not establish
  API acceptance. ScenarioSpec and IROSceneSpec vector fields were revised to
  strict-length lists, all 17 schemas were audited, and the same campaign path
  was rerun.
- Model-selected real outputs: GPT-5.6 Luna campaign
  `ccb7258c-2965-4724-bf90-f97cafb68fc3` discovered capabilities, inspected
  prior evidence, validated, built, and ran goal pose; Isaac run
  `1b44e66a-41b5-4188-a116-03244cee27f9` reached `[24.0, -0.75]` in 168 steps
  with `0.0992608 m` error, then the model inspected the final checkpoint and
  reported the result accurately. Campaign
  `2d4a333d-3eb8-4888-b3ad-5a59cf079242` used the same tool sequence for
  `fixed_velocity`; run `835761ff-21b9-4433-a03e-3e01601480c2` executed exactly
  60 steps, moved `0.26899 m`, terminated `control_steps_completed`, and was
  accurately reported as stable.
- Decision: keep both adapters and the schema correction. This accepts bounded
  pose reaching, not general path planning, obstacle avoidance, or VLA/ROS/RL
  control.
- Git: recorded by the cycle-2 checkpoint containing this entry.

## 2026-08-31 10:42:27 — Planned experiment: model-directed USD authoring v1

- Status: **PLANNED, NOT YET IMPLEMENTED OR RUN**.
- Branch: `experiment/model-directed-usd-authoring-v1`, created from accepted
  harness commit `2a20116`. The accepted `agents-sdk-harness` branch remains the
  control baseline.
- Hypothesis: a more capable planning model, initially Terra with Luna retained
  as a comparison baseline, can direct sufficiently reliable creation and
  modification of novel physical structures when edits are confined to an
  experiment-owned warehouse copy or run-owned USD layer.
- Proposed first task: create a supported platform with multiple poles, direct
  Nova Carter into one support, measure whether the structure topples, capture
  Isaac images containing the generated structure, prepare the Reactor seed and
  prompt, and compare the recorded visual outcome after capture.
- Experimental interface: add a sandboxed scene-authoring adapter for approved
  USD/PhysX primitives, rigid bodies, collision shapes, materials, supports and
  joints. It may edit only a disposable experimental world or run-owned layer;
  it must not mutate the accepted source warehouse or expose shell, arbitrary
  filesystem paths, Docker arguments, or unrestricted host Python.
- Evidence to retain: model/tool trace, exact authored USD/layer and digest,
  source-world hashes before/after, load diagnostics, settling state, named
  bodies/joints, controller trace, contacts, final poses, failure classification,
  camera frames, Reactor prompt/recording, and comparison result.
- Acceptance criteria: generated scenes load without repair; start from a stable
  settled state; expose valid collisions and named measurable failure bodies;
  permit rover interaction; reproduce the classified outcome for the same seed;
  preserve source hashes; and complete the actual harness-owned Isaac → Reactor
  → comparison path. Lint, mocks, schema checks, or standalone USD parsing do
  not count as end-to-end acceptance.
- Cycle decision rule: keep reliable authoring operations, revise operations
  that need deterministic repair, and remove unsafe or repeatedly invalid
  operations. Do not merge this branch into the accepted harness until its real
  evidence satisfies the stated criteria and the user approves the result.
