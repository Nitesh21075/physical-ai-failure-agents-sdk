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
- Git: recorded by the cycle-1 checkpoint containing this entry.
