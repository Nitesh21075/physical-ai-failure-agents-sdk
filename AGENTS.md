# Repository instructions

## Product boundary

This repository implements one `MineFailureResearcher` using the official
OpenAI Agents SDK. The SDK `Runner` owns model/tool iteration. Do not add a host
pipeline that prescribes inspect, Isaac, Reactor, or comparison order.

Before architecture or runtime changes, read:

1. `docs/AGENTS_SDK_RUNTIME.md`
2. `docs/MINE_ROVER_EXPERIMENT.md`
3. `assets/worlds/mine_v1/manifest.json`

## Authority and safety

- Isaac/PhysX is physics-grounded simulation evidence, not real-world truth.
- Reactor is neural-world visual evidence, not physical ground truth.
- A mismatch is a candidate discrepancy until reviewed.
- Never fabricate experiments, media, contact, motion, or measurements.
- Never add shell, arbitrary Python, generic file mutation, unrestricted
  Docker arguments, or unrestricted Omniverse access to the model tool surface.
- Keep Isaac tool arguments typed and bounded. One ordinary research step may
  claim at most one new Isaac run.
- Keep runtime context as local Python dependencies via `ToolContext`; do not
  serialize services or secrets into prompts.
- Never commit `.env`, API keys, `runs/`, simulator caches, or generated media.

## Persistence

- `agents.SQLiteSession(campaign_id, runs/agent_sessions.sqlite3)` is working
  conversation memory.
- `ResearchCampaignStore`, `ExperimentStore`, and `runs/` are authoritative
  scientific memory.
- Do not restore `previous_response_id` continuity or the deleted Responses API
  proposal pipeline.
- Hooks may persist lifecycle metadata and compact tool results, never hidden
  chain-of-thought.

## Iteration discipline and durable project memory

- Make material changes in bounded cycles: state the hypothesis and acceptance
  criteria, implement the smallest modular change, test it, and record a
  keep/revise/remove decision using evidence and user feedback.
- Remove rejected or superseded paths instead of accumulating inactive,
  unvalidated code. Keep experimental components behind replaceable module
  boundaries when practical.
- Append each material cycle, in UTC, to `docs/PROJECT_LOG.md`, including its
  timing, changes, tests and live evidence, failures, decision, and Git
  checkpoint. Never put secrets or generated media in the log.
- Maintain `docs/AGENTS_SDK_RUNTIME.md` as the curated statement of current
  architecture, capabilities, limitations, and acceptance status. Correct stale
  claims after each material cycle; the chronological log does not replace it.
- When authorized, create and push a recoverable Git checkpoint after an
  accepted cycle. A commit is not evidence that a simulator or model path works.

## Testing

Run focused tests and lint for every change. Unit tests do not establish live
acceptance: claims about SDK selection, Isaac physics, recording, Reactor, or
session resume require corresponding real evidence. Record failures honestly
in `docs/AGENTS_SDK_RUNTIME.md`; do not replace missing evidence with mocks.

Distinguish support checks (lint, unit tests, mocks, schema/import checks, and
ad-hoc diagnostics) from acceptance of the actual behavior. When the requested
claim concerns a program, function, integration, simulator, model, or generated
artifact, run the real in-scope entrypoint with its real dependencies when they
are available, inspect its output, and retain the observations or artifact/run
references. Never present a surrogate or debug check as the real result. If the
real path cannot run, label the claim unaccepted and record the concrete reason.

Inspect installed SDK and Isaac APIs before making version-specific changes.
