# Real Agents SDK E2E report

Date: 2026-08-24 UTC

This run used only the public research CLI and the existing `/reactor` browser
page. No experiment, pairing, assessment, or persistence function was invoked
directly to advance the workflow.

## Result

| Acceptance boundary | Result | Evidence |
|---|---|---|
| REAL AGENTS TOOL TEST | PASS | Trace `trace_d7cfc61c074a4232be483dc16540842b`; the model called `inspect_mine_world` and `get_recent_experiments` before choosing an experiment. |
| REAL MINE PHYSICS TEST | PASS | Agent-created Isaac run `d1d7716a-a6ba-4ffc-83f2-1658bd39ecaa` used Isaac Sim 6.0.1, Nova Carter wheel targets, 21 real frames, and measured rover/support/beam poses. |
| AGENT-DRIVEN ISAAC TEST | PASS | The CLI called the SDK Runner; its event stream shows the model-selected `run_mine_roof_support_experiment` tool. No run script was invoked by the tester. |
| REAL ISAAC RECORDING TEST | PASS | Isaac replay decodes in Chromium at 320x180. |
| REAL REACTOR PAIR TEST | PASS | Pair `b67ac8dd-62b5-4ba5-a532-16c913bc0ca3`; real WebRTC Reactor run `3066ccb8-7019-4b31-b505-c9dada522729`; 2,432,837-byte WebM decodes at 1664x960. |
| AGENT SESSION RESUME TEST | PASS | A later CLI process reused the same campaign/SQLite session, called `get_pair_status` and `assess_and_compare_pair`, and described the prior experiment correctly. Resume trace: `trace_15363940233b4d57b0fb2b860e50e57d`. |
| SECOND RESEARCH ITERATION | SKIPPED | The campaign was deliberately created with experiment budget 1 to cap cost. The resumed agent proposed a different 600-step duration but did not execute it. |
| NEGATIVE TOOL BOUNDARY TESTS | PASS | Campaign budget was exhausted after one Isaac run; only the seven bounded tools were present. Existing real rejection evidence also covers out-of-range speed, second budget claim, and no shell tool. |

## Scientific outcome

The agent chose 0.8 m/s for 300 control steps. Isaac measured support lateral
displacement of 0.1521 m and beam vertical drop of approximately 0.0000093 m,
below the documented 0.8 m collapse criterion. The simulator-derived outcome
was stable. The Reactor capture was real, but the visual assessor returned
`observed=null`, confidence 0.0, and `inconclusive`; human review is required.
No agreement or candidate discrepancy was fabricated.

## Issues found and fixed

1. The Reactor UI enabled Start while the SDK was still in `waiting`, causing
   seed upload rejection. The client now waits for actual transport status
   `ready`.
2. The UI issued `start` before asynchronous conditions were confirmed. It now
   waits for `conditions_ready`.
3. Pair finalization incorrectly required `Scenario.environment == isaac_sim`,
   although mine runs truthfully use `mine_v1` and record the executor in
   `ExperimentRecord.backend`. Plan C now validates those two concepts at their
   correct boundaries.
4. Non-JSON HTTP failures were hidden by a client JSON parse error. The page
   now reports text or JSON server failures correctly.
5. The CLI was silent during the 4 minute 36 second Isaac tool call. It now
   streams model/tool lifecycle names without outputting hidden reasoning.
6. Per-run SDK token usage is now persisted as `agent_run_usage`. The first
   two turns predated this fix; the final status-only turn recorded 14,529
   tokens across two model requests.

The first failed finalization wrote a real but unindexed WebM. It was moved to
`failed_attempts/ff82c340-f24a-462d-a136-cac41f4f4255` for diagnosis and is not
presented as accepted evidence.

## Cost boundary

Three bounded OpenAI agent turns were made. An exact dollar total cannot be
reconstructed because usage persistence was added after the first two turns,
but the run was limited to one Isaac experiment and no optional second
iteration. Reactor generation was kept well under one minute; the published
LingBot World 2 rate was approximately $12/hour at test time, far below the
$10 Reactor cap.

## Remaining limitation

The E2E pipeline itself completed. The comparison outcome is scientifically
inconclusive because the mine seed is very dark and the short Reactor future
does not expose a confidently assessable roof-support event. Improving mine
camera lighting/composition and capturing a longer, deliberately aligned
future may make the evidence assessable, but doing so would consume another
Reactor session and was not attempted under this cost cap.
