# Physical AI Failure Agents SDK

This repository contains one persistent, tool-using research agent for bounded
mine and native-USD warehouse danger experiments. The official OpenAI Agents SDK owns the model/tool
loop. Isaac Sim provides physics-grounded simulation evidence; Reactor provides
neural-world visual evidence; neither is presented as real-world truth.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
export OPENAI_API_KEY='<key>'
export AGENT_MODEL='<model available to this account>'

.venv/bin/python scripts/run_agents_researcher.py \
  'Investigate roof-support failure in the mine. Inspect the world and prior evidence, then use at most one new Isaac experiment.' \
  --experiment-budget 2
```

Reuse the printed campaign ID in a later process to resume the same persistent
SDK conversation:

```bash
.venv/bin/python scripts/run_agents_researcher.py \
  'Continue the research campaign.' \
  --campaign-id '<campaign-id>'
```

Run the dashboard:

```bash
.venv/bin/python scripts/run_dashboard.py \
  --database runs/experiments.sqlite3 --port 8000
```

Open `http://127.0.0.1:8000/agent` for campaigns,
`http://127.0.0.1:8000/reactor` for browser capture, and
`http://127.0.0.1:8000/` for recorded evidence.

## What is retained

- `src/harness/agent_runtime/`: agent definition, ten bounded tools, hooks,
  context, session, and Runner entrypoint.
- `src/harness/research/`: campaign persistence, world-prompt generation, and
  visual evidence assessment—not the former proposal/pipeline runtime.
- `src/harness/persistence/`, `pairing.py`, `comparison/`, and `media/`:
  authoritative evidence indexing and comparison.
- `scripts/run_mine_rover_experiment.py` and `assets/worlds/mine_v1/`: the
  deterministic Isaac Sim mine experiment boundary.
- `assets/worlds/warehouse_danger_v1/`: NVIDIA's native warehouse-with-forklifts
  composition plus the bounded, contact-driven rack-collapse demonstration.
- `assets/worlds/mine_v2_subt/`: the separate LTU SubT-derived perception
  prototype; the bounded route is accepted, while the model-visible agent tool
  remains an operator-only perception prototype.
- `src/harness/dashboard/`: the existing visual evidence and operator UI.

There is no shell, arbitrary Python, generic filesystem mutation, unrestricted
Docker, or Codex tool exposed to the model.

Read [docs/AGENTS_SDK_RUNTIME.md](docs/AGENTS_SDK_RUNTIME.md) for the exact
call graph, full agent instructions, tool surface, persistence model, verified
test evidence, and acceptance boundaries. The latest real browser/Isaac/agent
run is documented in
[runs/e2e-agents-sdk/20260824T090809Z/E2E_REPORT.md](runs/e2e-agents-sdk/20260824T090809Z/E2E_REPORT.md).
Read
[docs/MINE_ROVER_EXPERIMENT.md](docs/MINE_ROVER_EXPERIMENT.md) for the physical
world and experiment details.
Read [docs/WAREHOUSE_DANGER_DEMO.md](docs/WAREHOUSE_DANGER_DEMO.md) for the
agent's exact Isaac capabilities and the mandatory pre-recording sequence.
