"""Create or resume one persistent MineFailureResearcher campaign step."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harness.agent_runtime import MineFailureResearchService


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction")
    parser.add_argument("--campaign-id")
    parser.add_argument(
        "--objective",
        default="Investigate roof-support failure in mine_v1 and compare Isaac with Reactor.",
    )
    parser.add_argument("--experiment-budget", type=int, default=3)
    parser.add_argument("--model")
    args = parser.parse_args()
    load_env(PROJECT_ROOT / ".env")
    model = args.model or os.environ.get("AGENT_MODEL") or os.environ.get("RESEARCH_MODEL")
    if not model:
        raise SystemExit("set AGENT_MODEL to a model available to this OpenAI account")
    service = MineFailureResearchService(PROJECT_ROOT, model=model)
    campaign_id = args.campaign_id or service.create_campaign(
        args.objective, experiment_budget=args.experiment_budget
    )
    output = await service.run_step(campaign_id, args.instruction)
    print(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "session_id": campaign_id,
                "result": output.model_dump(mode="json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(_main())
