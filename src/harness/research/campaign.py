"""Durable campaign and operator-instruction state, separate from run artifacts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from enum import StrEnum
from pathlib import Path
from uuid import uuid4


class CampaignState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class IterationState(StrEnum):
    BUILDING_CONTEXT = "building_context"
    THINKING = "thinking"
    PROPOSAL_RECEIVED = "proposal_received"
    VALIDATING = "validating"
    COMPILED = "compiled"
    RUNNING_ISAAC = "running_isaac"
    RUNNING_REACTOR = "running_reactor"
    WAITING_FOR_ASSESSMENT = "waiting_for_assessment"
    COMPARING = "comparing"
    RECORDED = "recorded"
    FAILED = "failed"


class ResearchCampaignStore:
    """SQLite scientific memory; it never stores trajectory/media blobs."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS research_campaigns (
                    campaign_id TEXT PRIMARY KEY, objective TEXT NOT NULL, constraints_json TEXT NOT NULL,
                    status TEXT NOT NULL, experiment_budget INTEGER NOT NULL, experiments_used INTEGER NOT NULL DEFAULT 0,
                    current_iteration INTEGER NOT NULL DEFAULT 0, model_provider TEXT, model_name TEXT,
                    last_openai_response_id TEXT, capability_version TEXT,
                    simulator_metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS operator_instructions (
                    instruction_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, instruction TEXT NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(campaign_id) REFERENCES research_campaigns(campaign_id)
                );
                CREATE TABLE IF NOT EXISTS research_iterations (
                    iteration_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    state TEXT NOT NULL, context_json TEXT NOT NULL DEFAULT '{}', proposal_json TEXT,
                    compiled_json TEXT, isaac_run_id TEXT, reactor_run_id TEXT, plan_c_pair_id TEXT,
                    comparison_status TEXT, human_review_state TEXT, error TEXT, openai_response_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(campaign_id, ordinal), FOREIGN KEY(campaign_id) REFERENCES research_campaigns(campaign_id)
                );
                CREATE TABLE IF NOT EXISTS research_hypotheses (
                    hypothesis_id TEXT PRIMARY KEY, iteration_id TEXT NOT NULL, campaign_id TEXT NOT NULL,
                    hypothesis TEXT NOT NULL, rationale_summary TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(iteration_id) REFERENCES research_iterations(iteration_id),
                    FOREIGN KEY(campaign_id) REFERENCES research_campaigns(campaign_id)
                );
                CREATE TABLE IF NOT EXISTS research_proposals (
                    proposal_id TEXT PRIMARY KEY, iteration_id TEXT NOT NULL UNIQUE, campaign_id TEXT NOT NULL,
                    proposal_json TEXT NOT NULL, validation_state TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(iteration_id) REFERENCES research_iterations(iteration_id),
                    FOREIGN KEY(campaign_id) REFERENCES research_campaigns(campaign_id)
                );
                CREATE TABLE IF NOT EXISTS campaign_events (
                    event_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(campaign_id) REFERENCES research_campaigns(campaign_id)
                );
            """)
            self._ensure_column(connection, "research_campaigns", "capability_version", "TEXT")
            self._ensure_column(connection, "research_campaigns", "simulator_metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(connection, "research_iterations", "reactor_run_id", "TEXT")
            self._ensure_column(connection, "research_iterations", "plan_c_pair_id", "TEXT")
            self._ensure_column(connection, "research_iterations", "comparison_status", "TEXT")
            self._ensure_column(connection, "research_iterations", "human_review_state", "TEXT")
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create_campaign(self, objective: str, *, experiment_budget: int, constraints: dict | None = None,
                        model_provider: str | None = None, model_name: str | None = None,
                        capability_version: str | None = None, simulator_metadata: dict | None = None) -> str:
        if not objective.strip() or experiment_budget < 1:
            raise ValueError("objective and a positive experiment budget are required")
        campaign_id = str(uuid4())
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO research_campaigns(campaign_id, objective, constraints_json, status, experiment_budget, model_provider, model_name, capability_version, simulator_metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (campaign_id, objective.strip(), json.dumps(constraints or {}, sort_keys=True), CampaignState.CREATED,
                 experiment_budget, model_provider, model_name, capability_version,
                 json.dumps(simulator_metadata or {}, sort_keys=True)),
            )
            self._event(connection, campaign_id, "campaign_created", {})
            connection.commit()
        return campaign_id

    def add_instruction(self, campaign_id: str, instruction: str) -> str:
        if not instruction.strip(): raise ValueError("instruction must not be empty")
        instruction_id = str(uuid4())
        with closing(self._connect()) as connection:
            connection.execute("INSERT INTO operator_instructions(instruction_id, campaign_id, instruction) VALUES (?, ?, ?)", (instruction_id, campaign_id, instruction.strip()))
            self._event(connection, campaign_id, "operator_instruction_added", {"instruction_id": instruction_id})
            connection.commit()
        return instruction_id

    def pending_instructions(self, campaign_id: str) -> list[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT instruction FROM operator_instructions WHERE campaign_id = ? AND consumed = 0 ORDER BY created_at", (campaign_id,)).fetchall()
        return [row["instruction"] for row in rows]

    def consume_instructions(self, campaign_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("UPDATE operator_instructions SET consumed = 1 WHERE campaign_id = ? AND consumed = 0", (campaign_id,))
            connection.commit()

    def get_campaign(self, campaign_id: str) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM research_campaigns WHERE campaign_id = ?", (campaign_id,)).fetchone()
        if row is None: return None
        result = dict(row)
        result["constraints"] = json.loads(result.pop("constraints_json"))
        result["simulator_metadata"] = json.loads(result.pop("simulator_metadata_json") or "{}")
        return result

    def list_campaigns(self) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM research_campaigns ORDER BY updated_at DESC, campaign_id DESC").fetchall()
        return [self._campaign_row(row) for row in rows]

    def set_last_response_id(self, campaign_id: str, response_id: str | None) -> None:
        if response_id is None:
            return
        with closing(self._connect()) as connection:
            result = connection.execute("UPDATE research_campaigns SET last_openai_response_id = ?, updated_at = CURRENT_TIMESTAMP WHERE campaign_id = ?", (response_id, campaign_id))
            if result.rowcount != 1: raise KeyError(f"unknown campaign: {campaign_id}")
            connection.commit()

    def transition_campaign(self, campaign_id: str, state: CampaignState | str, *, reason: str | None = None) -> None:
        state = CampaignState(state)
        with closing(self._connect()) as connection:
            result = connection.execute("UPDATE research_campaigns SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE campaign_id = ?", (state, campaign_id))
            if result.rowcount != 1: raise KeyError(f"unknown campaign: {campaign_id}")
            self._event(connection, campaign_id, f"campaign_{state.value}", {"reason": reason} if reason else {})
            connection.commit()

    def begin_iteration(self, campaign_id: str, context: dict) -> str:
        iteration_id = str(uuid4())
        with closing(self._connect()) as connection:
            campaign = connection.execute("SELECT current_iteration FROM research_campaigns WHERE campaign_id = ?", (campaign_id,)).fetchone()
            if campaign is None: raise KeyError(f"unknown campaign: {campaign_id}")
            ordinal = campaign["current_iteration"] + 1
            connection.execute("INSERT INTO research_iterations(iteration_id, campaign_id, ordinal, state, context_json) VALUES (?, ?, ?, ?, ?)", (iteration_id, campaign_id, ordinal, IterationState.BUILDING_CONTEXT, json.dumps(context, sort_keys=True)))
            connection.execute("UPDATE research_campaigns SET current_iteration = ?, updated_at = CURRENT_TIMESTAMP WHERE campaign_id = ?", (ordinal, campaign_id))
            self._event(connection, campaign_id, "context_built", {"iteration_id": iteration_id})
            connection.commit()
        return iteration_id

    def transition_iteration(self, iteration_id: str, state: IterationState | str, *, proposal: dict | None = None, compiled: dict | None = None, response_id: str | None = None, error: str | None = None) -> None:
        state = IterationState(state)
        with closing(self._connect()) as connection:
            iteration = connection.execute("SELECT campaign_id FROM research_iterations WHERE iteration_id = ?", (iteration_id,)).fetchone()
            if iteration is None: raise KeyError(f"unknown iteration: {iteration_id}")
            connection.execute("UPDATE research_iterations SET state = ?, proposal_json = COALESCE(?, proposal_json), compiled_json = COALESCE(?, compiled_json), openai_response_id = COALESCE(?, openai_response_id), error = COALESCE(?, error), updated_at = CURRENT_TIMESTAMP WHERE iteration_id = ?", (state, json.dumps(proposal, sort_keys=True) if proposal else None, json.dumps(compiled, sort_keys=True) if compiled else None, response_id, error, iteration_id))
            if proposal:
                connection.execute("INSERT INTO research_hypotheses(hypothesis_id, iteration_id, campaign_id, hypothesis, rationale_summary) VALUES (?, ?, ?, ?, ?)", (str(uuid4()), iteration_id, iteration["campaign_id"], proposal["hypothesis"], proposal["rationale_summary"]))
                connection.execute("INSERT INTO research_proposals(proposal_id, iteration_id, campaign_id, proposal_json, validation_state) VALUES (?, ?, ?, ?, ?) ON CONFLICT(iteration_id) DO UPDATE SET proposal_json=excluded.proposal_json, validation_state=excluded.validation_state", (str(uuid4()), iteration_id, iteration["campaign_id"], json.dumps(proposal, sort_keys=True), state.value))
            self._event(connection, iteration["campaign_id"], state.value, {"iteration_id": iteration_id})
            connection.commit()

    def incomplete_iterations(self, campaign_id: str) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM research_iterations WHERE campaign_id = ? AND state NOT IN (?, ?) ORDER BY ordinal", (campaign_id, IterationState.RECORDED, IterationState.FAILED)).fetchall()
        return [dict(row) for row in rows]

    def get_iteration(self, iteration_id: str) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM research_iterations WHERE iteration_id = ?", (iteration_id,)).fetchone()
        if row is None: return None
        result = dict(row)
        for key in ("context_json", "proposal_json", "compiled_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key)) if result[key] else None
        return result

    def latest_iteration(self, campaign_id: str) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT iteration_id FROM research_iterations WHERE campaign_id = ? ORDER BY ordinal DESC LIMIT 1", (campaign_id,)).fetchone()
        return self.get_iteration(row["iteration_id"]) if row else None

    def record_isaac_run(self, iteration_id: str, run_id: str) -> None:
        with closing(self._connect()) as connection:
            iteration = connection.execute("SELECT campaign_id, isaac_run_id FROM research_iterations WHERE iteration_id = ?", (iteration_id,)).fetchone()
            if iteration is None: raise KeyError(f"unknown iteration: {iteration_id}")
            if iteration["isaac_run_id"] and iteration["isaac_run_id"] != run_id:
                raise ValueError("an iteration already has a different Isaac run ID")
            connection.execute("UPDATE research_iterations SET isaac_run_id = ?, updated_at = CURRENT_TIMESTAMP WHERE iteration_id = ?", (run_id, iteration_id))
            if not iteration["isaac_run_id"]:
                connection.execute("UPDATE research_campaigns SET experiments_used = experiments_used + 1, updated_at = CURRENT_TIMESTAMP WHERE campaign_id = ?", (iteration["campaign_id"],))
            self._event(connection, iteration["campaign_id"], "isaac_completed", {"iteration_id": iteration_id, "run_id": run_id})
            connection.commit()

    def record_plan_c_pair(self, campaign_id: str, isaac_run_id: str, pair_id: str) -> str:
        """Attach a prepared/finalized pair to the campaign iteration that owns its Isaac run."""
        with closing(self._connect()) as connection:
            iteration = connection.execute(
                "SELECT iteration_id, plan_c_pair_id FROM research_iterations "
                "WHERE campaign_id = ? AND isaac_run_id = ? ORDER BY ordinal DESC LIMIT 1",
                (campaign_id, isaac_run_id),
            ).fetchone()
            if iteration is None:
                raise KeyError("the Isaac run does not belong to this campaign")
            if iteration["plan_c_pair_id"] and iteration["plan_c_pair_id"] != pair_id:
                raise ValueError("the campaign iteration already has a different Plan C pair")
            connection.execute(
                "UPDATE research_iterations SET plan_c_pair_id = ?, state = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE iteration_id = ?",
                (pair_id, IterationState.WAITING_FOR_ASSESSMENT, iteration["iteration_id"]),
            )
            self._event(
                connection,
                campaign_id,
                "reactor_pair_prepared",
                {"iteration_id": iteration["iteration_id"], "run_id": isaac_run_id, "pair_id": pair_id},
            )
            connection.commit()
        return iteration["iteration_id"]

    def record_pair_comparison(
        self,
        campaign_id: str,
        pair_id: str,
        reactor_run_id: str,
        comparison_status: str,
    ) -> None:
        """Persist the completed Reactor/Plan C linkage without changing Isaac evidence."""
        with closing(self._connect()) as connection:
            iteration = connection.execute(
                "SELECT iteration_id FROM research_iterations WHERE campaign_id = ? AND plan_c_pair_id = ?",
                (campaign_id, pair_id),
            ).fetchone()
            if iteration is None:
                raise KeyError("the pair does not belong to this campaign")
            connection.execute(
                "UPDATE research_iterations SET reactor_run_id = ?, comparison_status = ?, state = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE iteration_id = ?",
                (
                    reactor_run_id,
                    comparison_status,
                    IterationState.RECORDED,
                    iteration["iteration_id"],
                ),
            )
            self._event(
                connection,
                campaign_id,
                "plan_c_comparison_recorded",
                {
                    "iteration_id": iteration["iteration_id"],
                    "pair_id": pair_id,
                    "reactor_run_id": reactor_run_id,
                    "comparison_status": comparison_status,
                },
            )
            connection.commit()

    def list_events(self, campaign_id: str) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM campaign_events WHERE campaign_id = ? ORDER BY rowid",
                (campaign_id,),
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in rows]

    def record_event(self, campaign_id: str, event_type: str, payload: dict | None = None) -> str:
        """Append an observable workflow event without storing model reasoning."""
        if not event_type.strip():
            raise ValueError("event_type must not be empty")
        event_id = str(uuid4())
        with closing(self._connect()) as connection:
            if connection.execute(
                "SELECT 1 FROM research_campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone() is None:
                raise KeyError(f"unknown campaign: {campaign_id}")
            connection.execute(
                "INSERT INTO campaign_events(event_id, campaign_id, event_type, payload_json) VALUES (?, ?, ?, ?)",
                (event_id, campaign_id, event_type.strip(), json.dumps(payload or {}, sort_keys=True)),
            )
            connection.commit()
        return event_id

    def has_equivalent_proposal(self, campaign_id: str, proposal: dict, *, exclude_iteration_id: str | None = None) -> bool:
        signature = json.dumps(
            {"task": proposal["experiment_intent"]["task"], "seed": proposal["experiment_intent"]["seed"], "parameter_changes": proposal["parameter_changes"]},
            sort_keys=True,
        )
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT proposal_json FROM research_proposals WHERE campaign_id = ? AND iteration_id != COALESCE(?, '')", (campaign_id, exclude_iteration_id)).fetchall()
        return any(
            json.dumps(
                {"task": (payload := json.loads(row["proposal_json"]))["experiment_intent"]["task"], "seed": payload["experiment_intent"]["seed"], "parameter_changes": payload["parameter_changes"]},
                sort_keys=True,
            ) == signature
            for row in rows
        )

    @staticmethod
    def _event(connection: sqlite3.Connection, campaign_id: str, event_type: str, payload: dict) -> None:
        connection.execute("INSERT INTO campaign_events(event_id, campaign_id, event_type, payload_json) VALUES (?, ?, ?, ?)", (str(uuid4()), campaign_id, event_type, json.dumps(payload, sort_keys=True)))

    @staticmethod
    def _campaign_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["constraints"] = json.loads(result.pop("constraints_json"))
        result["simulator_metadata"] = json.loads(result.pop("simulator_metadata_json") or "{}")
        return result

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
