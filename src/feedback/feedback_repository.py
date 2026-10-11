import json
import sqlite3
from pathlib import Path

from src.config import FEEDBACK_DB_PATH
from src.domain import FeedbackRecord, FeedbackReason

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT NOT NULL,
    model_recommendation TEXT NOT NULL,
    advisor_decision TEXT NOT NULL,
    override INTEGER NOT NULL,
    reason_codes TEXT NOT NULL,
    verified INTEGER NOT NULL,
    comment TEXT NOT NULL,
    immediate_reward REAL,
    outcome_reward REAL,
    model_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class FeedbackRepository:
    def __init__(self, db_path: Path = FEEDBACK_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add(self, record: FeedbackRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback (
                    application_id, model_recommendation, advisor_decision, override,
                    reason_codes, verified, comment, immediate_reward, outcome_reward,
                    model_version, policy_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.application_id,
                    record.model_recommendation.value,
                    record.advisor_decision.value,
                    int(record.override),
                    json.dumps([reason.value for reason in record.reason_codes]),
                    int(record.verified),
                    record.comment,
                    record.immediate_reward,
                    record.outcome_reward,
                    record.model_version,
                    record.policy_version,
                ),
            )

    def list_recent(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["reason_codes"] = [FeedbackReason(value) for value in json.loads(item["reason_codes"])]
            return_item = {**item, "override": bool(item["override"]), "verified": bool(item["verified"])}
            result.append(return_item)
        return result
