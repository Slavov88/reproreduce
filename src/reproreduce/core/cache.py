from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .run import RunResult
from ..oracle.base import OracleResult


class CandidateCache:
    """Small SQLite cache for candidate executions and oracle classifications."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_hash TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                run_json TEXT NOT NULL,
                oracle_json TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def key(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def get(self, source: str) -> tuple[RunResult, OracleResult] | None:
        row = self.connection.execute(
            "SELECT run_json, oracle_json FROM candidates WHERE candidate_hash = ?",
            (self.key(source),),
        ).fetchone()
        if row is None:
            return None
        run_data = json.loads(row[0])
        oracle_data = json.loads(row[1])
        run_data["command"] = tuple(run_data["command"])
        return (
            RunResult(**run_data),
            OracleResult(**oracle_data),
        )

    def put(self, source: str, run: RunResult, result: OracleResult) -> None:
        run_data = asdict(run)
        run_data["command"] = list(run.command)
        self.connection.execute(
            "INSERT OR REPLACE INTO candidates VALUES (?, ?, ?, ?)",
            (self.key(source), source, json.dumps(run_data), json.dumps(asdict(result))),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "CandidateCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
