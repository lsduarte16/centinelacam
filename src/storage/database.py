"""SQLite event storage with retention management."""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from src.config import settings
from src.gate_logic.events import GateEvent

logger = logging.getLogger(__name__)


class EventDatabase:
    """Local SQLite database for event persistence."""

    def __init__(self):
        self.db_path = Path(settings.storage.db_path)
        self.retention_days = settings.storage.retention_days
        self._conn: sqlite3.Connection | None = None

    def initialize(self):
        """Create database and tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("Database initialized: %s", self.db_path)

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                track_id INTEGER,
                zone TEXT,
                frame_id INTEGER,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS gate_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                reason TEXT,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS counters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                hour INTEGER NOT NULL,
                person_in INTEGER DEFAULT 0,
                person_out INTEGER DEFAULT 0,
                vehicle_in INTEGER DEFAULT 0,
                vehicle_out INTEGER DEFAULT 0,
                UNIQUE(date, hour)
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_counters_date ON counters(date);
        """)
        self._conn.commit()

    def insert_event(self, event: GateEvent):
        """Store a gate event."""
        import json

        self._conn.execute(
            """INSERT INTO events (event_type, severity, description, track_id,
               zone, frame_id, metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_type.value,
                event.severity.value,
                event.description,
                event.track_id,
                event.zone,
                event.frame_id,
                json.dumps(event.metadata),
                event.timestamp.isoformat(),
            ),
        )
        self._conn.commit()
        self._update_counter(event)

    def _update_counter(self, event: GateEvent):
        """Update hourly counters."""
        now = event.timestamp
        date_str = now.strftime("%Y-%m-%d")
        hour = now.hour

        field = None
        if event.event_type.value == "person_entry":
            field = "person_in"
        elif event.event_type.value == "person_exit":
            field = "person_out"
        elif event.event_type.value == "vehicle_entry":
            field = "vehicle_in"
        elif event.event_type.value == "vehicle_exit":
            field = "vehicle_out"

        if field:
            self._conn.execute(
                f"""INSERT INTO counters (date, hour, {field})
                    VALUES (?, ?, 1)
                    ON CONFLICT(date, hour) DO UPDATE SET {field} = {field} + 1""",
                (date_str, hour),
            )
            self._conn.commit()

    def get_events(
        self, since: datetime | None = None, event_type: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Query events with optional filters."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_daily_summary(self, date: str | None = None) -> dict:
        """Get aggregated counters for a day."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        row = self._conn.execute(
            """SELECT
                COALESCE(SUM(person_in), 0) as total_person_in,
                COALESCE(SUM(person_out), 0) as total_person_out,
                COALESCE(SUM(vehicle_in), 0) as total_vehicle_in,
                COALESCE(SUM(vehicle_out), 0) as total_vehicle_out
            FROM counters WHERE date = ?""",
            (date,),
        ).fetchone()

        return dict(row) if row else {}

    def cleanup_old_records(self):
        """Remove records older than retention period."""
        cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
        deleted = self._conn.execute(
            "DELETE FROM events WHERE timestamp < ?", (cutoff,)
        ).rowcount
        self._conn.commit()
        if deleted:
            logger.info("Cleaned up %d old event records", deleted)

    def close(self):
        if self._conn:
            self._conn.close()
