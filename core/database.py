"""
Database module for ingesting support_tickets.csv into SQLite,
providing secure read-only querying and schema validation.
"""

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

DB_FILE_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tickets.db")
CSV_FILE_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "support_tickets.csv")

FORBIDDEN_SQL_PATTERNS = [
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|REPLACE|TRUNCATE|EXEC|EXECUTE)\b",
    r"\b(ATTACH|DETACH|VACUUM|PRAGMA)\b",
]


class DatabaseManager:
    _instance: Optional["DatabaseManager"] = None

    def __init__(self, db_path: str = DB_FILE_DEFAULT, csv_path: str = CSV_FILE_DEFAULT):
        self.db_path = db_path
        self.csv_path = csv_path
        self._conn: Optional[sqlite3.Connection] = None
        self.init_db()

    @classmethod
    def get_instance(cls, db_path: str = DB_FILE_DEFAULT, csv_path: str = CSV_FILE_DEFAULT) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = DatabaseManager(db_path, csv_path)
        return cls._instance

    def get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init_db(self, force_reload: bool = False) -> None:
        """Loads support_tickets.csv into SQLite with derived analytical columns and indexes."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Check if table already populated
        if not force_reload:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets'")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM tickets")
                count = cursor.fetchone()[0]
                if count > 0:
                    return

        if not os.path.exists(self.csv_path):
            # Try alternate paths
            alt_path = os.path.join(os.getcwd(), "support_tickets.csv")
            if os.path.exists(alt_path):
                self.csv_path = alt_path
            else:
                raise FileNotFoundError(f"Support tickets CSV not found at {self.csv_path} or {alt_path}")

        df = pd.read_csv(self.csv_path)

        # Ensure datetime format and derived columns
        df["created_at"] = df["created_at"].astype(str)
        df["created_date"] = df["created_at"].str.slice(0, 10)
        df["created_year_month"] = df["created_at"].str.slice(0, 7)
        df["is_resolved"] = (df["status"] == "Resolved").astype(int)
        df["is_unresolved"] = (df["status"].isin(["Open", "Escalated"])).astype(int)

        cursor.execute("DROP TABLE IF EXISTS tickets")
        cursor.execute("""
            CREATE TABLE tickets (
                ticket_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                response_time_hrs REAL NOT NULL,
                resolution_time_hrs REAL,
                agent_id TEXT NOT NULL,
                customer_rating INTEGER,
                issue_summary TEXT NOT NULL,
                is_resolved INTEGER NOT NULL,
                is_unresolved INTEGER NOT NULL,
                created_date TEXT NOT NULL,
                created_year_month TEXT NOT NULL
            )
        """)

        # Insert rows
        records = df[[
            "ticket_id", "created_at", "category", "priority", "status",
            "response_time_hrs", "resolution_time_hrs", "agent_id", "customer_rating",
            "issue_summary", "is_resolved", "is_unresolved", "created_date", "created_year_month"
        ]].values.tolist()

        cursor.executemany("""
            INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)

        # Create indexes for optimal query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON tickets(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON tickets(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON tickets(priority)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent ON tickets(agent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON tickets(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_month ON tickets(created_year_month)")

        conn.commit()

    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Ensures SQL is purely read-only and safe."""
        cleaned_sql = re.sub(r"^```(?:sql)?\s*", "", sql.strip(), flags=re.IGNORECASE)
        cleaned_sql = re.sub(r"\s*```$", "", cleaned_sql).rstrip(";").rstrip("`").strip()
        if not cleaned_sql:
            return False, "Query cannot be empty"

        # Check if internal semicolon remains (stacked queries)
        if ";" in cleaned_sql:
            return False, "Stacked queries separated by ';' are not permitted."

        # Check for forbidden mutation keywords
        for pattern in FORBIDDEN_SQL_PATTERNS:
            if re.search(pattern, cleaned_sql, re.IGNORECASE):
                return False, f"Potentially unsafe or unsupported SQL operation detected: {pattern}"

        # Must start with SELECT or WITH
        first_word = cleaned_sql.split()[0].upper()
        if first_word not in ("SELECT", "WITH", "EXPLAIN"):
            return False, f"Only SELECT or WITH queries are allowed, got '{first_word}'"

        return True, None

    def execute_query(self, sql: str, params: tuple = (), max_rows: int = 200) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Executes a validated read-only SQL query safely."""
        is_valid, err = self.validate_sql(sql)
        if not is_valid:
            return [], err

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchmany(max_rows)
            column_names = [col[0] for col in cursor.description] if cursor.description else []
            result = [dict(zip(column_names, row)) for row in rows]
            return result, None
        except Exception as e:
            return [], str(e)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Returns comprehensive summary KPIs for the dashboard."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM tickets")
        total_tickets = cursor.fetchone()[0]

        cursor.execute("""
            SELECT
                SUM(CASE WHEN status = 'Open' THEN 1 ELSE 0 END) AS open_count,
                SUM(CASE WHEN status = 'Escalated' THEN 1 ELSE 0 END) AS escalated_count,
                SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) AS resolved_count,
                ROUND(AVG(response_time_hrs), 2) AS avg_response_time,
                ROUND(AVG(resolution_time_hrs), 2) AS avg_resolution_time,
                ROUND(AVG(customer_rating), 2) AS avg_rating
            FROM tickets
        """)
        row = cursor.fetchone()

        cursor.execute("SELECT category, COUNT(*) FROM tickets GROUP BY category ORDER BY COUNT(*) DESC")
        category_dist = dict(cursor.fetchall())

        cursor.execute("SELECT priority, COUNT(*) FROM tickets GROUP BY priority ORDER BY COUNT(*) DESC")
        priority_dist = dict(cursor.fetchall())

        cursor.execute("SELECT status, COUNT(*) FROM tickets GROUP BY status ORDER BY COUNT(*) DESC")
        status_dist = dict(cursor.fetchall())

        cursor.execute("""
            SELECT agent_id,
                   COUNT(*) as total,
                   SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) as resolved,
                   ROUND(AVG(customer_rating), 2) as avg_rating,
                   ROUND(AVG(resolution_time_hrs), 2) as avg_res_time
            FROM tickets
            GROUP BY agent_id
            ORDER BY resolved DESC
        """)
        agent_stats = [dict(zip(["agent_id", "total", "resolved", "avg_rating", "avg_res_time"], r)) for r in cursor.fetchall()]

        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM tickets")
        min_date, max_date = cursor.fetchone()

        return {
            "total_tickets": total_tickets,
            "open_count": row[0] or 0,
            "escalated_count": row[1] or 0,
            "resolved_count": row[2] or 0,
            "unresolved_count": (row[0] or 0) + (row[1] or 0),
            "avg_response_time_hrs": row[3] or 0.0,
            "avg_resolution_time_hrs": row[4] or 0.0,
            "avg_customer_rating": row[5] or 0.0,
            "category_distribution": category_dist,
            "priority_distribution": priority_dist,
            "status_distribution": status_dist,
            "agent_performance": agent_stats,
            "date_range": {"start": min_date, "end": max_date}
        }
